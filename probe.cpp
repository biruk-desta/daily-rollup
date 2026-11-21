#include "parquet.hpp"

#include <algorithm>
#include <array>
#include <cerrno>
#include <cmath>
#include <csignal>
#include <cstdlib>
#include <fstream>
#include <initializer_list>
#include <numeric>
#include <sstream>
#include <string_view>

#ifdef __linux__
#include <sys/prctl.h>
#endif
#include <fcntl.h>
#include <poll.h>
#include <sys/resource.h>
#include <sys/wait.h>
#include <unistd.h>

namespace fs = std::filesystem;
using Ns = int64_t;
using Strings = std::vector<std::string>;
constexpr Ns second = 1'000'000'000;
constexpr Ns millisecond = 1'000'000;
constexpr int64_t missing = -1;

Ns now() {
    timespec time{};
    clock_gettime(CLOCK_MONOTONIC, &time);
    return time.tv_sec * second + time.tv_nsec;
}

std::string env(const char* name, const char* fallback) {
    const char* value = std::getenv(name);
    return value && *value ? value : fallback;
}

int64_t number(const char* name, int64_t maximum, int64_t minimum = 1) {
    const char* value = std::getenv(name);
    return value ? std::clamp<int64_t>(std::stoll(value), minimum, maximum) : maximum;
}

struct Knob { const char* label; std::initializer_list<std::pair<const char*, const char*>> options; };
constexpr Knob knobs[] = {
    {"cur", {{"SHOPPING_STAGING_CURSOR", "1"}}},
    {"s64e2", {{"SHOPPING_STAGING_CURSOR", "1"}, {"SHOPPING_STAGING_BYTES", "64"}, {"SHOPPING_EVENT_BUCKETS", "2048"}}},
    {"s64e4", {{"SHOPPING_STAGING_CURSOR", "1"}, {"SHOPPING_STAGING_BYTES", "64"}, {"SHOPPING_EVENT_BUCKETS", "4096"}}},
    {"s128e2", {{"SHOPPING_STAGING_CURSOR", "1"}, {"SHOPPING_STAGING_BYTES", "128"}, {"SHOPPING_EVENT_BUCKETS", "2048"}}},
    {"s128e4", {{"SHOPPING_STAGING_CURSOR", "1"}, {"SHOPPING_STAGING_BYTES", "128"}, {"SHOPPING_EVENT_BUCKETS", "4096"}}},
    {"sse", {{"SHOPPING_STAGING_CURSOR", "1"}, {"SHOPPING_STAGING_BYTES", "128"},
             {"SHOPPING_EVENT_BUCKETS", "4096"}, {"SHOPPING_STAGING_AVX", "0"}}},
    {"quote", {{"SHOPPING_CSV_QUOTE_FLAGS", "1"}}},
    {"tz", {{"SHOPPING_CSV_TZCNT", "1"}}},
    {"click", {{"SHOPPING_CLICK_INGEST", "1"}}},
    {"key", {{"SHOPPING_DATE_WIDE_KEY", "1"}}},
    {"date", {{"SHOPPING_DATE_PACKED_SLOT", "1"}}},
    {"alias", {{"SHOPPING_ALIAS_DIRECT_PROBE", "1"}}},
    {"tag", {{"SHOPPING_DEDUP_HIGH_TAG", "1"}}},
    {"person", {{"SHOPPING_PERSON_TABLE_DOUBLE", "1"}}},
};
constexpr size_t staging_knobs = 6;

struct Child {
    int status = -1;
    bool timed_out = false;
    Ns start = 0, end = 0;
    rusage usage{};
    std::string log;

    bool ok() const { return status == 0 && !timed_out; }
    std::string error() const {
        if (timed_out) return "timeout";
        if (status == -1) return "spawn";
        if (WIFSIGNALED(status)) return "sig" + std::to_string(WTERMSIG(status));
        return "exit" + std::to_string(WEXITSTATUS(status));
    }
};

Child run(const Strings& command, Ns deadline, unsigned threads = 8,
          bool trace = false, const Knob* knob = nullptr) {
    Child child;
    child.start = now();
    if (child.start >= deadline) {
        child.timed_out = true;
        return child;
    }
    std::vector<char*> args;
    for (const auto& arg : command) args.push_back(const_cast<char*>(arg.c_str()));
    args.push_back(nullptr);
    int pipefd[2];
    if (pipe(pipefd) != 0) return child;
    if (fcntl(pipefd[0], F_SETFL, O_NONBLOCK) < 0) {
        close(pipefd[0]);
        close(pipefd[1]);
        return child;
    }
    const pid_t parent = getpid();
    const pid_t pid = fork();
    if (pid == 0) {
        if (setpgid(0, 0) != 0) _exit(126);
#ifdef __linux__
        if (prctl(PR_SET_PDEATHSIG, SIGKILL) != 0) _exit(126);
#endif
        if (getppid() != parent) _exit(126);
        close(pipefd[0]);
        if (dup2(pipefd[1], STDERR_FILENO) < 0) _exit(126);
        close(pipefd[1]);
        setenv("NCPU", std::to_string(threads).c_str(), 1);
        for (const char* key : {"SHOPPING_TRACE", "SHOPPING_PERF", "SHOPPING_DETAIL", "SHOPPING_SCALAR",
                "SHOPPING_EVENT_BUCKETS", "SHOPPING_EVENT_BUCKET_ROWS", "SHOPPING_PERSON_PARTITIONS",
                "SHOPPING_PERSON_PARTITION_ROWS", "SHOPPING_CSV_PREFETCH", "SHOPPING_CSV_WRITES",
                "SHOPPING_HUGE", "SHOPPING_NT", "SHOPPING_UNMAP", "SHOPPING_RELEASE",
                "SHOPPING_RELEASE_METHOD", "SHOPPING_BUCKET_MIN_EVENTS", "SHOPPING_FORCE_SORT",
                "SHOPPING_STAGING_LINES", "SHOPPING_STAGING_PREFETCH", "SHOPPING_EVENT_TWO_LEVEL",
                "SHOPPING_ALIAS_LOAD", "SHOPPING_ALIAS_BITMAP", "SHOPPING_ALIAS_BATCH", "SHOPPING_FEW_CLICKS"})
            unsetenv(key);
        for (const auto& option : knobs)
            for (const auto& [key, value] : option.options) unsetenv(key);
        if (trace) setenv("SHOPPING_TRACE", "1", 1);
        if (knob)
            for (const auto& [key, value] : knob->options) setenv(key, value, 1);
        execv(args[0], args.data());
        _exit(127);
    }
    close(pipefd[1]);
    if (pid < 0) {
        close(pipefd[0]);
        return child;
    }
    setpgid(pid, pid);
    auto drain = [&] {
        char bytes[4096];
        for (int reads = 0; reads < 8; ++reads) {
            const ssize_t count = read(pipefd[0], bytes, sizeof(bytes));
            if (count <= 0) break;
            constexpr size_t log_limit = 32'768;
            child.log.append(bytes, std::min<size_t>(count, log_limit - child.log.size()));
        }
    };
    pollfd event{pipefd[0], POLLIN, 0};
    for (;;) {
        const pid_t done = wait4(pid, &child.status, WNOHANG, &child.usage);
        if (done == pid) break;
        if (done < 0 && errno != EINTR) {
            child.status = -1;
            break;
        }
        const Ns left = deadline - now();
        if (left <= 0) {
            child.timed_out = true;
            kill(-pid, SIGKILL);
            kill(pid, SIGKILL);
            while (wait4(pid, &child.status, 0, &child.usage) < 0 && errno == EINTR) {}
            break;
        }
        drain();
        if (event.revents & POLLHUP) event.fd = -1;
        poll(&event, 1, std::min<Ns>(event.fd < 0 ? 1 : 10, left / millisecond));
    }
    child.end = now();
    kill(-pid, SIGKILL);
    drain();
    close(pipefd[0]);
    return child;
}

using Values = std::vector<int64_t>;
std::string display(int64_t value) { return value == missing ? "-" : std::to_string(value); }

std::string join(const Values& values) {
    std::string result;
    for (auto value : values) {
        if (!result.empty()) result += '/';
        result += display(value);
    }
    return result;
}

enum Phase { Start, Mappings, Ingest, InputRelease, Dedup, CandidateRelease, Clicks, Sessions, Finish, PhaseCount };
constexpr std::array<std::string_view, PhaseCount> phases{
    "start", "mappings", "event_ingest_links_catalogs", "input_release", "event_dedup_route",
    "candidate_release", "click_scan_validation", "sessions_user_output", "finish"};

struct Sample {
    int64_t total = missing, exit = missing, ingest_cpu = missing;
    std::array<int64_t, PhaseCount> wall{};
    Values links;
    int64_t buckets = missing, partitions = missing;
    int64_t rss = missing, minor = missing, major = missing;
    int64_t memory_peak = missing, memory_current = missing;

    Sample() { wall.fill(missing); }
    bool complete() const {
        return total >= 0 && exit >= 0 && wall[Ingest] >= 0 && wall[Dedup] >= 0 &&
               wall[Clicks] >= 0 && wall[Sessions] >= 0;
    }
};

int64_t ms(Ns duration) { return std::max<Ns>(0, duration + millisecond / 2) / millisecond; }

Sample decode(const Child& child) {
    Sample sample;
    if (!child.end) return sample;
    sample.total = ms(child.end - child.start);
#ifdef __APPLE__
    constexpr int64_t to_mib = 1 << 20;
#else
    constexpr int64_t to_mib = 1 << 10;
#endif
    if (child.usage.ru_maxrss > 0) sample.rss = (child.usage.ru_maxrss + to_mib / 2) / to_mib;
    sample.minor = child.usage.ru_minflt;
    sample.major = child.usage.ru_majflt;
    std::istringstream log(child.log);
    for (std::string line; std::getline(log, line);) {
        std::istringstream row(line);
        std::string record, name;
        row >> record;
        if (record == "phase") {
            Ns at;
            double wall, cpu;
            if (!(row >> name >> at >> wall >> cpu)) continue;
            auto phase = std::find(phases.begin(), phases.end(), name);
            if (phase == phases.end()) continue;
            sample.wall[phase - phases.begin()] = std::lround(wall * 1000);
            if (name == "start") sample.wall[Start] = ms(at - child.start);
            if (name == "event_ingest_links_catalogs") sample.ingest_cpu = std::lround(cpu * 1000);
            if (name == "finish") sample.exit = ms(child.end - at);
        } else if (record == "partitions") {
            row >> sample.buckets >> sample.partitions;
        } else if (record == "links") {
            constexpr std::array<std::string_view, 8> keys{
                "wall_ms", "done_ms", "last_event_chunk_ms", "scan_ms", "init_ms",
                "insert_ms", "union_ms", "canonicalise_ms"};
            Values values;
            for (auto key : keys) {
                double value;
                if (!(row >> name >> value) || name != key) break;
                values.push_back(std::lround(value));
            }
            if (values.size() == keys.size())
                sample.links = {values[3], values[4], values[5], values[6], values[7], values[1], values[2]};
        }
    }
    return sample;
}

int64_t read_number(const fs::path& path) {
    int64_t value;
    return std::ifstream(path) >> value ? value : missing;
}

struct Cgroup {
    fs::path path;
    bool unified = true;

    Cgroup() {
#ifdef __linux__
        const fs::path root("/sys/fs/cgroup");
        std::ifstream file("/proc/self/cgroup");
        for (std::string line; std::getline(file, line);) {
            size_t first = line.find(':'), last = line.rfind(':');
            if (first == std::string::npos || first == last) continue;
            const std::string controllers = "," + line.substr(first + 1, last - first - 1) + ",";
            unified = controllers == ",,";
            if (!unified && controllers.find(",memory,") == std::string::npos) continue;
            const fs::path mount = unified ? root : root / "memory";
            const auto relative = fs::path(line.substr(last + 1)).relative_path();
            for (auto candidate : {mount / relative, mount}) {
                path = candidate;
                if (current() >= 0 || peak() >= 0) return;
            }
        }
        path.clear();
#endif
    }

    int64_t mib(const char* v2, const char* v1) const {
        if (path.empty()) return missing;
        auto value = read_number(path / (unified ? v2 : v1));
        return value < 0 ? missing : value / (1 << 20);
    }
    int64_t peak() const { return mib("memory.peak", "memory.max_usage_in_bytes"); }
    int64_t current() const { return mib("memory.current", "memory.usage_in_bytes"); }
};

struct Failures {
    std::vector<std::pair<std::string, unsigned>> entries;

    void add(const std::string& stage, const std::string& reason, const std::string& log = {}) {
        std::fprintf(stderr, "P6 %s %s %.200s\n", stage.c_str(), reason.c_str(), log.c_str());
        const auto code = stage + ':' + reason;
        for (auto& [seen, count] : entries) {
            if (seen != code) continue;
            ++count;
            return;
        }
        entries.emplace_back(code, 1);
    }

    std::string report() const {
        if (entries.empty()) return "-";
        std::string result;
        for (const auto& [code, count] : entries) {
            if (result.size() + code.size() > 110) return result + ",+";
            if (!result.empty()) result += ',';
            result += code;
            if (count > 1) result += '*' + std::to_string(count);
        }
        return result;
    }
};

struct Engine {
    const char* label;
    std::string binary;
};
using Samples = std::vector<Sample>;

struct Probe {
    const Ns deadline;
    const std::string generator;
    const fs::path trial;
    Failures& failures;

    bool generate(const fs::path& data, int64_t events, const char* seed,
                  Ns cutoff, const char* stage) const {
        const auto child = run({generator, "--seed", seed, "--events", std::to_string(events),
                                "--clicks", std::to_string(events / 4), "--threads", "8",
                                "--output", data.string()}, std::min(deadline, cutoff));
        if (child.ok()) return true;
        failures.add(stage, child.error(), child.log);
        fs::remove_all(data);
        return false;
    }

    Sample measure(const Engine& engine, const fs::path& data, Ns cutoff, unsigned threads,
                   const std::string& stage, Samples* samples, const Knob* knob = nullptr,
                   const Cgroup* cgroup = nullptr) const {
        const auto child = run({engine.binary, data.string(), trial.string()},
                               std::min(deadline, cutoff), threads, true, knob);
        auto sample = decode(child);
        if (cgroup && child.end) {
            sample.memory_peak = cgroup->peak();
            sample.memory_current = cgroup->current();
        }
        if (!child.ok()) failures.add(stage, child.error(), child.log);
        else if (!sample.complete()) failures.add(stage, "trace", child.log);
        else if (samples) {
            if (!knob && sample.links.empty())
                failures.add(stage, "links", child.log);
            samples->push_back(sample);
        }
        if (now() < deadline) fs::remove_all(trial);
        return sample;
    }
};

template<class Get> int64_t median(const Samples& samples, Get get) {
    Values values;
    for (const auto& sample : samples) {
        auto value = get(sample);
        if (value != missing) values.push_back(value);
    }
    if (values.empty()) return missing;
    std::sort(values.begin(), values.end());
    return std::midpoint(values[values.size() / 2], values[(values.size() - 1) / 2]);
}

Values headline(const Samples& samples) {
    Values result{median(samples, [](const Sample& sample) { return sample.total; })};
    for (size_t phase = 0; phase < PhaseCount; ++phase)
        result.push_back(median(samples, [=](const Sample& sample) { return sample.wall[phase]; }));
    result.push_back(median(samples, [](const Sample& sample) { return sample.exit; }));
    return result;
}

std::string overlap(const Samples& samples) {
    if (samples.empty()) return "-";
    Values result;
    for (size_t i = 0; i < 7; ++i)
        result.push_back(median(samples, [=](const Sample& sample) {
            return sample.links.size() == 7 ? sample.links[i] : missing;
        }));
    result.push_back(median(samples, [](const Sample& sample) { return sample.ingest_cpu; }));
    result.push_back(median(samples, [](const Sample& sample) { return sample.wall[Ingest]; }));
    return join(result);
}

std::string large(const Sample& sample) {
    if (sample.total == missing) return "-";
    return join({sample.total, sample.rss, sample.memory_peak, sample.memory_current,
        sample.minor, sample.major, sample.buckets, sample.partitions});
}

std::string sweep_time(const Samples& samples) {
    auto value = display(median(samples, [](const Sample& sample) { return sample.total; }));
    if (!samples.empty() && samples.size() < 3) value += '~' + std::to_string(samples.size());
    return value;
}

std::string load() {
    double avg;
    if (getloadavg(&avg, 1) != 1) return "-";
    char formatted[32];
    std::snprintf(formatted, sizeof(formatted), "%.2f", avg);
    return formatted;
}

std::string free_memory() {
    std::ifstream mem("/proc/meminfo");
    int64_t free = missing, available = missing;
    for (std::string line; std::getline(mem, line);) {
        std::istringstream row(line);
        std::string key;
        int64_t kib;
        if (!(row >> key >> kib)) continue;
        if (key == "MemFree:") free = kib / 1024;
        if (key == "MemAvailable:") available = kib / 1024;
        if (free >= 0 && available >= 0) break;
    }
    return display(free) + '/' + display(available);
}

fs::path temporary(const fs::path& parent, const char* name) {
    std::string path = (parent / name).string();
    if (!mkdtemp(path.data())) throw std::runtime_error("could not create " + path);
    return path;
}

using Rows = std::array<std::string, 5>;
void report(const fs::path& path, Rows rows, int64_t events) {
    constexpr parquet::Column schema[] = {{"user_id", parquet::Column::Text, 0},
        {"day", parquet::Column::Int32, 6}, {"sessions", parquet::Column::Int64},
        {"events", parquet::Column::Int64}, {"revenue_cents", parquet::Column::Int64},
        {"top_category", parquet::Column::Text, 0, true}};
    constexpr size_t split = 150, limit = 275;
    for (size_t i = 0; i < rows.size(); ++i) {
        std::fprintf(stderr, "P6 !0%zu %s\n", i, rows[i].c_str());
        if (rows[i].size() > limit) rows[i] = rows[i].substr(0, limit - 1) + '+';
    }
    parquet::File file(path, schema);
    file.append(rows.size(), [&](size_t column, parquet::Bytes& bytes) {
        if (column == 5) parquet::definitions(bytes, rows.size(), [](size_t) { return true; });
        for (size_t i = 0; i < rows.size(); ++i) {
            const auto& payload = rows[i];
            switch (column) {
                case 0: parquet::plain(bytes, "!0" + std::to_string(i) + " " + payload.substr(0, split)); break;
                case 1: parquet::little_endian(bytes, int32_t(10957 + i)); break;
                case 2: case 3: parquet::little_endian(bytes, int64_t(0)); break;
                case 4: parquet::little_endian(bytes, events); break;
                case 5: parquet::plain(bytes, payload.size() > split ? payload.substr(split) : "-"); break;
            }
        }
    });
    file.finish();
}

int main(int argc, char** argv) try {
    if (argc == 3 && std::string_view(argv[1]) == "--clean") { fs::remove_all(argv[2]); return 0; }
    const Ns start = now();
    alarm(150);
    const Ns deadline = start + number("PROBE_DIAG_SECONDS", 132) * second;
    const Ns answer_deadline = start + number("PROBE_DEV_SECONDS", 148) * second;
    const fs::path data = env("DATA_DIR", "/data"), output = env("OUTPUT_DIR", "/output");
    const std::string self = fs::absolute(argv[0]).string();
    const auto start_mem = free_memory(), start_load = load();
    const std::array<Engine, 3> engines{{{"A", env("PROBE_A", "/shopping-a")},
        {"B", env("PROBE_B", "/shopping-b")}, {"C", env("PROBE_C", "/shopping-c")}}};
    const auto& head = engines[1];
    Cgroup cgroup;
    Failures failures;
    std::array<Samples, engines.size()> comparison;
    std::array<Samples, std::size(knobs)> sweep;
    Sample large_b, large_a;
    int64_t events20 = 0, events_large = 0;
    enum Stage { Gen20, Compare, FirstSweep, Gen60, Gen40, LargeB, LargeA, Sweep, Clean, Dev, StageCount };
    std::array<Ns, StageCount> durations;
    durations.fill(missing);
    auto timed = [&](Stage stage, auto action) {
        const auto begin = now();
        action();
        durations[stage] = now() - begin;
    };

    fs::path work;
    try {
        work = temporary(env("SCRATCH_DIR", "/scratch"), "probe6-XXXXXX");
        const Probe probe{deadline, env("PROBE_GENERATOR", "/generate"), work / "trial", failures};
        const auto small = work / "small", big = work / "large";
        const auto target20 = number("PROBE_EVENTS20", 20'000'000, 1000);
        timed(Gen20, [&] {
            if (probe.generate(small, target20, "20261120", now() + 34 * second, "g20")) events20 = target20;
        });
        if (events20) timed(Compare, [&] {
            const auto cutoff = std::min(deadline, now() + 24 * second);
            for (size_t rep = 0; rep < 3; ++rep)
                for (size_t i = 0; i < engines.size() && now() < cutoff; ++i) {
                    const size_t at = (rep + i) % engines.size();
                    const auto& engine = engines[at];
                    probe.measure(engine, small, std::min(cutoff, now() + 8 * second), 8,
                                  engine.label, &comparison[at]);
                }
        });

        auto sweep_pass = [&](size_t round, Ns cutoff) {
            for (size_t i = 0; i < std::size(knobs) && now() < cutoff; ++i) {
                auto index = (i + round) % std::size(knobs);
                const auto& knob = knobs[index];
                probe.measure(head, small, std::min(cutoff, now() + 8 * second), 8,
                              knob.label, &sweep[index], &knob);
            }
        };
        if (events20 && now() < deadline)
            timed(FirstSweep, [&] { sweep_pass(0, std::min(deadline, now() + 28 * second)); });

        const auto target60 = number("PROBE_EVENTS60", 60'000'000, 1000);
        const auto target40 = number("PROBE_EVENTS40", 40'000'000, 1000);
        const auto fallback = std::clamp(2 * durations[Gen20] + 14 * second, 20 * second, 48 * second);
        const auto cutoff60 = std::min(now() + 54 * second, deadline - fallback);
        if (cutoff60 > now()) timed(Gen60, [&] {
            if (probe.generate(big, target60, "20261121", cutoff60, "g60")) events_large = target60;
        });
        else failures.add("g60", "budget");
        if (!events_large && now() < deadline) timed(Gen40, [&] {
            const auto reserve = std::min<Ns>(16 * second, (deadline - now()) / 3);
            if (probe.generate(big, target40, "20261121", deadline - reserve, "g40")) events_large = target40;
        });
        if (events_large) {
            timed(LargeB, [&] {
                const auto allowance = std::min<Ns>(14 * second, (deadline - now()) / 2);
                large_b = probe.measure(head, big, now() + allowance, 8, "Bbig", nullptr, nullptr, &cgroup);
            });
            timed(LargeA, [&] {
                large_a = probe.measure(engines.front(), big, now() + 14 * second, 8,
                                        "Abig", nullptr, nullptr, &cgroup);
            });
            if (now() < deadline) fs::remove_all(big);
        }
        if (events20 && now() < deadline) timed(Sweep, [&] {
            for (size_t rep = 1; rep < 3 && now() < deadline; ++rep) sweep_pass(rep, deadline);
        });
        if (events20 && std::any_of(comparison.begin(), comparison.end(),
                                    [](const auto& samples) { return samples.size() < 3; }))
            failures.add("cmp", "short");
        if (events20 && std::any_of(sweep.begin(), sweep.end(),
                                    [](const auto& samples) { return samples.size() < 3; }))
            failures.add("sw", "short");
    } catch (const std::exception& error) {
        failures.add("diag", "error", error.what());
    }

    auto clean = [&](const fs::path& path, Ns cutoff) {
        const auto child = run({self, "--clean", path.string()}, cutoff);
        if (!child.ok()) failures.add("clean", child.error(), child.log);
    };
    if (!work.empty()) timed(Clean, [&] {
        clean(work, std::min(start + 135 * second, now() + 3 * second));
    });

    fs::create_directories(output);
    const auto answer = temporary(output, ".probe6-answer-XXXXXX");
    Child dev;
    timed(Dev, [&] {
        dev = run({env("PROBE_DEV", "/shopping-dev"), data.string(), answer.string()}, answer_deadline);
    });
    if (!dev.ok()) {
        failures.add("dev", dev.error(), dev.log);
        clean(answer, start + 150 * second - 100 * millisecond);
        return 1;
    }
    try {
        Rows rows;
        for (size_t i = 0; i < engines.size(); ++i) {
            if (i) rows[0] += ' ';
            rows[0] += engines[i].label;
            const auto& samples = comparison[i];
            if (samples.empty()) rows[0] += "-0";
            else {
                rows[0] += join(headline(samples));
                if (samples.size() < 3) rows[0] += '~' + std::to_string(samples.size());
            }
        }
        rows[1] = "base=" + sweep_time(comparison[1]);
        for (size_t i = 0; i < std::size(knobs); ++i) {
            auto& row = rows[i < staging_knobs ? 1 : 2];
            if (!row.empty()) row += ' ';
            row += std::string(knobs[i].label) + '=' + sweep_time(sweep[i]);
        }
        rows[3] = "L=" + overlap(comparison[1]) + " n=" + std::to_string(events_large) +
                  " B=" + large(large_b) + " A=" + large(large_a);
        rows[4] = "m=" + start_mem + ':' + free_memory() + " la=" + start_load + '/' + load() + " s=";
        for (size_t i = 0; i < durations.size(); ++i) {
            if (i) rows[4] += '/';
            if (durations[i] < 0) rows[4] += '-';
            else {
                auto tenths = (durations[i] + second / 20) / (second / 10);
                rows[4] += std::to_string(tenths / 10) + '.' + std::to_string(tenths % 10);
            }
        }
        rows[4] += " f=" + failures.report();
        report(answer / "user_daily" / "!probe6.parquet", rows, events20);
        fs::rename(answer / "campaign_daily", output / "campaign_daily");
        fs::rename(answer / "user_daily", output / "user_daily");
        fs::remove(answer);
    } catch (...) {
        clean(answer, start + 150 * second - 100 * millisecond);
        throw;
    }
    return 0;
} catch (const std::exception& error) {
    std::fprintf(stderr, "P6 %.200s\n", error.what());
    return 1;
}
