#include "parquet.hpp"

#include <algorithm>
#include <array>
#include <cerrno>
#include <cmath>
#include <csignal>
#include <cstdlib>
#include <fstream>
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

struct Setting {
    const char* key = nullptr;
    const char* value = nullptr;
};

Child run(const Strings& command, Ns deadline, const std::string& threads, Setting setting = {}) {
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
        setenv("NCPU", threads.c_str(), 1);
        for (const char* key : {"SHOPPING_TRACE", "SHOPPING_PERF", "SHOPPING_DETAIL", "SHOPPING_SCALAR",
                "SHOPPING_EVENT_BUCKETS", "SHOPPING_EVENT_BUCKET_ROWS", "SHOPPING_PERSON_PARTITIONS",
                "SHOPPING_PERSON_PARTITION_ROWS", "SHOPPING_CSV_PREFETCH", "SHOPPING_CSV_WRITES",
                "SHOPPING_HUGE", "SHOPPING_NT", "SHOPPING_UNMAP", "SHOPPING_RELEASE",
                "SHOPPING_RELEASE_METHOD", "SHOPPING_BUCKET_MIN_EVENTS", "SHOPPING_FORCE_SORT",
                "SHOPPING_ALIAS_BATCH", "SHOPPING_ALIAS_BITMAP", "SHOPPING_ALIAS_DIRECT_PROBE",
                "SHOPPING_ALIAS_LOAD", "SHOPPING_CLICK_INGEST", "SHOPPING_CSV_QUOTE_FLAGS",
                "SHOPPING_CSV_TZCNT", "SHOPPING_DATE_PACKED_SLOT", "SHOPPING_DATE_WIDE_KEY",
                "SHOPPING_DEDUP_HIGH_TAG", "SHOPPING_EVENT_TWO_LEVEL", "SHOPPING_FEW_CLICKS",
                "SHOPPING_PERSON_TABLE_DOUBLE", "SHOPPING_STAGING_AVX", "SHOPPING_STAGING_BYTES",
                "SHOPPING_STAGING_CURSOR", "SHOPPING_STAGING_LINES", "SHOPPING_STAGING_PREFETCH",
                "SHOPPING_ROW_LANES", "SHOPPING_ROW_TEMPLATE", "SHOPPING_RECORD_BYTES", "SHOPPING_SESSION_FAST"})
            unsetenv(key);
        if (setting.key) setenv(setting.key, setting.value, 1);
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
            // A child in uninterruptible I/O can outlive SIGKILL, especially during sync.
            const Ns reap_until = now() + 50 * millisecond;
            for (;;) {
                const pid_t killed = wait4(pid, &child.status, WNOHANG, &child.usage);
                if (killed == pid || (killed < 0 && errno != EINTR) || now() >= reap_until) break;
                poll(nullptr, 0, 1);
            }
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

template<class Range> std::string join(const Range& values) {
    std::string result;
    for (auto value : values) {
        if (!result.empty()) result += '/';
        result += display(value);
    }
    return result;
}

// Five data bits per character, with the high bit marking a following character.
void pack(std::string& result, int64_t value) {
    constexpr std::string_view alphabet = "0123456789ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz_~";
    if (value == missing) {
        result += '-';
        return;
    }
    auto bits = static_cast<uint64_t>(value);
    while (bits >= 32) {
        result += alphabet[(bits & 31) | 32];
        bits >>= 5;
    }
    result += alphabet[bits];
}

template<class Range> void pack_all(std::string& result, const Range& values) {
    for (auto value : values) pack(result, value);
}

enum Phase {
    Start, Mappings, Ingest, InputRelease, Dedup, CandidateRelease, Clicks, Sessions,
    Finish, ExitRelease, PhaseCount
};
constexpr std::array<std::string_view, PhaseCount> phases{
    "start", "mappings", "event_ingest_links_catalogs", "input_release", "event_dedup_route",
    "candidate_release", "click_scan_validation", "sessions_user_output", "finish", "exit_release"};

constexpr std::array<std::string_view, 4> vm_keys{
    "compact_stall", "thp_fault_alloc", "thp_fault_fallback", "allocstall"};
using Vmstat = std::array<int64_t, vm_keys.size()>;

Vmstat vmstat() {
    Vmstat values;
    values.fill(missing);
    int64_t zone_stalls = missing;
    std::ifstream file("/proc/vmstat");
    std::string key;
    int64_t value;
    while (file >> key >> value) {
        auto match = std::find(vm_keys.begin(), vm_keys.end(), key);
        if (match != vm_keys.end()) values[match - vm_keys.begin()] = value;
        else if (key.starts_with("allocstall_")) zone_stalls = std::max<int64_t>(0, zone_stalls) + value;
    }
    if (values[3] == missing) values[3] = zone_stalls;
    return values;
}

Vmstat delta(const Vmstat& before, const Vmstat& after) {
    Vmstat values;
    for (size_t i = 0; i < values.size(); ++i)
        values[i] = before[i] < 0 || after[i] < before[i] ? missing : after[i] - before[i];
    return values;
}

struct Sample {
    int64_t total = missing, exit = missing, user = missing, system = missing;
    std::array<int64_t, PhaseCount> wall{};
    Vmstat vm{};
    int64_t rss = missing, minor = missing;
    bool valid = false;

    Sample() {
        wall.fill(missing);
        vm.fill(missing);
    }
    bool complete() const {
        return total >= 0 && exit >= 0 && wall[Ingest] >= 0 && wall[Dedup] >= 0 &&
               wall[Clicks] >= 0 && wall[Sessions] >= 0;
    }
};

int64_t ms(Ns duration) { return std::max<Ns>(0, duration + millisecond / 2) / millisecond; }
int64_t ms(timeval cpu) { return ms(cpu.tv_sec * second + cpu.tv_usec * 1000); }

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
    if (child.status != -1) {
        sample.user = ms(child.usage.ru_utime);
        sample.system = ms(child.usage.ru_stime);
        sample.minor = child.usage.ru_minflt;
    }
    std::istringstream log(child.log);
    for (std::string line; std::getline(log, line);) {
        std::istringstream row(line);
        std::string record, name;
        row >> record;
        if (record != "phase") continue;
        Ns at;
        double wall;
        if (!(row >> name >> at >> wall)) continue;
        auto phase = std::find(phases.begin(), phases.end(), name);
        if (phase == phases.end()) continue;
        sample.wall[phase - phases.begin()] = std::lround(wall * 1000);
        if (name == "start") sample.wall[Start] = ms(at - child.start);
        if (name == "finish") sample.exit = ms(child.end - at);
    }
    return sample;
}

struct Failures {
    Strings entries;

    void add(const std::string& stage, const std::string& reason, const std::string& log = {}) {
        std::fprintf(stderr, "P9 %s %s %.200s\n", stage.c_str(), reason.c_str(), log.c_str());
        entries.push_back(stage + ':' + reason);
    }

    std::string report(size_t limit) const {
        if (entries.empty()) return "-";
        std::string result;
        for (const auto& code : entries) {
            if (result.size() + code.size() + 2 > limit) return result + '+';
            if (!result.empty()) result += ',';
            result += code;
        }
        return result;
    }
};

constexpr std::array<size_t, 6> order{0, 1, 1, 0, 0, 1};
using Trials = std::array<Sample, order.size()>;

Sample measure(const std::string& engine, const fs::path& data, const fs::path& output,
               Ns cutoff, const std::string& threads, const std::string& stage, Failures& failures) {
    const auto before = vmstat();
    const auto child = run({engine, data.string(), output.string()}, cutoff, threads, {"SHOPPING_TRACE", "1"});
    const auto after = vmstat();
    auto sample = decode(child);
    if (child.end) sample.vm = delta(before, after);
    sample.valid = child.ok() && sample.complete();
    if (!child.ok()) failures.add(stage, child.error(), child.log);
    else if (!sample.complete()) failures.add(stage, "trace", child.log);
    std::fprintf(stderr, "P9 run %s total=%s phase=%s exit=%s cpu=%s rss/minor=%s vm=%s\n",
                 stage.c_str(), display(sample.total).c_str(),
                 join(sample.wall).c_str(), display(sample.exit).c_str(),
                 join(std::array{sample.user, sample.system}).c_str(),
                 join(std::array{sample.rss, sample.minor}).c_str(), join(sample.vm).c_str());
    return sample;
}

template<class Get> int64_t median(const std::vector<const Sample*>& samples, Get get) {
    Values values;
    for (const auto* sample : samples) values.push_back(get(*sample));
    if (values.empty()) return missing;
    std::sort(values.begin(), values.end());
    return std::midpoint(values[values.size() / 2], values[(values.size() - 1) / 2]);
}

Values headline(const std::vector<const Sample*>& samples) {
    Values result{median(samples, [](const Sample& sample) { return sample.total; })};
    for (auto phase : {Ingest, Dedup, Clicks, Sessions})
        result.push_back(median(samples, [=](const Sample& sample) { return sample.wall[phase]; }));
    result.push_back(median(samples, [](const Sample& sample) { return sample.exit; }));
    return result;
}

std::array<int64_t, 2> dirty_pages() {
    std::ifstream mem("/proc/meminfo");
    int64_t dirty = missing, writeback = missing;
    for (std::string line; std::getline(mem, line);) {
        std::istringstream row(line);
        std::string key;
        int64_t kib;
        if (!(row >> key >> kib)) continue;
        if (key == "Dirty:") dirty = kib;
        if (key == "Writeback:") writeback = kib;
        if (dirty >= 0 && writeback >= 0) break;
    }
    return {dirty, writeback};
}

std::array<int64_t, 4> wait_writeback(const std::string& self, Ns deadline, const std::string& threads,
                                     Failures& failures) {
    const auto begin = now();
    const auto cutoff = std::min(deadline, begin + 20 * second);
    const auto dirty_before = dirty_pages()[0];
    const auto child = run({self, "--sync"}, cutoff, threads);
    if (!child.ok()) failures.add("sync", child.error(), child.log);
    auto pages = dirty_pages();
    constexpr int64_t threshold = 64 * 1024;
    while (child.ok() && std::min(pages[0], pages[1]) >= 0 &&
           std::max(pages[0], pages[1]) >= threshold && now() < cutoff) {
        poll(nullptr, 0, std::clamp<Ns>((cutoff - now()) / millisecond, 0, 100));
        pages = dirty_pages();
    }
#ifdef __linux__
    if (std::min(pages[0], pages[1]) < 0) failures.add("wb", "meminfo");
    else if (std::max(pages[0], pages[1]) >= threshold) failures.add("wb", "high");
#endif
    return {ms(now() - begin), dirty_before, pages[0], pages[1]};
}

fs::path temporary(const fs::path& parent, const char* name) {
    std::string path = (parent / name).string();
    if (!mkdtemp(path.data())) throw std::runtime_error("could not create " + path);
    return path;
}

using Rows = std::array<std::string, 5>;
constexpr size_t row_limit = 276;  // Leave four characters for "!0n ".
constexpr std::array options{Setting{}, Setting{"SHOPPING_ROW_LANES", "2"},
    Setting{"SHOPPING_ROW_TEMPLATE", "1"}, Setting{"SHOPPING_RECORD_BYTES", "24"},
    Setting{"SHOPPING_SESSION_FAST", "1"}};

enum Stage { Generate60, Writeback, Rotate, Sweep, Remove60, Generate20, Compare20,
             Cleanup, Dev, Diagnostics, StageCount };

struct Readings {
    Trials large;
    std::array<int64_t, options.size()> sweep;
    std::array<int64_t, 2> small;
    std::array<int64_t, 2> events{};
    std::array<int64_t, 4> writeback;
    std::array<int64_t, StageCount> durations;

    Readings() {
        sweep.fill(missing);
        small.fill(missing);
        writeback.fill(missing);
        durations.fill(missing);
    }
};

// Continue a packed field in a later row if a large reading would cross the row limit.
void append_packed(std::string& row, std::string_view prefix, std::string_view& packed) {
    if (packed.empty() || row.size() + prefix.size() >= row_limit) return;
    const size_t count = std::min(packed.size(), row_limit - row.size() - prefix.size());
    row += prefix;
    row += packed.substr(0, count);
    packed.remove_prefix(count);
}

Rows summarize(const Readings& readings, Failures& failures) {
    Rows rows;
    for (size_t engine = 0; engine < 2; ++engine) {
        std::vector<const Sample*> valid;
        for (size_t i = 0; i < order.size(); ++i) {
            if (order[i] == engine && readings.large[i].valid) valid.push_back(&readings.large[i]);
        }
        if (!rows[0].empty()) rows[0] += ' ';
        rows[0] += std::string(1, "LS"[engine]) + std::to_string(valid.size()) + '=' + join(headline(valid));
    }

    std::array<int64_t, order.size()> totals;
    std::string walls, usage;
    for (size_t i = 0; i < totals.size(); ++i) {
        const auto& sample = readings.large[i];
        totals[i] = sample.total;
        pack_all(walls, sample.wall);
        pack(walls, sample.exit);
        pack_all(usage, std::array{sample.user, sample.system, sample.rss, sample.minor});
        pack_all(usage, sample.vm);
    }
    rows[0] += " t=" + join(totals);
    rows[2] = "O=" + join(readings.sweep);
    rows[3] = "L/S=" + join(readings.small) + " n=" + join(readings.events) +
              " g=" + join(std::array{readings.durations[Generate60], readings.durations[Generate20]}) +
              " w=" + join(readings.writeback);
    rows[4] = "s=" + join(readings.durations);

    std::string_view packed_walls = walls, packed_usage = usage;
    append_packed(rows[0], " p=", packed_walls);
    append_packed(rows[3], " p=", packed_walls);
    append_packed(rows[1], "", packed_usage);
    append_packed(rows[2], " x=", packed_usage);
    append_packed(rows[3], " x=", packed_usage);
    if (!packed_walls.empty() || !packed_usage.empty()) failures.add("payload", "limit");
    rows[4] += " f=";
    rows[4] += failures.report(row_limit - std::min(row_limit, rows[4].size()));
    return rows;
}

void report(const fs::path& path, const Rows& rows, int64_t events) {
    constexpr parquet::Column schema[] = {{"user_id", parquet::Column::Text, 0},
        {"day", parquet::Column::Int32, 6}, {"sessions", parquet::Column::Int64},
        {"events", parquet::Column::Int64}, {"revenue_cents", parquet::Column::Int64},
        {"top_category", parquet::Column::Text, 0, true}};
    constexpr size_t split = 150;
    for (size_t i = 0; i < rows.size(); ++i) {
        std::fprintf(stderr, "P9 !0%zu %s\n", i, rows[i].c_str());
    }
    parquet::File file(path, schema);
    file.append(rows.size(), [&](size_t column, parquet::Bytes& bytes) {
        if (column == 5) parquet::definitions(bytes, rows.size(), [](size_t) { return true; });
        for (size_t i = 0; i < rows.size(); ++i) {
            const auto& payload = rows[i];
            const size_t boundary = payload.size() == split + 1 && payload.back() == '-' ? split - 1 : split;
            switch (column) {
                case 0: parquet::plain(bytes, "!0" + std::to_string(i) + " " + payload.substr(0, boundary)); break;
                case 1: parquet::little_endian(bytes, int32_t(10957 + i)); break;
                case 2: case 3: parquet::little_endian(bytes, int64_t(0)); break;
                case 4: parquet::little_endian(bytes, events); break;
                case 5: parquet::plain(bytes, payload.size() > boundary ? payload.substr(boundary) : "-"); break;
            }
        }
    });
    file.finish();
}

int main(int argc, char** argv) try {
    if (argc == 2 && std::string_view(argv[1]) == "--sync") {
#ifdef __linux__
        sync();
#endif
        return 0;
    }
    if (argc == 3 && std::string_view(argv[1]) == "--clean") {
        fs::remove_all(argv[2]);
        return 0;
    }
    const Ns start = now();
    alarm(150);
    const Ns deadline = start + number("PROBE_DIAG_SECONDS", 132) * second;
    const Ns answer_deadline = start + number("PROBE_DEV_SECONDS", 148) * second;
    const fs::path data = env("DATA_DIR", "/data"), output = env("OUTPUT_DIR", "/output");
    const std::string self = fs::absolute(argv[0]).string();
    const auto threads = std::to_string(number("PROBE_THREADS", 8));
    const std::array engines{env("PROBE_L", "/shopping-l"), env("PROBE_S", "/shopping-s")};
    const auto option_engine = env("PROBE_O", "/shopping-o");
    Failures failures;
    Readings readings;
    auto clean = [&](const fs::path& path, Ns cutoff, const std::string& stage) {
        const auto child = run({self, "--clean", path.string()}, std::min(cutoff, now() + 3 * second), threads);
        if (!child.ok()) failures.add(stage, child.error(), child.log);
        return child.ok();
    };

    fs::path work;
    try {
        work = temporary(env("SCRATCH_DIR", "/scratch"), "probe9-XXXXXX");
        const auto large = work / "large", small = work / "small";
        const auto generator = env("PROBE_GENERATOR", "/generate");
        auto generate = [&](const fs::path& path, int64_t events, const char* seed, Stage timer, const char* stage) {
            const auto child = run({generator, "--seed", seed, "--events", std::to_string(events),
                "--clicks", std::to_string(events / 4), "--threads", threads, "--output", path.string()},
                std::min(deadline, now() + 54 * second), threads);
            if (child.end) readings.durations[timer] = ms(child.end - child.start);
            if (!child.ok()) failures.add(stage, child.error(), child.log);
            return child.ok() ? events : 0;
        };
        auto total_only = [&](const std::string& engine, const fs::path& dataset, const fs::path& trial,
                              const std::string& stage, Setting setting = {}) {
            const auto child = run({engine, dataset.string(), trial.string()},
                std::min(deadline, now() + 14 * second), threads, setting);
            if (!child.ok()) failures.add(stage, child.error(), child.log);
            if (now() < deadline) clean(trial, deadline, "c" + stage);
            return child.ok() ? ms(child.end - child.start) : missing;
        };

        readings.events[0] = generate(large, number("PROBE_EVENTS60", 60'000'000, 1000),
                                      "20261123", Generate60, "g60");
        if (readings.events[0]) {
            readings.writeback = wait_writeback(self, deadline, threads, failures);
            readings.durations[Writeback] = readings.writeback[0];
            auto begin = now();
            for (size_t i = 0; i < order.size(); ++i) {
                const auto stage = "r" + std::to_string(i + 1) + "LS"[order[i]];
                if (now() >= deadline) {
                    failures.add(stage, "budget");
                    break;
                }
                const auto trial = work / stage;
                readings.large[i] = measure(engines[order[i]], large, trial,
                    std::min(deadline, now() + 14 * second), threads, stage, failures);
                if (now() < deadline) clean(trial, deadline, "c" + stage);
            }
            readings.durations[Rotate] = ms(now() - begin);
            begin = now();
            for (size_t i = 0; i < options.size(); ++i) {
                const auto stage = "o" + std::to_string(i);
                if (now() >= deadline) {
                    failures.add(stage, "budget");
                    break;
                }
                readings.sweep[i] = total_only(option_engine, large, work / stage, stage, options[i]);
            }
            readings.durations[Sweep] = ms(now() - begin);
        }
        if (now() < deadline) {
            const auto begin = now();
            const bool removed = clean(large, deadline, "c60");
            readings.durations[Remove60] = ms(now() - begin);
            if (removed && now() < deadline) {
                readings.events[1] = generate(small, number("PROBE_EVENTS20", 20'000'000, 1000),
                                              "20261120", Generate20, "g20");
                if (readings.events[1]) {
                    const auto compare = now();
                    for (size_t i = 0; i < engines.size(); ++i) {
                        const auto stage = std::string("s") + "LS"[i];
                        if (now() >= deadline) {
                            failures.add(stage, "budget");
                            break;
                        }
                        readings.small[i] = total_only(engines[i], small, work / stage, stage);
                    }
                    readings.durations[Compare20] = ms(now() - compare);
                }
            } else if (removed) {
                failures.add("g20", "budget");
            }
        } else {
            failures.add("g20", "budget");
        }
    } catch (const std::exception& error) {
        failures.add("diag", "error", error.what());
    }
    readings.durations[Diagnostics] = ms(now() - start);
    if (!work.empty()) {
        const auto begin = now();
        clean(work, start + 135 * second, "clean");
        readings.durations[Cleanup] = ms(now() - begin);
    }

    fs::create_directories(output);
    const auto answer = temporary(output, ".probe9-answer-XXXXXX");
    const auto dev = run({env("PROBE_DEV", "/shopping-dev"), data.string(), answer.string()}, answer_deadline, threads);
    if (dev.end) readings.durations[Dev] = ms(dev.end - dev.start);
    if (!dev.ok()) {
        failures.add("dev", dev.error(), dev.log);
        clean(answer, start + 150 * second - 100 * millisecond, "cdev");
        return 1;
    }
    try {
        report(answer / "user_daily" / "!probe9.parquet", summarize(readings, failures), readings.events[0]);
        fs::rename(answer / "campaign_daily", output / "campaign_daily");
        fs::rename(answer / "user_daily", output / "user_daily");
        fs::remove(answer);
    } catch (...) {
        clean(answer, start + 150 * second - 100 * millisecond, "cdev");
        throw;
    }
    return 0;
} catch (const std::exception& error) {
    std::fprintf(stderr, "P9 %.200s\n", error.what());
    return 1;
}
