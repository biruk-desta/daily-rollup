#include "parquet.hpp"

#include <algorithm>
#include <array>
#include <cerrno>
#include <csignal>
#include <cstdlib>
#include <fstream>
#include <sstream>
#include <utility>

#ifdef __linux__
#include <sched.h>
#endif

#include "isa.hpp"
#include <string>

#include <fcntl.h>
#include <poll.h>
#include <sys/resource.h>
#include <sys/wait.h>
#include <unistd.h>

namespace fs = std::filesystem;
using Strings = std::vector<std::string>;
using Ns = int64_t;
constexpr Ns second = 1'000'000'000;

Ns now() {
    timespec ts{};
    clock_gettime(CLOCK_MONOTONIC, &ts);
    return ts.tv_sec * second + ts.tv_nsec;
}

std::string env(const char* name, const char* fallback) {
    const char* value = std::getenv(name);
    return value && *value ? value : fallback;
}

double cpu_ms(const rusage& usage) {
    return (usage.ru_utime.tv_sec + usage.ru_stime.tv_sec) * 1000.0 +
           (usage.ru_utime.tv_usec + usage.ru_stime.tv_usec) / 1000.0;
}

using Settings = std::vector<std::pair<std::string, std::string>>;
struct Child {
    int status = -1;
    double cpu = 0, wall = 0;
    Ns finished = 0;
    std::string log;
    bool ok() const { return status == 0; }
};

Child run(const Strings& command, Ns deadline, unsigned threads = 8,
          const Settings& settings = {}, bool trace = false, bool perf = false) {
    Child child;
    Ns start = now();
    if (start >= deadline) return child;
    std::vector<char*> args;
    for (const auto& arg : command) args.push_back(const_cast<char*>(arg.c_str()));
    args.push_back(nullptr);
    int pipefd[2];
    if (pipe(pipefd) != 0) throw std::runtime_error("pipe failed");
    pid_t pid = fork();
    if (pid == 0) {
        close(pipefd[0]);
        if (dup2(pipefd[1], STDERR_FILENO) < 0) _exit(126);
        close(pipefd[1]);
        setenv("NCPU", std::to_string(threads).c_str(), 1);
        for (const char* key : {"SHOPPING_SCALAR", "SHOPPING_READ", "SHOPPING_TRACE", "SHOPPING_DETAIL",
             "SHOPPING_PERF", "SHOPPING_EVENT_BUCKETS", "SHOPPING_PERSON_PARTITIONS", "SHOPPING_HUGE",
             "SHOPPING_NT", "SHOPPING_UNMAP"})
            unsetenv(key);
        for (const auto& [name, value] : settings) setenv(name.c_str(), value.c_str(), 1);
        if (trace) setenv("SHOPPING_TRACE", "1", 1);
        if (perf) setenv("SHOPPING_PERF", "1", 1);
        execv(args[0], args.data());
        _exit(127);
    }
    close(pipefd[1]);
    if (pid < 0) {
        close(pipefd[0]);
        throw std::runtime_error("fork failed");
    }
    fcntl(pipefd[0], F_SETFL, O_NONBLOCK);
    auto drain = [&] {
        char bytes[4096];
        for (ssize_t n; (n = read(pipefd[0], bytes, sizeof(bytes))) > 0;)
            if (child.log.size() < 32'768) child.log.append(bytes, n);
    };
    rusage usage{};
    for (;;) {
        drain();
        pid_t done = wait4(pid, &child.status, WNOHANG, &usage);
        if (done == pid) break;
        if (done < 0 && errno != EINTR) { child.status = -1; break; }
        Ns left = deadline - now();
        if (left <= 0) {
            kill(pid, SIGKILL);
            while (wait4(pid, &child.status, 0, &usage) < 0 && errno == EINTR) {}
            child.status = -1;
            break;
        }
        pollfd event{pipefd[0], POLLIN, 0};
        poll(&event, 1, std::min<Ns>(25, left / 1'000'000 + 1));
    }
    child.finished = now();
    child.wall = (child.finished - start) / 1e6;
    child.cpu = cpu_ms(usage);
    drain();
    close(pipefd[0]);
    return child;
}

struct Time { double cpu = 0, wall = 0; };
using Counters = std::array<uint64_t, 4>;
constexpr std::array<std::string_view, 7> phases{
    "start", "mappings", "event_ingest_links_catalogs", "event_dedup_route",
    "click_scan_validation", "sessions_user_output", "finish"};
constexpr size_t first_counter_phase = 2, counter_phases = 4;
struct Sample {
    Time total, exit;
    std::array<Time, phases.size()> phase{};
    std::array<Counters, counter_phases> counters{};
    unsigned phases_found = 0, counters_found = 0;
    size_t buckets = 0, partitions = 0;
};

Sample decode(const Child& child) {
    Sample result{};
    result.total = {child.cpu, child.wall};
    std::istringstream log(child.log);
    Ns finish = 0;
    double phase_cpu = 0;
    for (std::string line; std::getline(log, line);) {
        std::istringstream row(line);
        std::string record, name;
        row >> record;
        if (record == "partitions") {
            row >> result.buckets >> result.partitions;
            continue;
        }
        row >> name;
        size_t i = std::find(phases.begin(), phases.end(), name) - phases.begin();
        if (record == "phase" && i != phases.size()) {
            Ns at;
            double cpu, wall;
            if (!(row >> at >> wall >> cpu)) continue;
            result.phase[i] = {cpu * 1000, wall * 1000};
            phase_cpu += cpu * 1000;
            ++result.phases_found;
            if (i == 0) result.phase[i].wall = (at - child.finished) / 1e6 + child.wall;
            if (i == phases.size() - 1) finish = at;
        } else if (record == "perf" && i >= first_counter_phase && i < first_counter_phase + counter_phases) {
            auto& c = result.counters[i - first_counter_phase];
            if (row >> c[0] >> c[1] >> c[2] >> c[3]) ++result.counters_found;
        }
    }
    if (finish) result.exit = {std::max(0.0, child.cpu - phase_cpu), (child.finished - finish) / 1e6};
    return result;
}

using Samples = std::vector<Sample>;

struct Variant {
    std::string name, binary;
    unsigned threads = 8;
    Settings settings{};
    bool supported = true;
    Samples samples{};
};

void measure(Variant& engine, const fs::path& data, const fs::path& out, Ns deadline, bool perf = false) {
    if (!engine.supported || now() >= deadline) return;
    auto child = run({engine.binary, data.string(), out.string()},
                     std::min(deadline, now() + 14 * second), engine.threads, engine.settings, true, perf);
    auto reading = decode(child);
    if (child.ok() && reading.phases_found == phases.size()) engine.samples.push_back(reading);
    else std::fprintf(stderr, "R8 %s failed (%d): %.180s\n", engine.name.c_str(), child.status, child.log.c_str());
    fs::remove_all(out);
}

template<class Get> auto median(const Samples& samples, Get get) {
    std::array<decltype(get(samples.front())), 3> values{};
    for (size_t i = 0; i < samples.size(); ++i) values[i] = get(samples[i]);
    std::sort(values.begin(), values.begin() + samples.size());
    return values[samples.size() / 2];
}

int64_t rounded(double number) { return static_cast<int64_t>(number + 0.5); }

template<class Get> std::string times(const Samples& samples, Get get) {
    if (samples.empty()) return "-";
    return std::to_string(rounded(median(samples, [&](const auto& s) { return get(s).cpu; }))) + "." +
           std::to_string(rounded(median(samples, [&](const auto& s) { return get(s).wall; })));
}

std::string digits36(uint64_t number) {
    std::string result;
    do {
        result.push_back("0123456789abcdefghijklmnopqrstuvwxyz"[number % 36]);
        number /= 36;
    } while (number);
    std::reverse(result.begin(), result.end());
    return result;
}

std::string phase_times(const Samples& side) {
    std::string result = std::to_string(side.size()) + " " + times(side, [](auto& s) { return s.total; });
    for (size_t i = 0; i < phases.size(); ++i)
        result += "/" + times(side, [i](auto& s) { return s.phase[i]; });
    return result + "/" + times(side, [](auto& s) { return s.exit; });
}

std::string counters(const Samples& samples, uint64_t events) {
    Samples complete;
    for (const auto& s : samples) if (s.counters_found == counter_phases) complete.push_back(s);
    if (complete.empty() || !events) return "-";
    std::string result;
    for (size_t i = 0; i < counter_phases; ++i) {
        for (size_t j = 0; j < Counters{}.size(); ++j) {
            if (i || j) result += j ? '.' : '/';
            uint64_t value = median(complete, [=](auto& s) { return s.counters[i][j]; });
            result += digits36(uint64_t(double(value) * 1000 / events + 0.5));
        }
    }
    return result;
}

int64_t wall_ms(const Samples& samples) {
    return samples.empty() ? -1 : rounded(median(samples, [](auto& s) { return s.total.wall; }));
}

std::string entry(const Variant& variant) {
    return variant.name + "=" + (variant.samples.size() == 3
        ? std::to_string(wall_ms(variant.samples))
        : (variant.supported ? "-" + std::to_string(variant.samples.size()) : "na"));
}

std::string cache_sizes() {
    std::array<std::string, 3> sizes{"?", "?", "?"};
#ifdef __linux__
    int cpu = std::max(0, sched_getcpu());
    auto base = fs::path("/sys/devices/system/cpu") / ("cpu" + std::to_string(cpu)) / "cache";
    for (int i = 0; i < 16; ++i) {
        auto index = base / ("index" + std::to_string(i));
        int level = 0;
        std::string type, size;
        std::ifstream(index / "level") >> level;
        std::ifstream(index / "type") >> type;
        std::ifstream(index / "size") >> size;
        if (level >= 1 && level <= 3 && type != "Instruction" && !size.empty()) sizes[level - 1] = size;
    }
#endif
    return "L=" + sizes[0] + "/" + sizes[1] + "/" + sizes[2];
}

void report(const fs::path& path, uint64_t events, const Variant& base, const Variant& current,
            const std::vector<Variant>& sweep, size_t buckets, size_t partitions, const std::string& pgo) {
    std::array<std::string, 5> payloads;
    payloads[0] = "B" + phase_times(base.samples) + " N" + phase_times(current.samples);
    payloads[1] = "E=" + std::to_string(buckets) + " P=" + std::to_string(partitions);
    payloads[3] = "PG=" + pgo;
    for (size_t i = 0; i < sweep.size(); ++i)
        payloads[i < 16 ? 1 + i / 8 : 3] += " " + entry(sweep[i]);
    payloads[3] += " exit=" + times(current.samples, [](auto& s) { return s.exit; });
    payloads[3] += "/" + (sweep.size() > 16 ? times(sweep.back().samples, [](auto& s) { return s.exit; }) : "-");
    auto old_counters = counters(base.samples, events), new_counters = counters(current.samples, events);
    if (new_counters != "-") {
        payloads[4] = "B=" + old_counters + " N=" + new_counters;
    } else {
        payloads[4] = cache_sizes();
        std::vector<const Variant*> best;
        for (const auto& variant : sweep)
            if (variant.samples.size() == 3) best.push_back(&variant);
        std::stable_sort(best.begin(), best.end(), [](auto* a, auto* b) {
            return wall_ms(a->samples) < wall_ms(b->samples);
        });
        for (size_t i = 0; i < std::min<size_t>(5, best.size()); ++i) payloads[4] += " " + entry(*best[i]);
    }

    constexpr parquet::Column schema[] = {{"user_id", parquet::Column::Text, 0},
        {"day", parquet::Column::Int32, 6}, {"sessions", parquet::Column::Int64},
        {"events", parquet::Column::Int64}, {"revenue_cents", parquet::Column::Int64},
        {"top_category", parquet::Column::Text, 0, true}};
    parquet::File file(path, schema);
    file.append(std::size(payloads), [&](size_t column, parquet::Bytes& bytes) {
        if (column == 5) parquet::definitions(bytes, std::size(payloads), [](size_t) { return true; });
        for (size_t i = 0; i < std::size(payloads); ++i) {
            const auto& payload = payloads[i];
            switch (column) {
                case 0: parquet::plain(bytes, "!0" + std::to_string(i) + " " + payload.substr(0, 155)); break;
                case 1: parquet::little_endian(bytes, int32_t(10957 + i)); break;
                case 2: parquet::little_endian(bytes, wall_ms(base.samples)); break;
                case 3: parquet::little_endian(bytes, wall_ms(current.samples)); break;
                case 4: parquet::little_endian(bytes, static_cast<int64_t>(events)); break;
                case 5: parquet::plain(bytes, payload.size() > 155 ? payload.substr(155) : "-"); break;
            }
        }
    });
    file.finish();
    for (size_t i = 0; i < std::size(payloads); ++i)
        std::fprintf(stderr, "R8 !0%zu %s\n", i, payloads[i].c_str());
}

int main(int argc, char** argv) try {
    if (argc == 3 && std::string_view(argv[1]) == "--clean") { fs::remove_all(argv[2]); return 0; }
    Ns start = now(), deadline = start + 132 * second;
    alarm(150);
    fs::path data = env("DATA_DIR", "/data"), output = env("OUTPUT_DIR", "/output");
    fs::path scratch = env("SCRATCH_DIR", "/scratch");
    std::string shipped = env("R8_SHIPPED", "/shopping");
    Variant base{"B", env("R8_BASE", "/shopping-base")};
    Variant current{"N", env("R8_NEW", "/shopping-new")};
    std::string generator = env("R8_GENERATOR", "/generate");
    std::string pgo = "???";
    std::ifstream(env("R8_PGO_STATUS", "/pgo-status")) >> pgo;
    fs::path trial, work;
    std::vector<Variant> sweep;
    uint64_t events = 0;
    size_t buckets = 0, partitions = 0;
    try {
        std::string temp = (scratch / "r8-XXXXXX").string();
        if (!mkdtemp(temp.data())) throw std::runtime_error("could not create probe scratch directory");
        work = temp;
        trial = output / ("." + work.filename().string());
        fs::path generated = work / "generated";
        events = std::clamp(std::stoull(env("R8_EVENTS", "20000000")), 1000ULL, 20'000'000ULL);
        auto generate = [&](unsigned budget) {
            return run({generator, "--seed", "20261120", "--events", std::to_string(events),
                        "--clicks", std::to_string(events / 4), "--threads", "8", "--output", generated.string()},
                       std::min(deadline, now() + budget * second));
        };
        auto child = generate(32);
        if (!child.ok() && events > 5'000'000) {
            fs::remove_all(generated);
            events = 5'000'000;
            child = generate(15);
        }
        if (!child.ok()) {
            events = 0;
            throw std::runtime_error("could not generate synthetic data: " + child.log.substr(0, 200));
        }
        for (unsigned repetition = 0; repetition < 3; ++repetition) {
            auto* first = repetition % 2 ? &current : &base;
            auto* second = repetition % 2 ? &base : &current;
            measure(*first, generated, trial, deadline, true);
            measure(*second, generated, trial, deadline, true);
        }
        if (current.samples.empty() || !current.samples.front().buckets || !current.samples.front().partitions)
            throw std::runtime_error("no Round 8 partition counts");
        buckets = current.samples.front().buckets;
        partitions = current.samples.front().partitions;
        auto size_at = [](size_t count, unsigned i) { return std::max<size_t>(1, (count << i) / 4); };
        for (unsigned e = 0; e < 4; ++e) for (unsigned p = 0; p < 4; ++p) {
            Variant variant{std::to_string(e) + std::to_string(p), current.binary};
            variant.settings = {{"SHOPPING_EVENT_BUCKETS", std::to_string(size_at(buckets, e))},
                                {"SHOPPING_PERSON_PARTITIONS", std::to_string(size_at(partitions, p))}};
            if (e == 2 && p == 2) variant.samples = current.samples;
            sweep.push_back(std::move(variant));
        }
        sweep.push_back({"t6", current.binary, 6});
        sweep.push_back({"t5", current.binary, 5});
        sweep.push_back({"v3", env("R8_V3", "/shopping-v3"), 8, {},
                         pgo.size() == 3 && pgo[1] != 'x' && supported_isa("v3")});
        sweep.push_back({"cl", env("R8_CASCADELAKE", "/shopping-cl"), 8, {},
                         pgo.size() == 3 && pgo[2] != 'x' && supported_isa("cascadelake")});
        sweep.push_back({"pg0", env("R8_PLAIN", "/shopping-plain")});
        sweep.push_back({"hp0", current.binary, 8, {{"SHOPPING_HUGE", "0"}}});
        sweep.push_back({"nt0", current.binary, 8, {{"SHOPPING_NT", "0"}}});
        sweep.push_back({"um", current.binary, 8, {{"SHOPPING_UNMAP", "1"}}});
        for (size_t i = 0; i < 3 && now() < deadline; ++i) {
            for (size_t j = 0; j < sweep.size() && now() < deadline; ++j) {
                size_t index = (j + sweep.size() - 8 * i) % sweep.size();
                if (sweep[index].samples.size() < 3)
                    measure(sweep[index], generated, trial, deadline);
            }
        }
    } catch (const std::exception& error) {
        std::fprintf(stderr, "R8 diagnostics: %.200s\n", error.what());
    }

    std::string self = fs::absolute(argv[0]).string();
    if (!trial.empty()) run({self, "--clean", trial.string()}, std::min(start + 135 * second, now() + 3 * second));
    if (!work.empty()) run({self, "--clean", work.string()}, std::min(start + 135 * second, now() + 3 * second));
    auto answer = run({shipped, data.string(), output.string()}, start + 148 * second);
    if (!answer.ok()) {
        std::fprintf(stderr, "R8 dev engine failed (%d): %.200s\n", answer.status, answer.log.c_str());
        return 1;
    }
    report(output / "user_daily" / "!r8.parquet", events, base, current, sweep, buckets, partitions, pgo);
    return 0;
} catch (const std::exception& error) {
    std::fprintf(stderr, "R8 %s\n", error.what());
    return 1;
}
