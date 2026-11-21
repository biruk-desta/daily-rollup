#include "parquet.hpp"

#include <algorithm>
#include <array>
#include <cerrno>
#include <csignal>
#include <cstdlib>
#include <sstream>
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

enum class Mode { Plain, Timing, Detail };
struct Child {
    int status = -1;
    double cpu = 0, wall = 0;
    std::string log;
    bool ok() const { return status == 0; }
};

Child run(const Strings& command, Ns deadline, unsigned threads = 8, Mode mode = Mode::Plain) {
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
        setenv("NCPU", threads == 4 ? "4" : "8", 1);
        for (const char* key : {"SHOPPING_SCALAR", "SHOPPING_READ", "SHOPPING_TRACE", "SHOPPING_DETAIL", "SHOPPING_PERF"})
            unsetenv(key);
        if (mode != Mode::Plain) setenv("SHOPPING_TRACE", "1", 1);
        if (mode == Mode::Timing) setenv("SHOPPING_PERF", "1", 1);
        if (mode == Mode::Detail) setenv("SHOPPING_DETAIL", "1", 1);
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
    child.wall = (now() - start) / 1e6;
    child.cpu = cpu_ms(usage);
    drain();
    close(pipefd[0]);
    return child;
}

struct Time { double cpu = 0, wall = 0; };
using Counters = std::array<uint64_t, 4>;
constexpr std::array<std::string_view, 4> phases{"event_ingest_links_catalogs", "event_dedup_route",
                                                "click_scan_validation", "sessions_user_output"};
constexpr std::array<std::string_view, 10> details{"scan", "parse", "validate", "hash", "route", "group",
                                                  "sessions", "attribution", "encode", "write"};
struct Sample {
    Time total;
    std::array<Time, phases.size()> phase{};
    std::array<Time, details.size()> detail{};
    std::array<Counters, phases.size()> counters{};
    unsigned phases_found = 0, details_found = 0, counters_found = 0;
};

template<class Names> size_t find(const Names& names, const std::string& name) {
    return std::find(names.begin(), names.end(), name) - names.begin();
}

Sample decode(const Child& child) {
    Sample result{{child.cpu, child.wall}};
    std::istringstream log(child.log);
    for (std::string line; std::getline(log, line);) {
        std::istringstream row(line);
        std::string record, name;
        row >> record >> name;
        size_t i = find(phases, name);
        if (record == "phase" && i != phases.size()) {
            Ns at;
            double cpu, wall;
            if (!(row >> at >> wall >> cpu)) continue;
            result.phase[i] = {cpu * 1000, wall * 1000};
            ++result.phases_found;
        } else if (record == "perf" && i != phases.size()) {
            auto& c = result.counters[i];
            if (row >> c[0] >> c[1] >> c[2] >> c[3]) ++result.counters_found;
        } else if (record == "detail") {
            size_t j = find(details, name);
            if (j != details.size() && row >> result.detail[j].cpu >> result.detail[j].wall) ++result.details_found;
        }
    }
    return result;
}

using Samples = std::vector<Sample>;
using Pair = std::array<Samples, 2>;

Pair measure(const std::array<std::string, 2>& engines, const fs::path& data, const fs::path& out,
             Ns deadline, unsigned threads, Mode mode) {
    Pair result;
    for (unsigned trial = 0; trial != 3; ++trial) {
        for (unsigned order = 0; order != 2; ++order) {
            if (now() >= deadline) return result;
            unsigned side = (trial + order) % 2;
            fs::path target = out / (std::to_string(side) + "-" + std::to_string(trial));
            auto child = run({engines[side], data.string(), target.string()},
                             std::min(deadline, now() + 14 * second), threads, mode);
            auto reading = decode(child);
            if (child.ok() && reading.phases_found == phases.size() &&
                (mode != Mode::Detail || reading.details_found == details.size())) {
                result[side].push_back(reading);
            } else {
                std::fprintf(stderr, "R7 engine %u sample %u failed (%d): %.200s\n",
                             side, trial, child.status, child.log.c_str());
            }
        }
    }
    return result;
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
    return result;
}

std::string detail_times(const Samples& side, size_t start) {
    std::string result = std::to_string(side.size()) + " ";
    for (size_t i = start; i < start + 5; ++i)
        result += (i == start ? "" : "/") + times(side, [i](auto& s) { return s.detail[i]; });
    return result;
}

std::string counters(const Samples& samples, uint64_t events) {
    Samples complete;
    for (const auto& s : samples) if (s.counters_found == phases.size()) complete.push_back(s);
    if (complete.empty() || !events) return "-";
    std::string result;
    for (size_t i = 0; i < phases.size(); ++i) {
        for (size_t j = 0; j < Counters{}.size(); ++j) {
            if (i || j) result += j ? '.' : '/';
            uint64_t value = median(complete, [=](auto& s) { return s.counters[i][j]; });
            result += digits36(uint64_t(double(value) * 1000 / events + 0.5));
        }
    }
    return result;
}

void report(const fs::path& path, uint64_t events, const Pair& four, const Pair& detail, const Pair& eight) {
    const std::string payloads[] = {
        "4 B " + phase_times(four[0]) + " N " + phase_times(four[1]),
        "I B " + detail_times(detail[0], 0) + " N " + detail_times(detail[1], 0),
        "S B " + detail_times(detail[0], 5) + " N " + detail_times(detail[1], 5),
        "P B " + counters(four[0], events) + " N " + counters(four[1], events),
        "8 B " + phase_times(eight[0]) + " N " + phase_times(eight[1])
    };
    constexpr parquet::Column schema[] = {{"user_id", parquet::Column::Text, 0},
        {"day", parquet::Column::Int32, 6}, {"sessions", parquet::Column::Int64},
        {"events", parquet::Column::Int64}, {"revenue_cents", parquet::Column::Int64},
        {"top_category", parquet::Column::Text, 0, true}};
    auto wall = [](const Samples& samples) {
        return samples.empty() ? int64_t{-1} : rounded(median(samples, [](auto& s) { return s.total.wall; }));
    };
    parquet::File file(path, schema);
    file.append(std::size(payloads), [&](size_t column, parquet::Bytes& bytes) {
        if (column == 5) parquet::definitions(bytes, std::size(payloads), [](size_t) { return true; });
        for (size_t i = 0; i < std::size(payloads); ++i) {
            const auto& payload = payloads[i];
            const Pair& timing = i == 4 ? eight : four;
            switch (column) {
                case 0: parquet::plain(bytes, "!0" + std::to_string(i) + " " + payload.substr(0, 155)); break;
                case 1: parquet::little_endian(bytes, int32_t(10957 + i)); break;
                case 2: parquet::little_endian(bytes, wall(timing[0])); break;
                case 3: parquet::little_endian(bytes, wall(timing[1])); break;
                case 4: parquet::little_endian(bytes, static_cast<int64_t>(events)); break;
                case 5: parquet::plain(bytes, payload.size() > 155 ? payload.substr(155) : "-"); break;
            }
        }
    });
    file.finish();
    for (size_t i = 0; i < std::size(payloads); ++i)
        std::fprintf(stderr, "R7 !0%zu %s\n", i, payloads[i].c_str());
}

int main(int argc, char** argv) try {
    if (argc == 3 && std::string_view(argv[1]) == "--clean") { fs::remove_all(argv[2]); return 0; }
    Ns start = now(), deadline = start + 132 * second;
    alarm(150);
    fs::path data = env("DATA_DIR", "/data"), output = env("OUTPUT_DIR", "/output");
    fs::path scratch = env("SCRATCH_DIR", "/scratch");
    std::string shipped = env("R7_SHIPPED", "/shopping");
    std::array engines{env("R7_BASE", "/shopping-base"), env("R7_NEW", "/shopping-new")};
    std::array detail_engines{env("R7_BASE_DETAIL", "/shopping-base-detail"),
                              env("R7_NEW_DETAIL", "/shopping-new-detail")};
    std::string generator = env("R7_GENERATOR", "/generate");
    fs::path trial, work;
    Pair four, detail, eight;
    uint64_t events = 0;
    try {
        std::string temp = (scratch / "r7-XXXXXX").string();
        if (!mkdtemp(temp.data())) throw std::runtime_error("could not create probe scratch directory");
        work = temp;
        trial = output / ("." + work.filename().string());
        fs::path generated = work / "generated";
        events = std::clamp(std::stoull(env("R7_EVENTS", "20000000")), 1000ULL, 20'000'000ULL);
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
        four = measure(engines, generated, trial / "four", deadline, 4, Mode::Timing);
        detail = measure(detail_engines, generated, trial / "detail", deadline, 4, Mode::Detail);
        eight = measure(engines, generated, trial / "eight", deadline, 8, Mode::Timing);
    } catch (const std::exception& error) {
        std::fprintf(stderr, "R7 diagnostics: %.200s\n", error.what());
    }

    std::string self = fs::absolute(argv[0]).string();
    if (!trial.empty()) run({self, "--clean", trial.string()}, std::min(start + 135 * second, now() + 3 * second));
    if (!work.empty()) run({self, "--clean", work.string()}, std::min(start + 135 * second, now() + 3 * second));
    auto answer = run({shipped, data.string(), output.string()}, start + 148 * second);
    if (!answer.ok()) {
        std::fprintf(stderr, "R7 dev engine failed (%d): %.200s\n", answer.status, answer.log.c_str());
        return 1;
    }
    report(output / "user_daily" / "!r7.parquet", events, four, detail, eight);
    return 0;
} catch (const std::exception& error) {
    std::fprintf(stderr, "R7 %s\n", error.what());
    return 1;
}
