#include "parquet.hpp"

#include <algorithm>
#include <array>
#include <bit>
#include <cerrno>
#include <cmath>
#include <csignal>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <span>
#include <sstream>
#include <string_view>

#ifdef __linux__
#include <sys/prctl.h>
#endif
#include <fcntl.h>
#include <poll.h>
#include <sys/mman.h>
#include <sys/resource.h>
#include <sys/stat.h>
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

Child run(const Strings& command, Ns deadline, const std::string& threads, const char* dense = nullptr) {
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
                "SHOPPING_ROW_LANES", "SHOPPING_ROW_TEMPLATE", "SHOPPING_RECORD_BYTES", "SHOPPING_SESSION_FAST",
                "SHOPPING_DEDUP_LOAD", "SHOPPING_DENSE", "SHOPPING_DENSE_SPANS", "SHOPPING_REUSE_ARENAS",
                "SHOPPING_PERSON_DOUBLE_HASH"})
            unsetenv(key);
        if (dense) {
            setenv("SHOPPING_TRACE", "1", 1);
            setenv("SHOPPING_DENSE", dense, 1);
        }
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

enum Metric { Total, Ingest, Dedup, Sessions, Exit, User, System, Rss, MetricCount };

struct Sample {
    std::array<int64_t, MetricCount> metrics;
    int64_t dense = missing, exact = missing, scanned = missing;

    Sample() { metrics.fill(missing); }
    bool complete() const { return std::ranges::find(metrics, missing) == metrics.end(); }
};

int64_t ms(Ns duration) { return std::max<Ns>(0, duration + millisecond / 2) / millisecond; }
int64_t ms(timeval cpu) { return ms(cpu.tv_sec * second + cpu.tv_usec * 1000); }

Sample decode(const Child& child, bool dense) {
    Sample sample;
    if (!child.end) return sample;
    auto& metrics = sample.metrics;
    metrics[Total] = ms(child.end - child.start);
#ifdef __APPLE__
    constexpr int64_t to_mib = 1 << 20;
#else
    constexpr int64_t to_mib = 1 << 10;
#endif
    if (child.usage.ru_maxrss > 0) metrics[Rss] = (child.usage.ru_maxrss + to_mib / 2) / to_mib;
    if (child.status != -1) {
        metrics[User] = ms(child.usage.ru_utime);
        metrics[System] = ms(child.usage.ru_stime);
    }
    std::istringstream log(child.log);
    Ns sessions_at = 0;
    double sessions_wall = 0;
    for (std::string line; std::getline(log, line);) {
        std::istringstream row(line);
        std::string record, name;
        row >> record >> name;
        if (dense && record == "dense") {
            if (name == "recovery") {
                std::string status;
                row >> status;
                if (status == "confirmed") sample.dense = 1;
                if (status == "fallback") sample.dense = 0;
            } else if (name == "candidates") {
                int64_t candidates, side, exact, empty;
                std::string label;
                if (row >> candidates >> label >> side >> label >> exact >> label >> empty) {
                    sample.exact = exact;
                    sample.scanned = candidates + exact + empty;
                }
            }
        }
        if (record != "phase") continue;
        Ns at;
        double wall;
        if (!(row >> at >> wall)) continue;
        if (name == "event_ingest_links_catalogs") metrics[Ingest] = std::lround(wall * 1000);
        else if (name == "event_dedup_route") metrics[Dedup] = std::lround(wall * 1000);
        else if (name == "sessions_user_output") {
            sessions_at = at;
            sessions_wall = wall;
        } else if (name == "finish") {
            if (sessions_at) metrics[Sessions] = std::lround(sessions_wall * 1000 + double(at - sessions_at) / millisecond);
            metrics[Exit] = ms(child.end - at);
        }
    }
    if (dense && child.ok() && sample.complete() && sample.dense != 1) {
        sample.dense = 0;
    }
    return sample;
}

struct Failures {
    Strings entries;

    void add(const std::string& stage, const std::string& reason, const std::string& log = {}) {
        std::fprintf(stderr, "P11 %s %s %.200s\n", stage.c_str(), reason.c_str(), log.c_str());
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

constexpr std::array<size_t, 8> order{0, 1, 1, 0, 0, 1, 1, 0};
using Trials = std::array<Sample, order.size()>;

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
constexpr size_t row_limit = 275;  // Leave four characters for "!0n ".

class MappedFile {
    void* data_;
    size_t size_;

public:
    explicit MappedFile(const fs::path& path) {
        const int fd = open(path.c_str(), O_RDONLY);
        if (fd < 0) throw std::runtime_error("cannot open " + path.string());
        struct stat info{};
        if (fstat(fd, &info) != 0 || info.st_size < 12) {
            close(fd);
            throw std::runtime_error("invalid Parquet file " + path.string());
        }
        size_ = info.st_size;
        data_ = mmap(nullptr, size_, PROT_READ, MAP_PRIVATE, fd, 0);
        close(fd);
        if (data_ == MAP_FAILED) throw std::runtime_error("cannot map " + path.string());
    }
    MappedFile(const MappedFile&) = delete;
    ~MappedFile() { munmap(data_, size_); }
    std::string_view bytes() const { return {static_cast<const char*>(data_), size_}; }
};

struct Cursor {
    std::string_view bytes;

    std::string_view take(size_t count) {
        if (count > bytes.size()) throw std::runtime_error("short Parquet page");
        const auto result = bytes.substr(0, count);
        bytes.remove_prefix(count);
        return result;
    }
    void expect(std::string_view expected) {
        if (take(expected.size()) != expected) throw std::runtime_error("unexpected Parquet page encoding");
    }
    uint32_t u32() {
        uint32_t result;
        std::memcpy(&result, take(sizeof(result)).data(), sizeof(result));
        return result;
    }
    uint32_t varint() {
        uint32_t value = 0;
        for (unsigned shift = 0; shift <= 28; shift += 7) {
            const uint8_t byte = take(1)[0];
            if (shift == 28 && byte > 15) break;
            value |= uint32_t(byte & 127) << shift;
            if (byte < 128) return value;
        }
        throw std::runtime_error("invalid Parquet varint");
    }
    uint32_t positive() {
        const auto value = varint();
        if (value & 1) throw std::runtime_error("negative Parquet page size or count");
        return value >> 1;
    }
};

uint64_t mix(uint64_t value) {
    value = (value ^ (value >> 30)) * 0xbf58476d1ce4e5b9ULL;
    value = (value ^ (value >> 27)) * 0x94d049bb133111ebULL;
    return value ^ (value >> 31);
}

uint64_t hash_field(std::string_view field) {
    uint64_t hash = field.size();
    while (field.size() >= 8) {
        uint64_t word;
        std::memcpy(&word, field.data(), sizeof(word));
        hash = mix(hash ^ word);
        field.remove_prefix(8);
    }
    uint64_t tail = 0;
    if (!field.empty()) std::memcpy(&tail, field.data(), field.size());
    return hash ^ mix(tail);
}

uint64_t hash_table(const fs::path& folder, std::span<const int> widths, bool nullable) {
    using namespace std::literals;
    uint64_t sum = 0, xors = 0, rows = 0;
    size_t files = 0;
    for (const auto& part : fs::directory_iterator(folder)) {
        if (part.path().extension() != ".parquet") continue;
        ++files;
        const MappedFile file(part.path());
        Cursor input{file.bytes()}, footer{input.bytes.substr(input.bytes.size() - 8)};
        input.expect("PAR1");
        const auto footer_size = footer.u32();
        footer.expect("PAR1");
        if (footer_size > input.bytes.size() - 8) throw std::runtime_error("invalid Parquet footer");
        input.bytes.remove_suffix(footer_size + 8);
        while (!input.bytes.empty()) {
            std::array<Cursor, 7> columns{};
            std::string_view present;
            size_t count = 0;
            for (size_t column = 0; column < widths.size(); ++column) {
                input.expect("\x15\0\x15"sv);
                const auto bytes = input.positive();
                input.expect("\x15"sv);
                if (input.positive() != bytes) throw std::runtime_error("compressed Parquet page");
                input.expect("\x2c\x15"sv);
                const auto values = input.positive();
                if (!column) count = values;
                if (!count || values != count) throw std::runtime_error("invalid Parquet row group");
                input.expect("\x15\0\x15\x06\x15\x06\0\0"sv);
                auto& page = columns[column];
                page.bytes = input.take(bytes);
                if (nullable && column + 1 == widths.size()) {
                    const auto level_bytes = page.u32();
                    Cursor levels{page.take(level_bytes)};
                    if (levels.varint() != (((count + 7) / 8) * 2 + 1))
                        throw std::runtime_error("unexpected Parquet definition levels");
                    present = levels.take((count + 7) / 8);
                    if (!levels.bytes.empty()) throw std::runtime_error("extra Parquet definition levels");
                }
            }
            for (size_t row = 0; row < count; ++row) {
                uint64_t hash = 0x9e3779b97f4a7c15ULL;
                for (size_t column = 0; column < widths.size(); ++column) {
                    if (nullable && column + 1 == widths.size() && !(uint8_t(present[row / 8]) & (1u << (row % 8)))) {
                        hash = mix(hash ^ 0xfedcba9876543210ULL);
                        continue;
                    }
                    auto& page = columns[column];
                    const auto length = widths[column] ? uint32_t(widths[column]) : page.u32();
                    hash = mix(hash ^ hash_field(page.take(length)));
                }
                sum += hash;
                xors ^= mix(hash);
            }
            for (const auto& page : columns)
                if (!page.bytes.empty()) throw std::runtime_error("extra Parquet values");
            rows += count;
        }
    }
    if (!files) throw std::runtime_error("missing Parquet output " + folder.string());
    return mix(sum ^ std::rotl(xors, 21) ^ mix(rows));
}

struct Dataset {
    Trials runs;
    std::array<fs::path, 2> retained;
    std::array<std::string, 2> hashes{"-", "-"};
    std::array<int64_t, 2> mode{missing, missing};
    int64_t events = 0, generate_ms = missing;
};

Rows summarize(const std::array<Dataset, 2>& data, const std::array<int64_t, 4>& writeback,
               int64_t diagnostic_ms, int64_t dev_ms, const Failures& failures) {
    Rows rows;
    for (size_t size = 0; size < data.size(); ++size) {
        for (size_t i = 0; i < order.size(); ++i) {
            auto& row = rows[size * 2 + i / 4];
            const auto& sample = data[size].runs[i];
            pack_all(row, sample.metrics);
            if (order[i]) pack_all(row, std::array{sample.dense, sample.exact, sample.scanned});
        }
    }
    rows[4] = "h=" + data[0].hashes[0] + '/' + data[0].hashes[1] + '/' + data[1].hashes[0] + '/' + data[1].hashes[1] +
              " d=" + join(std::array{data[0].mode[1], data[1].mode[1]}) +
              " n=" + join(std::array{data[0].events, data[1].events}) + " m=";
    pack_all(rows[4], std::array{data[0].generate_ms, data[1].generate_ms, diagnostic_ms, dev_ms});
    pack_all(rows[4], writeback);
    rows[4] += " f=";
    rows[4] += failures.report(row_limit - rows[4].size());
    return rows;
}

void report(const fs::path& path, const Rows& rows, int64_t events) {
    constexpr parquet::Column schema[] = {{"user_id", parquet::Column::Text, 0},
        {"day", parquet::Column::Int32, 6}, {"sessions", parquet::Column::Int64},
        {"events", parquet::Column::Int64}, {"revenue_cents", parquet::Column::Int64},
        {"top_category", parquet::Column::Text, 0, true}};
    constexpr size_t split = 150;
    for (size_t i = 0; i < rows.size(); ++i) {
        std::fprintf(stderr, "P11 !0%zu %s\n", i, rows[i].c_str());
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
    if (argc == 3 && std::string_view(argv[1]) == "--hash") {
        const fs::path output = argv[2];
        const auto users = hash_table(output / "user_daily", std::array{0, 4, 8, 8, 8, 0}, true);
        const auto campaigns = hash_table(output / "campaign_daily", std::array{0, 0, 4, 8, 8, 8, 8}, false);
        std::fprintf(stderr, "%016llx,%016llx\n", static_cast<unsigned long long>(users),
                     static_cast<unsigned long long>(campaigns));
        return 0;
    }
    const Ns start = now();
    alarm(150);
    const Ns deadline = start + number("PROBE_DIAG_SECONDS", 132) * second;
    const Ns rotate_deadline = deadline - std::min(12 * second, (deadline - start) / 10);
    const Ns answer_deadline = start + number("PROBE_DEV_SECONDS", 148) * second;
    const fs::path data = env("DATA_DIR", "/data"), output = env("OUTPUT_DIR", "/output");
    const std::string self = fs::absolute(argv[0]).string();
    const auto threads = std::to_string(number("PROBE_THREADS", 8));
    const std::array engines{env("PROBE_L", "/shopping-l"), env("PROBE_D", "/shopping-d")};
    Failures failures;
    std::array<Dataset, 2> readings;
    std::array<int64_t, 4> writeback{missing, missing, missing, missing};
    auto clean = [&](const fs::path& path, Ns cutoff, const std::string& stage) {
        const auto child = run({self, "--clean", path.string()}, std::min(cutoff, now() + 3 * second), threads);
        if (!child.ok()) failures.add(stage, child.error(), child.log);
        return child.ok();
    };

    fs::path work;
    try {
        work = temporary(env("SCRATCH_DIR", "/scratch"), "probe11-XXXXXX");
        const auto generator = env("PROBE_GENERATOR", "/generate");
        constexpr std::array sizes{"60", "20"};
        for (size_t size = 0; size < sizes.size(); ++size) {
            const std::string label = sizes[size];
            const auto dataset = work / (size == 0 ? "large" : "small");
            const auto events = size == 0 ? number("PROBE_EVENTS60", 60'000'000, 1000)
                                          : number("PROBE_EVENTS20", 20'000'000, 1000);
            const auto generation_cutoff = size == 0 ? deadline : rotate_deadline;
            if (now() >= generation_cutoff) {
                failures.add("g" + label, "budget");
                break;
            }
            auto& result = readings[size];
            const auto generated = run({generator, "--seed", size == 0 ? "20261123" : "20261120",
                "--events", std::to_string(events), "--clicks", std::to_string(events / 4),
                "--threads", threads, "--output", dataset.string()},
                std::min(generation_cutoff, now() + 54 * second), threads);
            if (generated.end) result.generate_ms = ms(generated.end - generated.start);
            if (!generated.ok()) {
                failures.add("g" + label, generated.error(), generated.log);
                if (now() < rotate_deadline && !clean(dataset, rotate_deadline, "c" + label)) break;
                continue;
            }
            result.events = events;
            const auto waited = wait_writeback(self, rotate_deadline, threads, failures);
            if (size == 0) writeback = waited;
            const Ns rotation_start = now();
            for (size_t i = 0; i < order.size(); ++i) {
                const size_t engine = order[i];
                const auto stage = "r" + label + '-' + std::to_string(i + 1) + "LD"[engine];
                const auto begin = now();
                if (begin >= rotate_deadline ||
                    (size == 1 && i == 4 && (begin - rotation_start) * 3 / 2 + second > rotate_deadline - begin)) {
                    failures.add(stage, "budget");
                    break;
                }
                const auto trial = work / stage;
                const auto child = run({engines[engine], dataset.string(), trial.string()},
                    std::min(rotate_deadline, begin + 14 * second), threads, engine ? "1" : "0");
                auto& sample = result.runs[i];
                sample = decode(child, engine == 1);
                if (!child.ok()) failures.add(stage, child.error(), child.log);
                else if (!sample.complete() || (sample.dense == 1 && sample.scanned == missing))
                    failures.add(stage, "trace", child.log);
                std::fprintf(stderr, "P11 run %s %s dense=%s exact/scanned=%s\n", stage.c_str(),
                    join(sample.metrics).c_str(), display(sample.dense).c_str(),
                    join(std::array{sample.exact, sample.scanned}).c_str());
                auto& saved = result.retained[engine];
                const bool keep_earlier = engine == 1 && !saved.empty() && result.mode[engine] == 1 && sample.dense != 1;
                if (child.ok() && !keep_earlier) {
                    if (!saved.empty() && now() < rotate_deadline) clean(saved, rotate_deadline, "c" + stage);
                    saved = trial;
                    result.mode[engine] = sample.dense;
                } else if (now() < rotate_deadline) {
                    clean(trial, rotate_deadline, "c" + stage);
                }
            }
            if (now() < rotate_deadline && !clean(dataset, rotate_deadline, "c" + label)) break;
        }
        for (size_t size = 0; size < readings.size(); ++size) {
            auto& result = readings[size];
            for (size_t engine = 0; engine < engines.size(); ++engine) {
                if (result.retained[engine].empty()) continue;
                const auto stage = std::string("h") + sizes[size] + "LD"[engine];
                const auto child = run({self, "--hash", result.retained[engine].string()}, deadline, threads);
                if (!child.ok()) failures.add(stage, child.error(), child.log);
                else if (child.log.size() != 34 || child.log.back() != '\n') failures.add(stage, "format", child.log);
                else result.hashes[engine] = child.log.substr(0, 33);
            }
        }
    } catch (const std::exception& error) {
        failures.add("diag", "error", error.what());
    }
    if (!work.empty()) clean(work, start + 135 * second, "clean");
    const auto diagnostic_ms = ms(now() - start);

    fs::create_directories(output);
    const auto answer = temporary(output, ".probe11-answer-XXXXXX");
    const auto dev = run({env("PROBE_DEV", "/shopping-dev"), data.string(), answer.string()}, answer_deadline, threads);
    if (!dev.ok()) {
        failures.add("dev", dev.error(), dev.log);
        clean(answer, start + 150 * second - 100 * millisecond, "cdev");
        return 1;
    }
    try {
        report(answer / "user_daily" / "!probe11.parquet",
               summarize(readings, writeback, diagnostic_ms, ms(dev.end - dev.start), failures),
               readings[0].events);
        fs::rename(answer / "campaign_daily", output / "campaign_daily");
        fs::rename(answer / "user_daily", output / "user_daily");
        fs::remove(answer);
    } catch (...) {
        clean(answer, start + 150 * second - 100 * millisecond, "cdev");
        throw;
    }
    return 0;
} catch (const std::exception& error) {
    std::fprintf(stderr, "P11 %.200s\n", error.what());
    return 1;
}
