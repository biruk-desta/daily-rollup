#include <algorithm>
#include <cstdarg>
#include <map>
#include <array>
#include <atomic>
#include <bit>
#include <cerrno>
#include <csignal>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <iterator>
#include <memory>
#include <sstream>
#include <stdexcept>
#include <string>
#include <string_view>
#include <thread>
#include <vector>

#include <fcntl.h>
#include <poll.h>
#include <sys/mman.h>
#include <sys/resource.h>
#include <sys/stat.h>
#include <sys/utsname.h>
#include <sys/wait.h>
#include <unistd.h>

#include "parquet.hpp"

namespace fs = std::filesystem;
using Strings = std::vector<std::string>;
using Ns = int64_t;
constexpr Ns second = 1'000'000'000;
constexpr size_t io_block = 1 << 20;

Ns now() {
    timespec ts{};
    clock_gettime(CLOCK_MONOTONIC, &ts);
    return ts.tv_sec * second + ts.tv_nsec;
}

std::vector<std::string> report_lines;

// Every R5 line goes to stderr at once and is kept for the report rows.
void say(const char* format, ...) {
    char buffer[2048];
    va_list args;
    va_start(args, format);
    std::vsnprintf(buffer, sizeof(buffer), format, args);
    va_end(args);
    std::fputs(buffer, stderr);
    std::string line(buffer);
    while (!line.empty() && (line.back() == '\n' || line.back() == '\r')) line.pop_back();
    if (!line.empty()) report_lines.push_back(line);
}

double ms(Ns elapsed) { return double(elapsed) / 1'000'000; }

std::string env(const char* key, std::string fallback) {
    const char* value = std::getenv(key);
    return value && *value ? value : std::move(fallback);
}

std::string one_line(std::string text) {
    auto start = text.find_first_not_of(" \t\r\n");
    if (start == std::string::npos) return "?";
    text = text.substr(start, text.find_last_not_of(" \t\r\n") + 1 - start);
    for (char& c : text) if (c == '\n' || c == '\r' || c == '\t') c = ' ';
    return text;
}

std::string contents(const fs::path& path) {
    std::ifstream file(path);
    return {std::istreambuf_iterator<char>(file), {}};
}

std::string field(const fs::path& path, std::string_view key) {
    std::ifstream file(path);
    for (std::string line; std::getline(file, line);) {
        size_t colon = line.find(':');
        if (colon != std::string::npos && one_line(line.substr(0, colon)) == key)
            return one_line(line.substr(colon + 1));
    }
    return "?";
}

bool word(std::string_view words, std::string_view item, char separator = ' ') {
    while (!words.empty()) {
        size_t end = words.find(separator);
        if (words.substr(0, end) == item) return true;
        if (end == std::string_view::npos) break;
        words.remove_prefix(end + 1);
    }
    return false;
}

struct Mount { std::string root, at, flags, type, super; };

std::string unescape(std::string value) {
    std::string result;
    for (size_t i = 0; i < value.size(); ++i) {
        if (value[i] == '\\' && i + 3 < value.size() &&
            std::string_view(value).substr(i + 1, 3).find_first_not_of("01234567") == std::string::npos) {
            result += char((value[i + 1] - '0') * 64 + (value[i + 2] - '0') * 8 + value[i + 3] - '0');
            i += 3;
        } else {
            result += value[i];
        }
    }
    return result;
}

std::vector<Mount> mounts() {
    std::vector<Mount> result;
    std::ifstream file("/proc/self/mountinfo");
    for (std::string line; std::getline(file, line);) {
        std::istringstream row(line);
        std::string ignored, token;
        Mount m;
        row >> ignored >> ignored >> ignored >> m.root >> m.at >> m.flags;
        while (row >> token && token != "-") {}
        if (row >> m.type >> ignored >> m.super) {
            m.root = unescape(m.root);
            m.at = unescape(m.at);
            result.push_back(std::move(m));
        }
    }
    return result;
}

bool below(std::string_view path, std::string_view directory) {
    return path == directory || (path.starts_with(directory) &&
        (directory.ends_with('/') || path[directory.size()] == '/'));
}

void show_mount(const std::vector<Mount>& all, const char* label, const fs::path& path) {
    std::string absolute = fs::weakly_canonical(path).string();
    const Mount* best = nullptr;
    for (const auto& m : all)
        if (below(absolute, m.at) && (!best || m.at.size() >= best->at.size())) best = &m;
    if (!best) return;
    say("R5 mount %s %s %s;%s\n", label, best->type.c_str(), best->flags.c_str(),
                 best->super.substr(0, 140).c_str());
}

std::string limit(const std::vector<Mount>& all, std::string_view controller,
                  const char* v2, const char* v1, const char* fallback = nullptr) {
    std::ifstream file("/proc/self/cgroup");
    for (std::string row; std::getline(file, row);) {
        size_t first = row.find(':'), last = row.find(':', first + 1);
        if (first == std::string::npos || last == std::string::npos) continue;
        std::string ctr = row.substr(first + 1, last - first - 1);
        bool unified = ctr.empty();
        if (!unified && !word(ctr, controller, ',')) continue;
        std::string group = row.substr(last + 1);
        for (const auto& m : all) {
            if (unified ? m.type != "cgroup2" : m.type != "cgroup" || !word(m.super, controller, ','))
                continue;
            std::string relative = below(group, m.root) ? group.substr(m.root.size()) : group;
            std::string value;
            if (!relative.empty()) value = contents(fs::path(m.at + '/' + relative) / (unified ? v2 : v1));
            if (value.empty()) value = contents(fs::path(m.at) / (unified ? v2 : v1));
            if (value.empty()) continue;
            if (!unified && fallback) {
                std::string period;
                if (!relative.empty()) period = contents(fs::path(m.at + '/' + relative) / fallback);
                if (period.empty()) period = contents(fs::path(m.at) / fallback);
                return one_line(value) + " " + one_line(period);
            }
            return one_line(value);
        }
    }
    return "?";
}

void machine(const fs::path& data, const fs::path& scratch, const fs::path& output) {
    auto all = mounts();
    auto flags = field("/proc/cpuinfo", "flags");
    utsname kernel{};
    uname(&kernel);
    say("R5 cpu=%s avx2/bw/vl=%d/%d/%d vbmi2/vbmi/ifma/vnni/bmi2=%d/%d/%d/%d/%d online=%s affinity=%s(%s) page=%ld kernel=%s\n",
                 field("/proc/cpuinfo", "model name").c_str(), word(flags, "avx2"), word(flags, "avx512bw"),
                 word(flags, "avx512vl"), word(flags, "avx512_vbmi2"), word(flags, "avx512vbmi"),
                 word(flags, "avx512ifma"), word(flags, "avx512_vnni"), word(flags, "bmi2"), one_line(contents("/sys/devices/system/cpu/online")).c_str(),
                 field("/proc/self/status", "Cpus_allowed_list").c_str(),
                 field("/proc/self/status", "Cpus_allowed").c_str(), sysconf(_SC_PAGESIZE), kernel.release);
    auto cpu = limit(all, "cpu", "cpu.max", "cpu.cfs_quota_us", "cpu.cfs_period_us");
    auto memory = limit(all, "memory", "memory.max", "memory.limit_in_bytes");
    say("R5 cg cpu=%s mem=%s thp=%s defrag=%s\n", cpu.c_str(), memory.c_str(),
                 one_line(contents("/sys/kernel/mm/transparent_hugepage/enabled")).c_str(),
                 one_line(contents("/sys/kernel/mm/transparent_hugepage/defrag")).c_str());
    show_mount(all, "data", data);
    show_mount(all, "scratch", scratch);
    show_mount(all, "output", output);
}

struct File {
    int fd;
    File(const fs::path& path, int flags) : fd(open(path.c_str(), flags, 0644)) {
        if (fd < 0) throw std::runtime_error(path.string() + ": " + std::strerror(errno));
    }
    File(const File&) = delete;
    ~File() { if (fd >= 0) close(fd); }
    void finish() {
        int closing = fd;
        fd = -1;
        if (close(closing) != 0) throw std::runtime_error("close failed");
    }
};

std::vector<fs::path> files(const fs::path& directory) {
    std::vector<fs::path> result;
    for (const auto& entry : fs::recursive_directory_iterator(directory))
        if (entry.is_regular_file()) result.push_back(entry.path().lexically_relative(directory));
    std::sort(result.begin(), result.end());
    return result;
}

long cache_resident = -1, cache_sampled = -1;

void cache(const char* label, const fs::path& dir) {
    size_t page = sysconf(_SC_PAGESIZE), sampled = 0, resident = 0, opaque = 0;
    for (const auto& path : files(dir)) {
        size_t bytes = fs::file_size(dir / path);
        if (!bytes) continue;
        File file(dir / path, O_RDONLY);
        size_t pages = (bytes + page - 1) / page;
        size_t stride = std::max<size_t>(1, (pages + 127) / 128);
#ifdef __linux__
        // Linux can report non-owned, non-writable files as all resident.
        struct stat info{};
        if (fstat(file.fd, &info) == 0 && info.st_uid != geteuid() && access((dir / path).c_str(), W_OK) != 0) {
            opaque += (pages + stride - 1) / stride;
            continue;
        }
#endif
        void* mapped = mmap(nullptr, bytes, PROT_READ, MAP_PRIVATE, file.fd, 0);
        if (mapped == MAP_FAILED) continue;
        for (size_t i = 0; i < pages; i += stride) {
#ifdef __APPLE__
            char vector;
#else
            unsigned char vector;
#endif
            if (mincore(static_cast<char*>(mapped) + i * page, page, &vector) == 0) {
                ++sampled;
                resident += vector & 1;
            }
        }
        munmap(mapped, bytes);
    }
    if (std::string_view(label) == "data-initial") cache_resident = resident, cache_sampled = sampled;
    say("R5 cache %s %zu/%zu opaque=%zu\n", label, resident, sampled, opaque);
}

template<class Work> void parallel(unsigned threads, size_t jobs, Work work) {
    std::atomic<size_t> next = 0;
    std::vector<std::jthread> pool;
    auto run = [&] {
        for (size_t job; (job = next.fetch_add(1, std::memory_order_relaxed)) < jobs;) work(job);
    };
    for (unsigned i = 1; i < std::min<size_t>(jobs, threads); ++i) pool.emplace_back(run);
    run();
}

void micro() {
    size_t bytes = std::clamp(std::stoull(env("R5_MEM_MIB", "1024")), 1ULL, 1024ULL) << 20;
    auto src = std::make_unique_for_overwrite<char[]>(bytes);
    auto dst = std::make_unique_for_overwrite<char[]>(bytes);
    parallel(8, 8, [&](size_t i) {
        std::memset(src.get() + i * bytes / 8, 17, bytes / 8);
        std::memset(dst.get() + i * bytes / 8, 31, bytes / 8);
    });
    auto bandwidth = [&](unsigned threads, bool copy) {
        Ns start = now();
        parallel(threads, threads, [&](size_t i) {
            size_t count = bytes / threads, at = i * count;
            if (copy) std::memcpy(dst.get() + at, src.get() + at, count);
            else std::memset(dst.get() + at, 63, count);
        });
        asm volatile("" : : "r"(dst.get()) : "memory");
        return double(bytes) / (now() - start);
    };
    double set1 = bandwidth(1, false), set8 = bandwidth(8, false);
    double copy1 = bandwidth(1, true), copy8 = bandwidth(8, true);
    uint64_t value = 1;
    constexpr unsigned iterations = 100'000'000;
    Ns start = now();
    for (unsigned i = 0; i < iterations; ++i)
        value = std::rotl((value ^ i) * 0x94d049bb133111ebULL, 13);
    double scalar = double(now() - start) / iterations;
    std::fprintf(stderr, "R5 micro %zuMiB set1/8=%.1f/%.1f copy1/8=%.1f/%.1f GB/s scalar=%.2f ns/iter sum=%llx\n",
                 bytes >> 20, set1, set8, copy1, copy8, scalar, static_cast<unsigned long long>(value));
}

struct Child {
    int status = -1;
    rusage usage{};
    double elapsed = 0;
    std::string log;
    Ns started = 0;
    bool ok() const { return status == 0; }
};

Child run(const Strings& args, Ns deadline, unsigned threads = 8, const char* flag = nullptr) {
    Child result;
    if (now() >= deadline) return result;
    std::vector<char*> argv;
    for (const auto& arg : args) argv.push_back(const_cast<char*>(arg.c_str()));
    argv.push_back(nullptr);
    int pipefd[2];
    if (pipe(pipefd) != 0) throw std::runtime_error("pipe failed");
    result.started = now();
    pid_t pid = fork();
    if (pid == 0) {
        close(pipefd[0]);
        if (dup2(pipefd[1], STDERR_FILENO) < 0) _exit(126);
        close(pipefd[1]);
        setenv("NCPU", threads == 4 ? "4" : "8", 1);
        setenv("SHOPPING_TRACE", "1", 1);
        unsetenv("SHOPPING_SCALAR");
        for (const char* key : {"R5_AVX2", "R5_AVX512", "R5_POPULATE", "R5_PREFAULT", "R5_NO_THP", "R5_WRONG_ROW"})
            unsetenv(key);
        if (flag) setenv(flag, "1", 1);
        execv(argv[0], argv.data());
        _exit(127);
    }
    close(pipefd[1]);
    if (pid < 0) {
        close(pipefd[0]);
        throw std::runtime_error("fork failed");
    }
    fcntl(pipefd[0], F_SETFL, O_NONBLOCK);
    auto drain = [&] {
        char buffer[4096];
        for (ssize_t count; (count = read(pipefd[0], buffer, sizeof(buffer))) > 0;)
            if (result.log.size() < 16384) result.log.append(buffer, count);
    };
    for (;;) {
        drain();
        pid_t done = wait4(pid, &result.status, WNOHANG, &result.usage);
        if (done == pid) break;
        if (done < 0 && errno != EINTR) { result.status = -1; break; }
        Ns left = deadline - now();
        if (left <= 0) {
            kill(pid, SIGKILL);
            while (wait4(pid, &result.status, 0, &result.usage) < 0 && errno == EINTR) {}
            result.status = -1;
            break;
        }
        pollfd event{pipefd[0], POLLIN, 0};
        poll(&event, 1, std::min(50, int(ms(left)) + 1));
    }
    result.elapsed = ms(now() - result.started);
    drain();
    close(pipefd[0]);
    return result;
}

void write_bytes(File& file, const char* bytes, size_t pending) {
    while (pending) {
        ssize_t count = write(file.fd, bytes, pending);
        if (count < 0 && errno == EINTR) continue;
        if (count <= 0) throw std::runtime_error("write failed");
        bytes += count;
        pending -= count;
    }
}

void read_files(const fs::path& src, const std::string& mode, const fs::path& dest) {
    int flags = O_RDONLY;
    bool copying = mode == "copy";
    if (mode == "direct") {
#ifdef O_DIRECT
        flags |= O_DIRECT;
#else
        throw std::runtime_error("direct unavailable");
#endif
    }
    void* raw = nullptr;
    if (posix_memalign(&raw, 4096, io_block) != 0) throw std::bad_alloc();
    std::unique_ptr<char, decltype(&free)> buffer(static_cast<char*>(raw), free);
    for (const auto& path : files(src)) {
        File input(src / path, flags);
        std::unique_ptr<File> output;
        if (copying) {
            fs::create_directories((dest / path).parent_path());
            output = std::make_unique<File>(dest / path, O_WRONLY | O_CREAT | O_TRUNC);
        }
        size_t remaining = fs::file_size(src / path);
        while (remaining) {
            ssize_t count = read(input.fd, buffer.get(), io_block);
            if (count < 0 && errno == EINTR) continue;
            if (count <= 0) throw std::runtime_error(std::string("read failed: ") + std::strerror(errno));
            if (output) write_bytes(*output, buffer.get(), count);
            remaining -= count;
        }
    }
}

void touch_files(const fs::path& src) {
    unsigned sum = 0;
    size_t page = sysconf(_SC_PAGESIZE);
    for (const auto& path : files(src)) {
        size_t bytes = fs::file_size(src / path);
        if (!bytes) continue;
        File input(src / path, O_RDONLY);
        void* raw = mmap(nullptr, bytes, PROT_READ, MAP_PRIVATE, input.fd, 0);
        if (raw == MAP_FAILED) throw std::runtime_error("mmap failed");
        auto* data = static_cast<const volatile unsigned char*>(raw);
        for (size_t i = 0; i < bytes; i += page) sum += data[i];
        munmap(raw, bytes);
    }
    std::fprintf(stderr, "%u\n", sum);
}

void output_only(const fs::path& src, const fs::path& dest) {
    struct Blob { fs::path path; std::string bytes; };
    std::vector<Blob> blobs;
    size_t bytes = 0;
    for (const auto& path : files(src)) {
        std::string data(fs::file_size(src / path), '\0');
        std::ifstream file(src / path, std::ios::binary);
        if (!file.read(data.data(), data.size())) throw std::runtime_error("output sample read failed");
        bytes += data.size();
        blobs.push_back({path, std::move(data)});
    }
    for (const auto& blob : blobs) fs::create_directories((dest / blob.path).parent_path());
    Ns start = now();
    std::atomic<bool> failed = false;
    parallel(8, blobs.size(), [&](size_t i) {
        const auto& blob = blobs[i];
        try {
            File out(dest / blob.path, O_WRONLY | O_TRUNC | O_CREAT);
            write_bytes(out, blob.bytes.data(), blob.bytes.size());
            out.finish();
        } catch (const std::exception&) {
            failed = true;
        }
    });
    if (failed) throw std::runtime_error("output-only write failed");
    std::fprintf(stderr, "%zu %.3f\n", bytes, ms(now() - start));
}

template<class T> T median(std::vector<T> values) {
    std::sort(values.begin(), values.end());
    return values[values.size() / 2];
}

std::string show(double value) {
    if (value < 0) return "NA";
    char text[32];
    std::snprintf(text, sizeof(text), "%.1f", value);
    return text;
}

double io_first = -1, io_warm = -1;

void input_io(const std::string& self, const fs::path& data, const fs::path& work, Ns deadline) {
    fs::path copy = work / "dev-copy";
    auto read_trial = [&](const fs::path& source, const char* mode, const fs::path& target) {
        return run({self, "--read", source.string(), mode, target.string()},
                   std::min(deadline, now() + 5 * second));
    };
    auto time = [](const Child& child) { return child.ok() ? child.elapsed : -1.0; };
    auto data_first = read_trial(data, "touch", {});
    auto copied = read_trial(data, "copy", copy);
    if (!copied.ok()) {
        say("R5 io data first=%sms major=%ld; copy failed %d\n",
                     show(time(data_first)).c_str(), data_first.usage.ru_majflt, copied.status);
        return;
    }
    cache("scratch-copy", copy);
    auto copy_first = read_trial(copy, "touch", {});
    auto readings = [&](const fs::path& dir, const char* mode) {
        std::vector<double> samples;
        for (unsigned i = 0; i < 3; ++i) {
            double t = time(read_trial(dir, mode, {}));
            if (t < 0) return t;
            samples.push_back(t);
        }
        return median(samples);
    };
    double data_warm = readings(data, "touch"), copy_warm = readings(copy, "touch");
    double data_direct = readings(data, "direct"), copy_direct = readings(copy, "direct");
    io_first = time(data_first);
    io_warm = data_warm;
    say("R5 io touch/warm/direct ms data=%s/%s/%s scratch=%s/%s/%s first_major=%ld/%ld\n",
                 show(time(data_first)).c_str(), show(data_warm).c_str(), show(data_direct).c_str(),
                 show(time(copy_first)).c_str(), show(copy_warm).c_str(), show(copy_direct).c_str(),
                 data_first.usage.ru_majflt, copy_first.usage.ru_majflt);
}

struct Sample {
    // Startup, maps, pre-fault, ingest, dedup, clicks, sessions, finish, exit.
    std::array<double, 9> phases{};
    double total = 0;
    long rss = 0, minor = 0, major = 0;
    std::string scanner;
};

Sample sample(const Child& child) {
    Sample result;
    result.total = child.elapsed;
#ifdef __APPLE__
    result.rss = child.usage.ru_maxrss / (1 << 20);
#else
    result.rss = child.usage.ru_maxrss / 1024;
#endif
    result.minor = child.usage.ru_minflt;
    result.major = child.usage.ru_majflt;
    constexpr std::array names = {"start", "mappings", "prefault", "event_ingest_links_catalogs", "event_dedup_route",
                                  "click_scan_validation", "sessions_user_output", "finish"};
    std::istringstream log(child.log);
    Ns finish = child.started;
    for (std::string line; std::getline(log, line);) {
        if (line.starts_with("CSV ")) result.scanner = line.substr(4);
        if (!line.starts_with("phase ")) continue;
        std::istringstream row(line);
        std::string tag, name;
        Ns at;
        double elapsed;
        if (!(row >> tag >> name >> at >> elapsed)) continue;
        auto it = std::find(names.begin(), names.end(), name);
        if (it == names.end()) continue;
        result.phases[it - names.begin()] = name == "start" ? ms(at - child.started) : elapsed * 1000;
        finish = at;
    }
    result.phases.back() = std::max(0.0, child.elapsed - ms(finish - child.started));
    return result;
}

struct Result {
    size_t n = 0;
    std::string scanner, stop;
    double total = -1;
    std::array<double, 9> phases{};
    long rss = -1, minor = -1, major = -1;
};
std::map<std::string, Result> results;

std::string fixed(double value) {
    char text[32];
    std::snprintf(text, sizeof(text), "%.0f", value);
    return text;
}

struct Variant {
    const char* name;
    const std::string& engine;
    unsigned threads;
    fs::path out;
    const char* flag = nullptr;
    const char* scanner = nullptr;
};

bool measure(const Variant& v, const fs::path& data, Ns deadline) {
    std::vector<Sample> samples;
    for (unsigned i = 0; i < 3; ++i) {
        auto child = run({v.engine, data.string(), (v.out / std::to_string(i)).string()},
                         std::min(deadline, now() + 14 * second), v.threads, v.flag);
        auto trial = sample(child);
        if (!child.ok() || trial.scanner.empty() || (v.scanner && trial.scanner != v.scanner)) {
            results[v.name].stop = "s" + std::to_string(i + 1) + ":" + std::to_string(child.status) + ":" +
                one_line(child.log).substr(0, 60);
            say("R5 %s sample=%u stop=%d %s\n", v.name, i + 1, child.status,
                         one_line(child.log).substr(0, 140).c_str());
            break;
        }
        samples.push_back(std::move(trial));
    }
    if (samples.empty()) return false;
    auto middle = [&](auto get) {
        using T = decltype(get(samples.front()));
        std::vector<T> values;
        for (const auto& s : samples) values.push_back(get(s));
        return median(values);
    };
    Result& r = results[v.name];
    r.n = samples.size();
    r.scanner = samples.front().scanner;
    r.total = middle([](const auto& s) { return s.total; });
    for (size_t i = 0; i < r.phases.size(); ++i) r.phases[i] = middle([i](const auto& s) { return s.phases[i]; });
    r.rss = middle([](const auto& s) { return s.rss; });
    r.minor = middle([](const auto& s) { return s.minor; });
    r.major = middle([](const auto& s) { return s.major; });
    std::string line = "R5 " + std::string(v.name) + " n=" + std::to_string(r.n) + " " + r.scanner + " " +
        fixed(r.total) + " [";
    for (size_t i = 0; i < r.phases.size(); ++i) line += (i ? "/" : "") + fixed(r.phases[i]);
    line += "] " + std::to_string(r.rss) + "MiB f=" + std::to_string(r.minor) + "/" + std::to_string(r.major);
    say("%s\n", line.c_str());
    return samples.size() == 3;
}

// Plain ASCII without quotes, commas or backslashes survives CSV and JSON display.
std::string printable(std::string text) {
    std::string result;
    for (char c : text) {
        auto u = static_cast<unsigned char>(c);
        if (u < 32 || u > 126) c = ' ';
        else if (c == ',') c = ';';
        else if (c == '"' || c == '\'' || c == '\\') c = '_';
        if (c == ' ' && !result.empty() && result.back() == ' ') continue;
        result += c;
    }
    return result;
}

std::string lines_with(std::initializer_list<std::string_view> keys) {
    std::string text;
    for (const auto& line : report_lines) {
        std::string_view body = line;
        if (body.starts_with("R5 ")) body.remove_prefix(3);
        for (auto key : keys)
            if (body.find(key) != std::string_view::npos) {
                text += (text.empty() ? "" : " | ") + std::string(body);
                break;
            }
    }
    return text;
}

int64_t whole(double value) { return value < 0 ? -1 : static_cast<int64_t>(value + 0.5); }

void write_report(const fs::path& path, double elapsed) {
    struct Row { std::string text; int64_t a, b, c; };
    auto total = [](const char* name) -> double {
        auto it = results.find(name);
        return it == results.end() ? -1 : it->second.total;
    };
    std::vector<Row> rows;
    auto flags = field("/proc/cpuinfo", "flags");
    long cpus = long(std::thread::hardware_concurrency());
    {
        std::string text = "shipped ";
        auto it = results.find("shipped");
        if (it != results.end()) {
            const auto& r = it->second;
            text += "ms=" + fixed(r.total) + " " + r.scanner + " n=" + std::to_string(r.n) + " st/map/pf/ing/dd/clk/ses/fin/exit=";
            for (size_t i = 0; i < r.phases.size(); ++i) text += (i ? "/" : "") + fixed(r.phases[i]);
            if (!r.stop.empty()) text += " stop=" + r.stop;
        } else {
            text += "NA";
        }
        text += " threads=" + std::to_string(cpus) + " ncpu=" + env("NCPU", "?") + " " + lines_with({"gen "});
        auto r = it == results.end() ? Result{} : it->second;
        rows.push_back({text, whole(r.total), whole(r.phases[3]), whole(r.phases[6])});
    }
    {
        std::string text = "variants ms";
        for (const char* name : {"new", "t4", "avx2", "avx512", "read", "populate", "prefault", "thpoff", "scratch", "shipped"}) {
            auto it = results.find(name);
            text += std::string(" ") + name + "=";
            if (it == results.end()) text += "NA";
            else {
                text += fixed(it->second.total) + "/" + std::to_string(it->second.n);
                if (!it->second.stop.empty()) text += "/stop";
            }
        }
        rows.push_back({text, whole(total("new")), whole(total("read")), whole(total("avx512"))});
    }
    {
        std::string text;
        for (const char* name : {"shipped", "read"}) {
            auto it = results.find(name);
            if (it == results.end()) { text += std::string(name) + "=NA "; continue; }
            text += std::string(name) + " f=" + std::to_string(it->second.minor) + "/" +
                std::to_string(it->second.major) + " mem=" + std::to_string(it->second.rss) + "MiB ";
        }
        text += lines_with({"micro "});
        auto shipped = results.count("shipped") ? results["shipped"] : Result{};
        auto read = results.count("read") ? results["read"] : Result{};
        rows.push_back({text, shipped.rss, read.rss, shipped.major});
    }
    rows.push_back({lines_with({"cpu=", "cg "}), cpus, word(flags, "avx512_vbmi2"), word(flags, "avx512bw")});
    rows.push_back({lines_with({"cache ", "io "}) + " | " + lines_with({"mount "}), cache_resident, cache_sampled, whole(io_first)});
    {
        std::string text = "elapsed=" + fixed(elapsed) + "ms " +
            lines_with({"write n=", "stop", "unsupported", "unavailable", "failed", "budget"});
        int64_t problems = 0;
        for (const auto& line : report_lines)
            for (std::string_view key : {"stop", "unavailable", "failed", "budget"})
                if (line.find(key) != std::string::npos) { ++problems; break; }
        rows.push_back({text, whole(elapsed), whole(io_warm), problems});
    }
    rows.push_back(rows.front());

    std::vector<std::string> ids, categories;
    for (size_t i = 0; i < rows.size(); ++i) {
        char tag[8];
        std::snprintf(tag, sizeof(tag), "!%02zu ", i);
        std::string text = printable(rows[i].text);
        ids.push_back(tag + text.substr(0, 160));
        categories.push_back(text.size() > 160 ? text.substr(160, 600) : std::string("-"));
        say("R5 row %s | %s | %lld %lld %lld\n", ids.back().c_str(), categories.back().c_str(),
            static_cast<long long>(rows[i].a), static_cast<long long>(rows[i].b), static_cast<long long>(rows[i].c));
    }
    static constexpr parquet::Column schema[] = {{"user_id", parquet::Column::Text, 0},
        {"day", parquet::Column::Int32, 6}, {"sessions", parquet::Column::Int64},
        {"events", parquet::Column::Int64}, {"revenue_cents", parquet::Column::Int64},
        {"top_category", parquet::Column::Text, 0, true}};
    parquet::File file(path, schema);
    file.append(int32_t(rows.size()), [&](size_t column, parquet::Bytes& bytes) {
        if (column == 5) parquet::definitions(bytes, rows.size(), [](size_t) { return true; });
        for (size_t i = 0; i < rows.size(); ++i) {
            switch (column) {
                case 0: parquet::plain(bytes, ids[i]); break;
                case 1: parquet::little_endian(bytes, int32_t(10957 + i)); break;
                case 2: parquet::little_endian(bytes, rows[i].a); break;
                case 3: parquet::little_endian(bytes, rows[i].b); break;
                case 4: parquet::little_endian(bytes, rows[i].c); break;
                case 5: parquet::plain(bytes, categories[i]); break;
            }
        }
    });
    file.finish();
}

int main(int argc, char** argv) try {
    if (argc == 2 && std::string_view(argv[1]) == "--micro") { micro(); return 0; }
    if (argc == 5 && std::string_view(argv[1]) == "--read") {
        if (std::string_view(argv[3]) == "touch") touch_files(argv[2]);
        else read_files(argv[2], argv[3], argv[4]);
        return 0;
    }
    if (argc == 4 && std::string_view(argv[1]) == "--write") { output_only(argv[2], argv[3]); return 0; }
    if (argc == 3 && std::string_view(argv[1]) == "--clean") { fs::remove_all(argv[2]); return 0; }

    Ns start = now(), deadline = start + 132 * second;
    std::string self = fs::absolute(argv[0]).string();
    fs::path data = env("DATA_DIR", "/data"), output = env("OUTPUT_DIR", "/output");
    fs::path scratch = env("SCRATCH_DIR", "/scratch");
    std::string normal = env("R5_NORMAL", "/shopping"), tuned = env("R5_TUNED", "/shopping-tuned");
    std::string new_engine = env("R5_NEW_ENGINE", "/shopping-new");
    std::string read_engine = env("R5_READ_ENGINE", "/shopping-read"), generator = env("R5_GENERATOR", "/generate");
    fs::path output_trial;
    try {
        machine(data, scratch, output);
        cache("data-initial", data);
        std::string temp = (scratch / "r5-XXXXXX").string();
        if (!mkdtemp(temp.data())) throw std::runtime_error("cannot create probe scratch directory");
        fs::path work = temp, synth = work / "generated", local_out = work / "parquet";
        output_trial = output / ("." + fs::path(temp).filename().string());
        input_io(self, data, work, std::min(deadline, now() + 11 * second));
        auto bench = run({self, "--micro"}, std::min(deadline, now() + 7 * second));
        if (bench.ok()) say("%s", bench.log.c_str());
        else say("R5 micro unavailable (%d)\n", bench.status);
        unsigned long long events = std::clamp(std::stoull(env("R5_EVENTS", "20000000")), 1000ULL, 20'000'000ULL);
        auto generate = [&](unsigned long long count, int seconds) {
            return run({generator, "--seed", "20261120", "--events", std::to_string(count), "--clicks",
                        std::to_string(count / 4), "--threads", "8", "--output", synth.string()},
                       std::min(deadline, now() + seconds * second));
        };
        auto generated = generate(events, 32);
        if (!generated.ok() && events > 5'000'000) {
            fs::remove_all(synth);
            events = 5'000'000;
            generated = generate(events, 15);
        }
        if (!generated.ok()) throw std::runtime_error("generator failed " + one_line(generated.log).substr(0, 140));
        say("R5 gen events=%llu clicks=%llu seed=20261120 ms=%.0f\n", events, events / 4,
                     generated.elapsed);

        fs::path shipped = output_trial / "shipped";
        measure({"shipped", normal, 8, shipped}, synth, deadline);
        // Candidate 48ddd03 with shipped settings, early so the budget never cuts it.
        measure({"new", new_engine, 8, output_trial / "new"}, synth, deadline);
        measure({"scratch", normal, 8, local_out}, synth, deadline);
        if (fs::exists(local_out / "0")) {
            std::vector<double> writes;
            size_t bytes = 0;
            for (unsigned i = 0; i < 3 && now() < deadline; ++i) {
                auto replay = run({self, "--write", (local_out / "0").string(),
                                   (output_trial / "write" / std::to_string(i)).string()},
                                  std::min(deadline, now() + 7 * second));
                double time;
                if (!(replay.ok() && (std::istringstream(replay.log) >> bytes >> time))) break;
                writes.push_back(time);
            }
            if (!writes.empty()) say("R5 write n=%zu %.1fms %zuB\n", writes.size(),
                                               median(writes), bytes);
        }

        auto flags = field("/proc/cpuinfo", "flags");
        const std::array variants = {
            Variant{"read", read_engine, 8, output_trial / "read"}, Variant{"t4", normal, 4, output_trial / "t4"},
            Variant{"populate", tuned, 8, output_trial / "populate", "R5_POPULATE"},
            Variant{"prefault", tuned, 8, output_trial / "prefault", "R5_PREFAULT"},
            Variant{"thpoff", tuned, 8, output_trial / "thpoff", "R5_NO_THP"},
            Variant{"avx2", tuned, 8, output_trial / "avx2", "R5_AVX2", "AVX2"},
            Variant{"avx512", tuned, 8, output_trial / "avx512", "R5_AVX512", "AVX-512BW"},
        };
        for (const auto& variant : variants) {
            if (variant.scanner && !word(flags, std::string_view(variant.name) == "avx2" ? "avx2" : "avx512bw")) {
                say("R5 %s unsupported\n", variant.name);
                continue;
            }
            if (now() + 2 * second > deadline) {
                say("R5 budget stopped before %s\n", variant.name);
                break;
            }
            measure(variant, synth, deadline);
        }
    } catch (const std::exception& error) {
        say("R5 diagnostics stopped: %.180s\n", error.what());
    }

    if (!output_trial.empty())
        run({self, "--clean", output_trial.string()}, std::min(start + 135 * second, now() + 3 * second));
    // The shipped engine writes the exact dev tables; the report rides in extra rows.
    auto right = run({normal, data.string(), output.string()}, start + 148 * second, 8);
    if (!right.ok()) {
        say("R5 dev output failed (%d): %.180s\n", right.status, right.log.c_str());
        return 1;
    }
    double elapsed = ms(now() - start);
    say("R5 exact dev tables written; report rows follow; %.2fs elapsed\n", elapsed / 1000);
    write_report(output / "user_daily" / "!r5.parquet", elapsed);
    return 0;
} catch (const std::exception& error) {
    say("R5 %s\n", error.what());
    return 1;
}
