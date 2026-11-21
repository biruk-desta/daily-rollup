#include <algorithm>
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
    std::fprintf(stderr, "R5 mount %s %s %s;%s\n", label, best->type.c_str(), best->flags.c_str(),
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
    std::fprintf(stderr, "R5 cpu=%s avx2/bw/vl=%d/%d/%d online=%s affinity=%s(%s) page=%ld kernel=%s\n",
                 field("/proc/cpuinfo", "model name").c_str(), word(flags, "avx2"), word(flags, "avx512bw"),
                 word(flags, "avx512vl"), one_line(contents("/sys/devices/system/cpu/online")).c_str(),
                 field("/proc/self/status", "Cpus_allowed_list").c_str(),
                 field("/proc/self/status", "Cpus_allowed").c_str(), sysconf(_SC_PAGESIZE), kernel.release);
    auto cpu = limit(all, "cpu", "cpu.max", "cpu.cfs_quota_us", "cpu.cfs_period_us");
    auto memory = limit(all, "memory", "memory.max", "memory.limit_in_bytes");
    std::fprintf(stderr, "R5 cg cpu=%s mem=%s thp=%s defrag=%s\n", cpu.c_str(), memory.c_str(),
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
    std::fprintf(stderr, "R5 cache %s %zu/%zu opaque=%zu\n", label, resident, sampled, opaque);
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
        std::fprintf(stderr, "R5 io data first=%sms major=%ld; copy failed %d\n",
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
    std::fprintf(stderr, "R5 io touch/warm/direct ms data=%s/%s/%s scratch=%s/%s/%s first_major=%ld/%ld\n",
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
            std::fprintf(stderr, "R5 %s sample=%u stop=%d %s\n", v.name, i + 1, child.status,
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
    std::fprintf(stderr, "R5 %s n=%zu %s %.1f [", v.name, samples.size(), samples.front().scanner.c_str(),
                 middle([](const auto& s) { return s.total; }));
    for (size_t i = 0; i < samples.front().phases.size(); ++i)
        std::fprintf(stderr, "%s%.1f", i ? "/" : "", middle([i](const auto& s) { return s.phases[i]; }));
    std::fprintf(stderr, "] %ldMiB f=%ld/%ld\n", middle([](const auto& s) { return s.rss; }),
                 middle([](const auto& s) { return s.minor; }), middle([](const auto& s) { return s.major; }));
    return samples.size() == 3;
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
        if (bench.ok()) std::fputs(bench.log.c_str(), stderr);
        else std::fprintf(stderr, "R5 micro unavailable (%d)\n", bench.status);
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
        std::fprintf(stderr, "R5 gen events=%llu clicks=%llu seed=20261120 ms=%.0f\n", events, events / 4,
                     generated.elapsed);

        fs::path shipped = output_trial / "shipped";
        measure({"shipped", normal, 8, shipped}, synth, deadline);
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
            if (!writes.empty()) std::fprintf(stderr, "R5 write n=%zu %.1fms %zuB\n", writes.size(),
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
                std::fprintf(stderr, "R5 %s unsupported\n", variant.name);
                continue;
            }
            if (now() + 2 * second > deadline) {
                std::fprintf(stderr, "R5 budget stopped before %s\n", variant.name);
                break;
            }
            measure(variant, synth, deadline);
        }
    } catch (const std::exception& error) {
        std::fprintf(stderr, "R5 diagnostics stopped: %.180s\n", error.what());
    }

    if (!output_trial.empty())
        run({self, "--clean", output_trial.string()}, std::min(start + 135 * second, now() + 3 * second));
    auto wrong = run({tuned, data.string(), output.string()}, start + 148 * second, 8, "R5_WRONG_ROW");
    if (!wrong.ok()) {
        std::fprintf(stderr, "R5 dev output failed (%d): %.180s\n", wrong.status, wrong.log.c_str());
        return 1;
    }
    std::fprintf(stderr, "R5 deliberately changed one dev user_daily.events cell; %.2fs elapsed\n",
                 double(now() - start) / second);
    return 0;
} catch (const std::exception& error) {
    std::fprintf(stderr, "R5 %s\n", error.what());
    return 1;
}
