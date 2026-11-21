#include "parquet.hpp"

#include <algorithm>
#include <array>
#include <atomic>
#include <charconv>
#include <chrono>
#include <cstdlib>
#include <exception>
#include <memory>
#include <mutex>
#include <optional>
#include <thread>
#include <unordered_map>
#include <utility>

#include <fcntl.h>
#include <sys/mman.h>
#include <sys/resource.h>
#include <sys/stat.h>
#include <unistd.h>

#if defined(__x86_64__)
#include <cpuid.h>
#include <immintrin.h>
#elif defined(__aarch64__)
#include <arm_neon.h>
#endif

extern "C" void __gcov_dump(void) __attribute__((weak));
extern "C" int __llvm_profile_write_file(void) __attribute__((weak));

namespace fs = std::filesystem;
using View = std::string_view;
constexpr int64_t day_ms = 86'400'000;
constexpr int64_t week_ms = 7 * day_ms;
constexpr int64_t gap_ms = 1'800'000;
constexpr int64_t bad_time = INT64_MIN;
template<class Category> constexpr Category no_category = std::numeric_limits<Category>::max();

struct Hash {
    size_t operator()(View s) const noexcept {
        uint64_t hash = uint64_t(s.size()) * 0x9e3779b97f4a7c15ULL;
        while (s.size() >= 8) {
            uint64_t word;
            std::memcpy(&word, s.data(), 8);
            hash = (hash ^ word) * 0xbf58476d1ce4e5b9ULL;
            s.remove_prefix(8);
        }
        uint64_t tail = 0;
        if (!s.empty()) std::memcpy(&tail, s.data(), s.size());
        hash = (hash ^ tail) * 0x94d049bb133111ebULL;
        return hash ^ (hash >> 32);
    }
};

template<class V> using StringMap = memory::Map<View, V, Hash>;

class Mapping {
    View data_;
public:
    explicit Mapping(const fs::path& path) {
        int fd = open(path.c_str(), O_RDONLY);
        if (fd == -1) throw std::runtime_error("cannot open " + path.string());
        struct stat info{};
        if (fstat(fd, &info) != 0) {
            close(fd);
            throw std::runtime_error("cannot stat " + path.string());
        }
        if (info.st_size) {
            auto size = static_cast<size_t>(info.st_size);
            void* address = mmap(nullptr, size, PROT_READ, MAP_PRIVATE, fd, 0);
            close(fd);
            if (address == MAP_FAILED) throw std::runtime_error("cannot map " + path.string());
            try {
                memory::remember({address, size});
            } catch (...) {
                munmap(address, size);
                throw;
            }
            data_ = View(static_cast<const char*>(address), size);
        } else {
            close(fd);
        }
    }
    Mapping(const Mapping&) = delete;
    Mapping& operator=(const Mapping&) = delete;
    ~Mapping() {
        if (data_.empty()) return;
        auto region = take();
        munmap(region.address, region.bytes);
    }
    View bytes() const { return data_; }

    memory::Region take() {
        memory::Region region{const_cast<char*>(data_.data()), data_.size()};
        if (region.bytes) memory::forget(region.address);
        data_ = {};
        return region;
    }
};

struct Input {
    std::vector<std::unique_ptr<Mapping>> maps;
    std::vector<View> chunks;
    size_t bytes = 0;

    Input(const fs::path& path, bool directory) {
        if (directory) {
            std::vector<fs::path> paths;
            for (const auto& entry : fs::directory_iterator(path))
                if (entry.path().extension() == ".csv" && entry.is_regular_file())
                    paths.push_back(entry.path());
            std::sort(paths.begin(), paths.end());
            for (const auto& file : paths) add(file);
        } else {
            add(path);
        }
    }

    void add(const fs::path& path) {
        maps.push_back(std::make_unique<Mapping>(path));
        View input = maps.back()->bytes();
        size_t header = input.find('\n');
        if (header == View::npos) return;
        input.remove_prefix(header + 1);
        bytes += input.size();
        constexpr size_t chunk_size = 2 << 20;
        while (!input.empty()) {
            size_t end = input.size();
            if (end > chunk_size) {
                size_t newline = input.find('\n', chunk_size);
                if (newline != View::npos) end = newline + 1;
            }
            chunks.push_back(input.substr(0, end));
            input.remove_prefix(end);
        }
    }

    void take_mappings(std::vector<memory::Region>& regions) const {
        regions.reserve(regions.size() + maps.size());
        for (const auto& map : maps) {
            if (!map->bytes().empty()) regions.push_back(map->take());
        }
    }
};


template<class Work, class Finish = std::nullptr_t>
void parallel(unsigned threads, size_t tasks, Work work, Finish finish = nullptr) {
    if (!tasks) return;
    unsigned workers = std::min<size_t>(threads, tasks);
    std::atomic<size_t> next{0};
    std::mutex error_mutex;
    std::exception_ptr error;
    auto run = [&](unsigned worker) {
        SHOPPING_TIME(timing::Local local;)
        try {
            for (size_t task; (task = next.fetch_add(1, std::memory_order_relaxed)) < tasks;)
                work(worker, task);
            if constexpr (!std::is_same_v<Finish, std::nullptr_t>) finish(worker);
        } catch (...) {
            next.store(tasks, std::memory_order_relaxed);
            std::lock_guard lock(error_mutex);
            if (!error) error = std::current_exception();
        }
    };
    {
        std::vector<std::jthread> pool;
        pool.reserve(workers - 1);
        for (unsigned i = 1; i < workers; ++i) pool.emplace_back(run, i);
        run(0);
    }
    if (error) std::rethrow_exception(error);
}

void release_parallel(const std::vector<memory::Region>& regions, unsigned threads) {
    constexpr size_t piece_bytes = 4 << 20;
    std::vector<memory::Region> pieces;
    for (const auto& [address, bytes] : regions) {
        for (size_t offset = 0; offset < bytes; offset += piece_bytes)
            pieces.push_back({static_cast<char*>(address) + offset, std::min(bytes - offset, piece_bytes)});
    }
    auto method = memory::settings().method;
    parallel(threads, pieces.size(), [&](unsigned, size_t i) { memory::release(pieces[i], method); });
}

template<size_t N> using Fields = std::array<View, N>;

template<size_t N> bool split(View line, Fields<N>& fields) {
    for (size_t i = 0; i < N; ++i) {
        size_t comma = line.find(',');
        if ((comma == View::npos) != (i == N - 1)) return false;
        View value = line.substr(0, comma);
        if (value.starts_with('"')) {
            if (value.size() < 2 || !value.ends_with('"')) return false;
            value.remove_prefix(1);
            value.remove_suffix(1);
        }
        if (value.find('"') != View::npos) return false;
        fields[i] = value;
        if (comma != View::npos) line.remove_prefix(comma + 1);
    }
    return true;
}

template<size_t N, class Row> void scan_scalar(View chunk, Row row) {
    while (!chunk.empty()) {
        size_t newline = chunk.find('\n');
        View line = chunk.substr(0, newline);
        Fields<N> fields;
        if (split(line, fields)) row(line, fields);
        if (newline == View::npos) break;
        chunk.remove_prefix(newline + 1);
    }
}

enum class Scanner { Scalar, Neon, Avx2, Avx512 };

Scanner select_scanner() {
    if (std::getenv("SHOPPING_SCALAR")) return Scanner::Scalar;
#if defined(__x86_64__)
    unsigned a, b, c, d;
    constexpr unsigned vector_support = bit_AVX | bit_OSXSAVE | bit_POPCNT;
    if (!__get_cpuid(1, &a, &b, &c, &d) || (c & vector_support) != vector_support ||
        !__get_cpuid_count(7, 0, &a, &b, &c, &d)) return Scanner::Scalar;
    unsigned xcr0, upper;
    __asm__ volatile("xgetbv" : "=a"(xcr0), "=d"(upper) : "c"(0));
    constexpr unsigned avx512 = bit_AVX512F | bit_AVX512BW;
    if ((xcr0 & 0xe6) == 0xe6 && (b & avx512) == avx512) return Scanner::Avx512;
    if ((xcr0 & 0x06) == 0x06 && (b & bit_AVX2)) return Scanner::Avx2;
#elif defined(__aarch64__)
    return Scanner::Neon;
#endif
    return Scanner::Scalar;
}

const Scanner scanner = select_scanner();

class Trace {
    bool enabled_ = [] {
        const char* value = std::getenv("SHOPPING_TRACE");
        return value && View(value) == "1";
    }();
    int64_t time_ = 0;
    rusage usage_{};
    SHOPPING_COUNT(timing::Hardware hardware_;)

    static int64_t micros(timeval t) { return int64_t(t.tv_sec) * 1'000'000 + t.tv_usec; }

public:
    Trace() {
        if (!enabled_) return;
        constexpr const char* names[] = {"scalar", "NEON", "AVX2", "AVX-512BW"};
        std::fprintf(stderr, "CSV %s\n", names[static_cast<unsigned>(scanner)]);
        std::fprintf(stderr, "trace columns: phase, monotonic_ns, wall_s, cpu_s, minor_faults, major_faults\n");
        SHOPPING_TIME(timing::calibrate();)
        mark("start");
    }

    bool enabled() const { return enabled_; }

    void mark(const char* phase) {
        if (!enabled_) return;
        timespec now{};
        clock_gettime(CLOCK_MONOTONIC, &now);
        int64_t time = int64_t(now.tv_sec) * 1'000'000'000 + now.tv_nsec;
        rusage usage{};
        getrusage(RUSAGE_SELF, &usage);
        int64_t cpu = micros(usage.ru_utime) + micros(usage.ru_stime)
            - micros(usage_.ru_utime) - micros(usage_.ru_stime);
        std::fprintf(stderr, "phase %s %lld %.6f %.6f %ld %ld\n", phase,
            static_cast<long long>(time), time_ ? double(time - time_) / 1e9 : 0,
            double(cpu) / 1e6, usage.ru_minflt - usage_.ru_minflt, usage.ru_majflt - usage_.ru_majflt);
        time_ = time;
        usage_ = usage;
        SHOPPING_COUNT(hardware_.mark(phase);)
    }

};

struct GroupTrace {
    int64_t cpu_ns = 0;
    size_t bucket_partitions = 0, sorted_partitions = 0, bucket_rows = 0;

    static int64_t cpu() {
        timespec now{};
        clock_gettime(CLOCK_THREAD_CPUTIME_ID, &now);
        return int64_t(now.tv_sec) * 1'000'000'000 + now.tv_nsec;
    }

    struct Scope {
        GroupTrace* trace;
        int64_t start = trace ? cpu() : 0;
        ~Scope() { if (trace) trace->cpu_ns += cpu() - start; }
    };
};

struct LinkTrace {
    int64_t begin = 0, end = 0, scan_ns = 0, init_ns = 0, insert_ns = 0, union_ns = 0, canon_ns = 0;

    static int64_t now() {
        timespec value{};
        clock_gettime(CLOCK_MONOTONIC, &value);
        return int64_t(value.tv_sec) * 1'000'000'000 + value.tv_nsec;
    }

    void report(int64_t ingest_start, int64_t events_end) const {
        auto ms = [](int64_t ns) { return double(ns) / 1e6; };
        std::fprintf(stderr,
            "links wall_ms %.3f done_ms %.3f last_event_chunk_ms %.3f scan_ms %.3f init_ms %.3f "
            "insert_ms %.3f union_ms %.3f canonicalise_ms %.3f\n",
            ms(end - begin), ms(end - ingest_start), ms(events_end - ingest_start),
            ms(scan_ns), ms(init_ns), ms(insert_ns), ms(union_ns), ms(canon_ns));
    }
};

template<size_t N, unsigned Shift, class Row> class Delimited {
    Row& row_;
    const char* line_;
    const char* field_;
    Fields<N> fields_;
    size_t column_ = 0;
    unsigned quotes_ = 0;
    bool malformed_ = false;

    void field(const char* end) {
        if (column_ == N) {
            malformed_ = true;
            return;
        }
        const char* start = field_;
        if (quotes_) {
            if (quotes_ != 2 || end - start < 2 || *start != '"' || end[-1] != '"') {
                malformed_ = true;
            } else {
                ++start;
                --end;
            }
        }
        fields_[column_++] = View(start, end - start);
    }

public:
    Delimited(const char* start, Row& row) : row_(row), line_(start), field_(start) {}

#if defined(__x86_64__)
    __attribute__((target("popcnt"), always_inline))
#endif
    void block(const char* data, uint64_t ends, uint64_t quotes, uint64_t newlines = 0) {
        while (ends) {
            unsigned bit = std::countr_zero(ends);
            const char* separator = data + (bit >> Shift);
            uint64_t before = quotes & ((uint64_t{1} << bit) - 1);
            if (before) quotes_ = std::min(3u, quotes_ + unsigned(__builtin_popcountll(before)));
            quotes ^= before;
            field(separator);
            field_ = separator + 1;
            quotes_ = 0;
            const bool newline = Shift ? *separator == '\n' : ((newlines >> bit) & 1);
            if (newline) {
                if (!malformed_ && column_ == N) row_(View(line_, separator - line_), fields_);
                line_ = field_;
                column_ = 0;
                malformed_ = false;
            }
            ends &= ends - 1;
        }
        if (quotes) quotes_ = std::min(3u, quotes_ + unsigned(__builtin_popcountll(quotes)));
    }

    void finish(const char* end) {
        if (line_ < end) scan_scalar<N>(View(line_, end - line_), row_);
    }
};

#if defined(__x86_64__)
struct Markers {
    uint64_t commas, newlines, quotes;
};

__attribute__((target("avx2"), always_inline)) inline Markers avx2_markers(const char* p) {
    auto data = _mm256_loadu_si256(reinterpret_cast<const __m256i*>(p));
    return {uint32_t(_mm256_movemask_epi8(_mm256_cmpeq_epi8(data, _mm256_set1_epi8(',')))),
            uint32_t(_mm256_movemask_epi8(_mm256_cmpeq_epi8(data, _mm256_set1_epi8('\n')))),
            uint32_t(_mm256_movemask_epi8(_mm256_cmpeq_epi8(data, _mm256_set1_epi8('"'))))};
}

__attribute__((target("avx512bw"), always_inline)) inline Markers avx512_markers(const char* p) {
    auto data = _mm512_loadu_si512(p);
    return {_mm512_cmpeq_epi8_mask(data, _mm512_set1_epi8(',')),
            _mm512_cmpeq_epi8_mask(data, _mm512_set1_epi8('\n')),
            _mm512_cmpeq_epi8_mask(data, _mm512_set1_epi8('"'))};
}

template<size_t N, class Row> __attribute__((target("avx2,popcnt")))
void scan_avx2(View chunk, Row row) {
    const char* p = chunk.data();
    const char* end = p + chunk.size();
    Delimited<N, 0, Row> parser(p, row);
    size_t prefix = -reinterpret_cast<uintptr_t>(p) & 31;
    if (prefix && end - p >= 32) {
        const auto [commas, newlines, quotes] = avx2_markers(p);
        uint64_t keep = (uint64_t{1} << prefix) - 1;
        parser.block(p, (commas | newlines) & keep, quotes & keep, newlines);
        p += prefix;
    }
    for (; end - p >= 32; p += 32) {
        const auto [commas, newlines, quotes] = avx2_markers(p);
        parser.block(p, commas | newlines, quotes, newlines);
    }
    parser.finish(end);
}

template<size_t N, class Row> __attribute__((target("avx512bw,popcnt")))
void scan_avx512(View chunk, Row row) {
    const char* p = chunk.data();
    const char* end = p + chunk.size();
    Delimited<N, 0, Row> parser(p, row);
    size_t prefix = -reinterpret_cast<uintptr_t>(p) & 63;
    if (prefix && end - p >= 64) {
        const auto [commas, newlines, quotes] = avx512_markers(p);
        uint64_t keep = (uint64_t{1} << prefix) - 1;
        parser.block(p, (commas | newlines) & keep, quotes & keep, newlines);
        p += prefix;
    }
    for (; end - p >= 64; p += 64) {
        const auto [commas, newlines, quotes] = avx512_markers(p);
        parser.block(p, commas | newlines, quotes, newlines);
    }
    parser.finish(end);
}
#elif defined(__aarch64__)
uint64_t neon_mask(uint8x16_t compared) {
    auto nibbles = vshrn_n_u16(vreinterpretq_u16_u8(compared), 4);
    return vget_lane_u64(vreinterpret_u64_u8(nibbles), 0) & 0x1111111111111111ULL;
}

template<size_t N, class Row> void scan_neon(View chunk, Row row) {
    const char* p = chunk.data();
    const char* end = p + chunk.size();
    Delimited<N, 2, Row> parser(p, row);
    for (; end - p >= 16; p += 16) {
        auto data = vld1q_u8(reinterpret_cast<const uint8_t*>(p));
        auto comma = vceqq_u8(data, vdupq_n_u8(','));
        auto newline = vceqq_u8(data, vdupq_n_u8('\n'));
        auto quotes = vceqq_u8(data, vdupq_n_u8('"'));
        parser.block(p, neon_mask(vorrq_u8(comma, newline)), neon_mask(quotes));
    }
    parser.finish(end);
}
#endif

#include "csv_index.hpp"

template<size_t N, class Row> void scan_fields(View chunk, Row row) {
#if defined(__x86_64__)
    if (scanner == Scanner::Avx512) return scan_indexed<N, csv_avx512>(chunk, row);
    if (scanner == Scanner::Avx2) return scan_indexed<N, csv_avx2>(chunk, row);
#elif defined(__aarch64__)
    if (scanner == Scanner::Neon) return scan_indexed<N, csv_neon>(chunk, row);
#endif
    scan_scalar<N>(chunk, [&](View, const Fields<N>& fields) { row(ScalarCsvRow<N>{fields}); });
}

template<size_t N, class Row> void scan(View chunk, Row row) {
#if defined(__x86_64__)
    if (scanner == Scanner::Avx512) return scan_avx512<N>(chunk, row);
    if (scanner == Scanner::Avx2) return scan_avx2<N>(chunk, row);
#elif defined(__aarch64__)
    if (scanner == Scanner::Neon) return scan_neon<N>(chunk, row);
#endif
    scan_scalar<N>(chunk, row);
}

int two_digits(const char* text) {
    unsigned word = uint8_t(text[0]) | unsigned(uint8_t(text[1])) << 8;
    if ((word & 0xf0f0) != 0x3030 || ((word + 0x0606) & 0xf0f0) != 0x3030) return -1;
    return int((word & 15) * 10 + ((word >> 8) & 15));
}


int64_t decimal(View field, int64_t limit) {
    if (field.empty()) return bad_time;
    int64_t value = 0;
    if (field.size() <= 18) {
        for (char c : field) {
            unsigned digit = unsigned(c - '0');
            if (digit > 9) return bad_time;
            value = value * 10 + digit;
        }
        return value <= limit ? value : bad_time;
    }
    for (char c : field) {
        unsigned digit = unsigned(c - '0');
        if (digit > 9 || value > limit / 10 ||
            (value == limit / 10 && digit > uint64_t(limit % 10))) return bad_time;
        value = value * 10 + digit;
    }
    return value;
}

struct Year { int32_t days; bool leap; };
constexpr auto years = [] {
    std::array<Year, 10000> table{};
    for (int y = 1; y <= 9999; ++y) {
        int prior = y - 1;
        table[y] = {prior * 365 + prior / 4 - prior / 100 + prior / 400 - 719162,
                    y % 4 == 0 && (y % 100 != 0 || y % 400 == 0)};
    }
    return table;
}();
constexpr int month_lengths[] = {31,28,31,30,31,30,31,31,30,31,30,31};
constexpr int month_offsets[] = {0,31,59,90,120,151,181,212,243,273,304,334};

int64_t timestamp(View s) {
    if ((s.size() != 20 && s.size() != 25) || s[4] != '-')
        return decimal(s, INT64_MAX);
    if (s[7] != '-' || s[10] != 'T' || s[13] != ':' || s[16] != ':') return bad_time;
    const char* p = s.data();
    int century = two_digits(p), yr = two_digits(p + 2);
    int m = two_digits(p + 5), d = two_digits(p + 8);
    int h = two_digits(p + 11), minute = two_digits(p + 14), second = two_digits(p + 17);
    int y = century * 100 + yr;
    if ((century | yr) < 0 || y < 1 || m < 1 || m > 12 || d < 1 || unsigned(h) > 23 ||
        unsigned(minute) > 59 || unsigned(second) > 59) return bad_time;
    auto [year_days, leap] = years[y];
    if (d > month_lengths[m - 1] + (m == 2 && leap)) return bad_time;
    int days = year_days + month_offsets[m - 1] + d - 1 + (m > 2 && leap);
    int offset = 0;
    if (s.size() == 20) {
        if (s[19] != 'Z') return bad_time;
    } else {
        if ((s[19] != '-' && s[19] != '+') || s[22] != ':') return bad_time;
        int oh = two_digits(p + 20), om = two_digits(p + 23);
        if (unsigned(oh) > 23 || unsigned(om) > 59) return bad_time;
        offset = (oh * 60 + om) * (s[19] == '+' ? 1 : -1);
    }
    return int64_t(days) * day_ms + ((h * 60 + minute - offset) * 60 + second) * 1000;
}

int64_t epoch_millis_13(const char* p) {
    uint64_t first;
    uint32_t next;
    std::memcpy(&first, p, sizeof(first));
    std::memcpy(&next, p + sizeof(first), sizeof(next));
    unsigned last = uint8_t(p[12]) - unsigned('0');
    first ^= 0x3030303030303030ULL;
    next ^= 0x30303030U;
    if (((first | (first + 0x0606060606060606ULL) | next | (next + 0x06060606U)) &
         0xf0f0f0f0f0f0f0f0ULL) || last > 9) return bad_time;
    auto four = [](uint64_t digits) {
        uint64_t pairs = (digits * 10 + (digits >> 8)) & 0x00ff00ff00ff00ffULL;
        return (pairs * 100 + (pairs >> 16)) & 0x0000ffff0000ffffULL;
    };
    uint64_t groups = four(first);
    uint64_t first_eight = uint32_t(groups) * 10000 + (groups >> 32);
    return first_eight * 100000 + four(next) * 10 + last;
}

int clock_seconds(const char* p) {
    uint64_t word;
    std::memcpy(&word, p, sizeof(word));
    word ^= 0x30303a30303a3030ULL;
    if ((word | (word + 0x0606000606000606ULL)) & 0xf0f0fff0f0fff0f0ULL) return -1;
    uint64_t pairs = word * 10 + (word >> 8);
    unsigned h = uint8_t(pairs), m = uint8_t(pairs >> 24), s = uint8_t(pairs >> 48);
    if ((h > 23) | (m > 59) | (s > 59)) return -1;
    return h * 3600 + m * 60 + s;
}

int ingest_pair(const char* p) { return (p[0] - '0') * 10 + (p[1] - '0'); }

int ingest_seconds(const char* p) {
    uint64_t digits;
    std::memcpy(&digits, p, sizeof(digits));
    digits = (digits ^ 0x3030003030003030ULL) & 0xffff00ffff00ffffULL;
    if ((digits | (digits + 0x0606000606000606ULL)) & 0xf0f000f0f000f0f0ULL)
        return (ingest_pair(p) * 60 + ingest_pair(p + 3)) * 60 + ingest_pair(p + 6);
    uint64_t pairs = digits * 10 + (digits >> 8);
    return uint8_t(pairs) * 3600 + uint8_t(pairs >> 24) * 60 + uint8_t(pairs >> 48);
}

int64_t ingest_timestamp(const char* p) {
    int y = ingest_pair(p) * 100 + ingest_pair(p + 2), m = ingest_pair(p + 5), d = ingest_pair(p + 8);
    if (unsigned(y) >= years.size() || unsigned(m - 1) >= 12) return bad_time;
    auto [year_days, leap] = years[y];
    int days = year_days + month_offsets[m - 1] + d - 1 + (m > 2 && leap);
    return int64_t(days) * day_ms + int64_t(ingest_seconds(p + 11)) * 1000;
}

int32_t utc_day(int64_t t) { return t / day_ms - (t % day_ms < 0); }

class ManifestJson {
    View rest_;

    [[noreturn]] static void fail() { throw std::runtime_error("invalid manifest JSON"); }

    char peek() const { return rest_.empty() ? '\0' : rest_.front(); }
    char next() {
        if (rest_.empty()) fail();
        char c = rest_.front();
        rest_.remove_prefix(1);
        return c;
    }
    void whitespace() {
        while (peek() == ' ' || peek() == '\n' || peek() == '\r' || peek() == '\t') next();
    }
    bool take(char c) {
        if (peek() != c) return false;
        next();
        return true;
    }
    void require(char c) { if (!take(c)) fail(); }
    void literal(View word) {
        if (!rest_.starts_with(word)) fail();
        rest_.remove_prefix(word.size());
    }
    static bool digit(char c) { return unsigned(c - '0') < 10; }
    void digits() {
        if (!digit(peek())) fail();
        do { next(); } while (digit(peek()));
    }
    void number() {
        take('-');
        if (!take('0')) digits();
        if (take('.')) digits();
        if (take('e') || take('E')) {
            if (!take('+')) take('-');
            digits();
        }
    }
    unsigned hex4() {
        unsigned result = 0;
        for (int i = 0; i < 4; ++i) {
            char c = next();
            if (!digit(c) && unsigned((c | 32) - 'a') >= 6) fail();
            result = (result << 4) + (digit(c) ? c - '0' : (c | 32) - 'a' + 10);
        }
        return result;
    }
    static void utf8(std::string& value, unsigned cp) {
        if (cp <= 0x7f) {
            value += char(cp);
        } else if (cp <= 0x7ff) {
            value += char(0xc0 | (cp >> 6));
            value += char(0x80 | (cp & 63));
        } else if (cp <= 0xffff) {
            value += char(0xe0 | (cp >> 12));
            value += char(0x80 | ((cp >> 6) & 63));
            value += char(0x80 | (cp & 63));
        } else {
            value += char(0xf0 | (cp >> 18));
            value += char(0x80 | ((cp >> 12) & 63));
            value += char(0x80 | ((cp >> 6) & 63));
            value += char(0x80 | (cp & 63));
        }
    }
    std::string string() {
        require('"');
        std::string value;
        for (char c; (c = next()) != '"';) {
            if (uint8_t(c) < 32) fail();
            if (c != '\\') {
                value += c;
                continue;
            }
            switch (c = next()) {
                case '"': case '\\': case '/': value += c; break;
                case 'b': value += '\b'; break;
                case 'f': value += '\f'; break;
                case 'n': value += '\n'; break;
                case 'r': value += '\r'; break;
                case 't': value += '\t'; break;
                case 'u': {
                    unsigned cp = hex4();
                    if (cp >= 0xd800 && cp <= 0xdbff && rest_.starts_with("\\u")) {
                        View saved = rest_;
                        rest_.remove_prefix(2);
                        unsigned low = hex4();
                        if (low >= 0xdc00 && low <= 0xdfff)
                            cp = 0x10000 + ((cp - 0xd800) << 10) + low - 0xdc00;
                        else
                            rest_ = saved;
                    }
                    utf8(value, cp);
                    break;
                }
                default: fail();
            }
        }
        return value;
    }
    template<class Item> void items(char end, Item item) {
        whitespace();
        if (take(end)) return;
        do {
            whitespace();
            item();
            whitespace();
        } while (take(','));
        require(end);
    }
    void colon() {
        whitespace();
        require(':');
        whitespace();
    }
    void skip() {
        switch (peek()) {
            case '{':
                next();
                items('}', [&] { string(); colon(); skip(); });
                break;
            case '[': next(); items(']', [&] { skip(); }); break;
            case '"': string(); break;
            case 't': literal("true"); break;
            case 'f': literal("false"); break;
            case 'n': literal("null"); break;
            default: number(); break;
        }
    }

public:
    explicit ManifestJson(View bytes) : rest_(bytes) {
        if (rest_.starts_with("\xef\xbb\xbf")) rest_.remove_prefix(3);
    }

    std::array<std::string, 2> window() {
        std::array<std::string, 2> fields;
        whitespace();
        require('{');
        items('}', [&] {
            std::string name = string();
            colon();
            if (name == "window_start" || name == "window_end") {
                auto& field = fields[name == "window_end"];
                if (peek() == '"') {
                    field = string();
                    return;
                }
                field.clear();
            }
            skip();
        });
        whitespace();
        if (!rest_.empty()) fail();
        return fields;
    }
};

struct Window {
private:
    struct Date { uint64_t prefix; uint32_t suffix; int32_t day; };
    std::array<Date, 64> dates_{};

    static size_t date_slot(const char* s) {
        return ((uint8_t(s[6]) & 1) << 5) | ((uint8_t(s[8]) * 10 + uint8_t(s[9])) & 31);
    }

    static Date date_key(const char* s) {
        Date result{};
        std::memcpy(&result.prefix, s, 8);
        std::memcpy(&result.suffix, s + 8, 3);
        return result;
    }

    const Date* cached_date(const char* s) const {
        auto key = date_key(s);
        const auto& entry = dates_[date_slot(s)];
        return entry.prefix && ((entry.prefix ^ key.prefix) | (entry.suffix ^ key.suffix)) == 0
            ? &entry : nullptr;
    }

public:
    int64_t start, end;
    explicit Window(const fs::path& path) {
        Mapping file(path);
        auto fields = ManifestJson(file.bytes()).window();
        start = timestamp(fields[0]);
        end = timestamp(fields[1]);
        if (start == bad_time || end == bad_time || start >= end)
            throw std::runtime_error("invalid manifest window");
        if (start < int64_t(years[1].days + 9) * day_ms ||
            end > int64_t(years[9999].days + 350) * day_ms ||
            uint64_t(end) - uint64_t(start) > 32 * day_ms) return;
        using namespace std::chrono;
        for (int32_t day = utc_day(start) - 8; day <= utc_day(end - 1) + 3; ++day) {
            year_month_day date{sys_days{days{day}}};
            char key[12];
            std::snprintf(key, sizeof(key), "%04d-%02u-%02uT", int(date.year()),
                          unsigned(date.month()), unsigned(date.day()));
            auto& entry = dates_[date_slot(key)];
            entry = date_key(key);
            entry.day = day;
        }
    }

    int64_t event_time(View text) const {
        if (text.size() == 13) return epoch_millis_13(text.data());
        if (text.size() == 20 && text[19] == 'Z') {
            if (auto* date = cached_date(text.data())) {
                int seconds = clock_seconds(text.data() + 11);
                return seconds < 0 ? bad_time : int64_t(date->day) * day_ms + seconds * 1000;
            }
        }
        return timestamp(text);
    }

    int64_t ingest_time(View text) const {
        if (text.size() != 20) return bad_time;
        if (auto* date = cached_date(text.data()))
            return int64_t(date->day) * day_ms + int64_t(ingest_seconds(text.data() + 11)) * 1000;
        return ingest_timestamp(text.data());
    }
};

using Person = uint64_t;

class Ids {
    static constexpr uint64_t external = uint64_t{1} << 63;
    static constexpr uint64_t prefixed = uint64_t{1} << 55;
    static constexpr uint64_t multiplier = 0x94d049bb133111ebULL;
    static constexpr uint64_t inverse = 0x319642b2d24d8ec3ULL;
    static_assert(multiplier * inverse == 1);
    struct Shard {
        mutable std::mutex lock;
        StringMap<uint32_t> indices;
        memory::Vector<View> names;
        size_t longest = 0;
    };
    std::array<Shard, 64> shards_;
    char prefix_;

    static uint64_t decode(Person key) { return (key ^ (key >> 32)) * inverse; }

    uint64_t inline_id(View bytes) const {
        if (bytes.size() == 9 && bytes[0] == prefix_) {
            uint64_t suffix;
            std::memcpy(&suffix, bytes.data() + 1, sizeof(suffix));
            constexpr uint64_t high_bits = 0x8080808080808080ULL;
            if (!((suffix | (suffix - 0x0101010101010101ULL)) & high_bits))
                return suffix | external | prefixed;
        }
        if (bytes.size() == 8 || bytes.size() == 9) {
            uint64_t word;
            std::memcpy(&word, bytes.data(), 8);
            constexpr uint64_t high = 0x8080808080808080ULL;
            if ((word & high) || ((word - 0x0101010101010101ULL) & ~word & high)) return 0;
            uint64_t ninth = bytes.size() == 9 ? uint8_t(bytes[8]) : 0;
            if (bytes.size() == 9 && (ninth == 0 || ninth > 127)) return 0;
#if defined(__BMI2__)
            return _pext_u64(word, 0x7f7f7f7f7f7f7f7fULL) | (ninth << 56);
#else
            word = (word & 0x007f007f007f007fULL) | ((word >> 1) & 0x3f803f803f803f80ULL);
            word = (word & 0x00003fff00003fffULL) | ((word >> 2) & 0x0fffc0000fffc000ULL);
            return (word & 0x0fffffffULL) | ((word >> 4) & 0x00fffffff0000000ULL) | (ninth << 56);
#endif
        }
        if (bytes.size() > 7) return 0;
        uint64_t value = 0;
        unsigned shift = 0;
        for (uint8_t c : bytes) {
            if (!c || c > 127) return 0;
            value |= uint64_t(c) << shift;
            shift += 7;
        }
        return value;
    }

public:
    explicit Ids(char prefix) : prefix_(prefix) {}

    Person key(View bytes) {
        uint64_t value = inline_id(bytes);
        if (!value) {
            size_t index = Hash{}(bytes) % shards_.size();
            auto& shard = shards_[index];
            std::lock_guard lock(shard.lock);
            auto position = shard.indices.find(bytes);
            if (position == shard.indices.end()) {
                if (shard.names.size() > UINT32_MAX) throw std::runtime_error("too many distinct IDs");
                auto number = static_cast<uint32_t>(shard.names.size());
                shard.names.push_back(bytes);
                position = shard.indices.emplace(bytes, number).first;
                shard.longest = std::max(shard.longest, bytes.size());
            }
            value = external | (uint64_t(index) << 32) | position->second;
        }
        value *= multiplier;
        return value ^ (value >> 32);
    }

    size_t longest() const {
        size_t length = 9;
        for (const auto& shard : shards_) length = std::max(length, shard.longest);
        return length;
    }

    bool has_input_views() const {
        return std::any_of(shards_.begin(), shards_.end(), [](const Shard& shard) { return !shard.names.empty(); });
    }
private:
    template<bool Concurrent> View unpack_id(uint64_t value, char (&buffer)[9]) const {
        if (value & external) {
            if (!(value & prefixed)) {
                const auto& shard = shards_[(value >> 32) & 63];
                if constexpr (Concurrent) {
                    std::lock_guard lock(shard.lock);
                    return shard.names[uint32_t(value)];
                } else {
                    return shard.names[uint32_t(value)];
                }
            }
            uint64_t suffix = value ^ (external | prefixed);
            buffer[0] = prefix_;
            std::memcpy(buffer + 1, &suffix, sizeof(suffix));
            return View(buffer, sizeof(buffer));
        }
#if defined(__BMI2__)
        uint64_t word = _pdep_u64(value, 0x7f7f7f7f7f7f7f7fULL);
#else
        uint64_t word = (value & 0x0fffffffULL) | ((value & 0x00fffffff0000000ULL) << 4);
        word = (word & 0x00003fff00003fffULL) | ((word & 0x0fffc0000fffc000ULL) << 2);
        word = (word & 0x007f007f007f007fULL) | ((word & 0x3f803f803f803f80ULL) << 1);
#endif
        std::memcpy(buffer, &word, 8);
        buffer[8] = value >> 56;
        return View(buffer, buffer[8] ? 9 : (std::bit_width(word) + 7) / 8);
    }

public:
    View name(Person key, char (&buffer)[9]) const { return unpack_id<false>(decode(key), buffer); }

    bool less(Person first, Person second) const {
        uint64_t a = decode(first), b = decode(second);
        constexpr uint64_t mark = external | prefixed;
        if ((a & b & mark) == mark) return __builtin_bswap64(a ^ mark) < __builtin_bswap64(b ^ mark);
        char left[9], right[9];
        return unpack_id<true>(a, left) < unpack_id<true>(b, right);
    }
};

constexpr size_t max_alias_batch = 64;

class People {
    struct Alias { Person key, value; };
    static_assert(sizeof(Alias) == 16);
    static constexpr uint64_t root_flag = uint64_t{1} << 63;
    static constexpr uint64_t resolved_flag = uint64_t{1} << 62;
    static constexpr size_t block = (2 << 20) / sizeof(Alias);
    std::optional<memory::HugeArray<Alias>> aliases_;
    std::vector<uint64_t> has_alias_;
    size_t mask_ = 0;
    unsigned shift_ = 0;

    size_t alias_bit(Person key) const {
        return (key >> 12) & (has_alias_.size() * 64 - 1);
    }

    size_t slot(Person key) const {
        const auto* data = aliases_->data();
        size_t i = key & mask_;
        while (data[i].key && data[i].key != key) i = (i + 1) & mask_;
        return i;
    }

    template<bool Concurrent> size_t insert(Person key) {
        auto* data = aliases_->data();
        if constexpr (!Concurrent) {
            size_t i = slot(key);
            if (!data[i].key) data[i] = {key, root_flag | (uint64_t{1} << shift_) | i};
            return i;
        } else {
            for (size_t i = key & mask_;; i = (i + 1) & mask_) {
                std::atomic_ref current(data[i].key);
                Person found = current.load(std::memory_order_relaxed);
                if (!found && current.compare_exchange_strong(found, key, std::memory_order_relaxed)) {
                    data[i].value = root_flag | (uint64_t{1} << shift_) | i;
                    return i;
                }
                if (found == key) return i;
            }
        }
    }

    size_t root(size_t i) {
        auto* data = aliases_->data();
        uint64_t parent = data[i].value;
        while (!(parent & root_flag)) {
            uint64_t next = data[parent].value;
            if (!(next & root_flag)) data[i].value = next;
            i = parent;
            parent = next;
        }
        return i;
    }

    void join(size_t a, size_t b, const Ids& ids) {
        a = root(a);
        b = root(b);
        if (a == b) return;
        auto* data = aliases_->data();
        uint64_t left = data[a].value, right = data[b].value;
        auto left_min = left & mask_, right_min = right & mask_;
        size_t smallest = ids.less(data[right_min].key, data[left_min].key) ? right_min : left_min;
        uint64_t left_size = (left ^ root_flag) >> shift_, right_size = (right ^ root_flag) >> shift_;
        if (left_size < right_size) std::swap(a, b);
        uint64_t size = std::min(left_size + right_size, (root_flag - 1) >> shift_);
        data[b].value = a;
        data[a].value = root_flag | (size << shift_) | smallest;
    }

public:
    using Pairs = memory::Vector<std::array<Person, 2>>;

    template<bool Concurrent> void insert_rows(Pairs& pairs, size_t depth) {
        auto* data = aliases_->data();
        for (size_t start = 0; start < pairs.size(); start += depth) {
            size_t count = std::min(depth, pairs.size() - start);
            for (size_t i = 0; i < count; ++i) {
                auto pair = pairs[start + i];
                __builtin_prefetch(data + (pair[0] & mask_), 1, 1);
                __builtin_prefetch(data + (pair[1] & mask_), 1, 1);
            }
            for (size_t i = 0; i < count; ++i) {
                auto& pair = pairs[start + i];
                pair = {insert<Concurrent>(pair[0]), insert<Concurrent>(pair[1])};
            }
        }
    }

    void join_rows(const Pairs& pairs, const Ids& ids, size_t depth) {
        auto* data = aliases_->data();
        for (size_t start = 0; start < pairs.size(); start += depth) {
            size_t count = std::min(depth, pairs.size() - start);
            for (size_t i = 0; i < count; ++i) {
                auto pair = pairs[start + i];
                __builtin_prefetch(data + pair[0], 0, 1);
                __builtin_prefetch(data + pair[1], 0, 1);
            }
            for (size_t i = 0; i < count; ++i) {
                auto pair = pairs[start + i];
                __builtin_prefetch(data + (data[pair[0]].value & mask_), 0, 1);
                __builtin_prefetch(data + (data[pair[1]].value & mask_), 0, 1);
            }
            for (size_t i = 0; i < count; ++i) join(pairs[start + i][0], pairs[start + i][1], ids);
        }
    }

    void allocate(const Input& links, size_t rows, bool huge, int load, bool use_bitmap) {
        size_t bound = std::min(links.bytes / 2 + links.chunks.size(), 2 * rows);
        if (bound > (size_t{1} << (std::numeric_limits<size_t>::digits - 6))) throw std::bad_alloc();
        size_t requested = load == 50 ? bound * 2 : (bound * 4 + 2) / 3;
        size_t slots = std::bit_ceil(std::max<size_t>(32, requested));
        aliases_.emplace(slots, huge);
        mask_ = slots - 1;
        shift_ = std::countr_zero(slots);
        if (use_bitmap) has_alias_.resize(std::min<size_t>(slots / 8, 16'384));
    }

    size_t blocks() const { return (aliases_->size() + block - 1) / block; }

    void touch(size_t task) {
        size_t start = task * block;
        std::memset(aliases_->data() + start, 0, std::min(block, aliases_->size() - start) * sizeof(Alias));
    }

    void resolve(size_t task) {
        auto* data = aliases_->data();
        size_t start = task * block, end = std::min(start + block, aliases_->size());
        for (size_t i = start; i < end; ++i) {
            if (!data[i].key) continue;
            std::atomic_ref current(data[i].value);
            uint64_t value = current.load(std::memory_order_relaxed);
            if (value & root_flag) continue;
            while (!(value & (root_flag | resolved_flag)))
                value = std::atomic_ref(data[value].value).load(std::memory_order_relaxed);
            current.store(resolved_flag | (value & mask_), std::memory_order_relaxed);
        }
    }

    void publish(size_t task) {
        auto* data = aliases_->data();
        size_t start = task * block, end = std::min(start + block, aliases_->size());
        for (size_t i = start; i < end; ++i) {
            auto& alias = data[i];
            if (!alias.key) continue;
            alias.value = data[alias.value & mask_].key;
            if (bitmap() && alias.key != alias.value) {
                size_t bit = alias_bit(alias.key);
                std::atomic_ref(has_alias_[bit / 64]).fetch_or(uint64_t{1} << (bit % 64), std::memory_order_relaxed);
            }
        }
    }

    bool bitmap() const { return !has_alias_.empty(); }

    bool might_have_alias(Person key) const {
        size_t bit = alias_bit(key);
        return has_alias_[bit / 64] & (uint64_t{1} << (bit % 64));
    }

    void prefetch_membership(Person key) const { __builtin_prefetch(&has_alias_[alias_bit(key) / 64], 0, 1); }

    void prefetch_alias(Person key) const { __builtin_prefetch(aliases_->data() + (key & mask_), 0, 1); }

    Person lookup(Person key) const {
        auto value = (*aliases_)[slot(key)].value;
        return value ? value : key;
    }

    Person name(Person key) const { return !bitmap() || might_have_alias(key) ? lookup(key) : key; }
};

class LinkBuild {
public:
    enum Stage : unsigned { Scan, Touch, Insert, Join, Resolve, Publish, Done, Cancelled };

private:
    People& people_;
    const Input& input_;
    Ids& ids_;
    const size_t depth_;
    const int load_;
    const bool bitmap_, huge_, concurrent_;
    LinkTrace* trace_;
    std::vector<People::Pairs> pairs_;
    std::atomic<Stage> stage_{Scan};
    std::array<std::atomic<size_t>, Done> next_{};
    std::atomic<size_t> remaining_;
    int64_t scanned_ = 0, touched_ = 0, inserted_ = 0, joined_ = 0;

    size_t tasks(Stage stage) const {
        if (stage == Scan || stage == Insert) return std::max<size_t>(1, pairs_.size());
        if (stage == Join) return 1;
        return people_.blocks();
    }

    void scan(size_t task) {
        if (pairs_.empty()) return;
        View chunk = input_.chunks[task];
        auto& rows = pairs_[task];
        rows.reserve(chunk.size() / 18);
        scan_fields<2>(chunk, [&](const auto& fields) {
            const char *a, *b;
            size_t first = fields.template field<0>(a), second = fields.template field<1>(b);
            if (first && second) rows.push_back({ids_.key(View(a, first)), ids_.key(View(b, second))});
        });
    }

    void advance(Stage current) {
        switch (current) {
        case Scan: {
            if (trace_) {
                scanned_ = LinkTrace::now();
                trace_->scan_ns = scanned_ - trace_->begin;
            }
            size_t count = 0;
            for (const auto& part : pairs_) count += part.size();
            people_.allocate(input_, count, huge_, load_, bitmap_);
            break;
        }
        case Touch:
            if (trace_) {
                touched_ = LinkTrace::now();
                trace_->init_ns = touched_ - scanned_;
            }
            break;
        case Insert:
            if (trace_) {
                inserted_ = LinkTrace::now();
                trace_->insert_ns = inserted_ - touched_;
            }
            break;
        case Join:
            if (trace_) {
                joined_ = LinkTrace::now();
                trace_->union_ns = joined_ - inserted_;
            }
            break;
        case Publish:
            if (trace_) {
                trace_->end = LinkTrace::now();
                trace_->canon_ns = trace_->end - joined_;
            }
            break;
        default: break;
        }
        auto next = static_cast<Stage>(current + 1);
        if (next != Done) remaining_.store(tasks(next), std::memory_order_relaxed);
        if (stage_.compare_exchange_strong(current, next, std::memory_order_release, std::memory_order_relaxed))
            stage_.notify_all();
    }

public:
    LinkBuild(People& people, const Input& input, Ids& ids, size_t depth, int load, bool bitmap, bool huge,
              unsigned threads, LinkTrace* trace)
        : people_(people), input_(input), ids_(ids), depth_(std::max<size_t>(1, depth)), load_(load),
          bitmap_(bitmap), huge_(huge), concurrent_(threads > 1), trace_(trace), pairs_(input.chunks.size()),
          remaining_(tasks(Scan)) {}

    Stage stage() const { return stage_.load(std::memory_order_acquire); }
    void wait(Stage stage) const { stage_.wait(stage, std::memory_order_acquire); }
    void cancel() {
        stage_.store(Cancelled, std::memory_order_release);
        stage_.notify_all();
    }

    bool work(Stage stage) {
        if (stage >= Done) return false;
        size_t task = next_[stage].fetch_add(1, std::memory_order_relaxed);
        if (task >= tasks(stage)) return false;
        switch (stage) {
        case Scan: scan(task); break;
        case Touch: people_.touch(task); break;
        case Insert:
            if (pairs_.empty()) break;
            if (concurrent_) people_.insert_rows<true>(pairs_[task], depth_);
            else people_.insert_rows<false>(pairs_[task], depth_);
            break;
        case Join:
            for (const auto& part : pairs_) people_.join_rows(part, ids_, depth_);
            break;
        case Resolve: people_.resolve(task); break;
        case Publish: people_.publish(task); break;
        default: break;
        }
        if (remaining_.fetch_sub(1, std::memory_order_acq_rel) == 1) advance(stage);
        return true;
    }
};

template<class Row, class Emit> class PersonBatch {
    const People& people_;
    const size_t limit_;
    const bool bitmap_;
    Emit emit_;
    std::array<Row, max_alias_batch> rows_;
    size_t size_ = 0;

public:
    PersonBatch(const People& people, size_t limit, Emit emit)
        : people_(people), limit_(limit), bitmap_(people.bitmap()), emit_(emit) {}

    void append(Row row) {
        if (!limit_) {
            row.person = people_.name(row.person);
            emit_(row);
            return;
        }
        if (bitmap_) people_.prefetch_membership(row.person);
        else people_.prefetch_alias(row.person);
        rows_[size_++] = row;
        if (size_ == limit_) finish();
    }

    void finish() {
        std::array<bool, max_alias_batch> probe;
        if (bitmap_)
            for (size_t i = 0; i < size_; ++i) {
                auto person = rows_[i].person;
                probe[i] = people_.might_have_alias(person);
                if (probe[i]) people_.prefetch_alias(person);
            }
        for (size_t i = 0; i < size_; ++i) {
            auto& row = rows_[i];
            if (!bitmap_ || probe[i]) row.person = people_.lookup(row.person);
            emit_(row);
        }
        size_ = 0;
    }
};


struct Catalogs {
    StringMap<size_t> products;
    memory::Vector<size_t> numbered_products;
    StringMap<uint32_t> campaign_ids;
    memory::Vector<uint32_t> numbered_campaigns;
    memory::Vector<View> categories{"unknown"};
    memory::Vector<View> campaigns{"organic"}, channels{"organic"};

    static int product_number(View s) {
        if (s.size() < 2 || s.size() > 6 || s[0] != 'p' || (s.size() > 2 && s[1] == '0')) return -1;
        unsigned number = 0;
        for (char c : s.substr(1)) {
            unsigned digit = uint8_t(c) - unsigned('0');
            if (digit > 9) return -1;
            number = number * 10 + digit;
        }
        return number;
    }

    static int campaign_number(View s) {
        if (s.size() < 3 || s.size() > 6 || !s.starts_with("c_")) return -1;
        if (s.size() == 3) {
            unsigned digit = uint8_t(s[2]) - unsigned('0');
            return digit <= 9 ? static_cast<int>(digit) : -1;
        }
        if (s[2] == '0') return -1;
        uint32_t word;
        std::memcpy(&word, s.data() + s.size() - 4, 4);
        word = (word ^ 0x30303030U) & (UINT32_MAX << (8 * (6 - s.size())));
        if ((word | (word + 0x06060606U)) & 0xf0f0f0f0U) return -1;
        uint32_t pairs = (word * 10 + (word >> 8)) & 0x00ff00ffU;
        return (pairs & 255) * 100 + (pairs >> 16);
    }

    Catalogs(const Input& product_input, const Input& campaign_input) {
        StringMap<size_t> category_ids;
        for (View chunk : product_input.chunks) scan<2>(chunk, [&](View, const Fields<2>& f) {
            auto [where, inserted] = category_ids.try_emplace(f[1], categories.size());
            if (inserted) categories.push_back(f[1]);
            if (int number = product_number(f[0]); number >= 0) {
                if (size_t limit = number + 1; limit > numbered_products.size()) numbered_products.resize(limit);
                if (!numbered_products[number]) numbered_products[number] = where->second;
            } else {
                products.emplace(f[0], where->second);
            }
        });
        for (View chunk : campaign_input.chunks) scan<2>(chunk, [&](View, const Fields<2>& f) {
            campaign_ids.emplace(f[0], campaigns.size());
            if (int number = campaign_number(f[0]); number >= 0) {
                if (size_t limit = number + 1; limit > numbered_campaigns.size()) numbered_campaigns.resize(limit);
                numbered_campaigns[number] = campaigns.size();
            }
            campaigns.push_back(f[0]);
            channels.push_back(f[1]);
        });
    }
    size_t category(View product) const {
        if (int number = product_number(product); number >= 0)
            return size_t(number) < numbered_products.size() ? numbered_products[number] : 0;
        auto result = products.find(product);
        return result == products.end() ? 0 : result->second;
    }

    uint32_t campaign(View id) const {
        unsigned number = campaign_number(id);
        if (number < numbered_campaigns.size()) return numbered_campaigns[number];
        auto result = campaign_ids.find(id);
        return result == campaign_ids.end() ? 0 : result->second;
    }
};

template<class Category> struct EventValue { int64_t time; uint32_t price; Category category; };
template<class Value> struct PersonEvent { Person person; Value value; };
template<class Category> using Event = PersonEvent<EventValue<Category>>;
using CompactEvent = PersonEvent<uint64_t>;
template<class Time> struct ClickValue { const char* id; Time time; uint32_t campaign; };
template<class Time> using Click = PersonEvent<ClickValue<Time>>;

bool click_id_after(const char* a, const char* b) {
    for (;;) {
        uint8_t left = *a++, right = *b++;
        bool left_end = left == ',' || left == '"', right_end = right == ',' || right == '"';
        if (left_end || right_end) return right_end && !left_end;
        if (left != right) return left > right;
    }
}

constexpr uint32_t price_mask = (1u << 24) - 1;

bool compact_events(const Window& window, const Catalogs& catalogs) {
    return uint64_t(window.end) - uint64_t(window.start) <= UINT32_MAX - week_ms &&
           catalogs.categories.size() <= 256;
}

struct CompactCandidate {
    using Survivor = CompactEvent;
    using Category = uint32_t;
    uint64_t id;
    int64_t ingest;
    CompactEvent payload;

    CompactCandidate() = default;
    CompactCandidate(uint64_t id, int64_t ingest, Event<Category> event, const Window& window)
        : id(id), ingest(ingest), payload{event.person, event.person
            ? uint64_t(event.value.time - window.start) << 32 |
              uint64_t(event.value.category) << 24 | event.value.price
            : 0} {}
};

template<class Index> struct WideCandidate {
    using Category = Index;
    using Survivor = Event<Category>;
    uint64_t id;
    int64_t ingest;
    Survivor payload;

    WideCandidate() = default;
    WideCandidate(uint64_t id, int64_t ingest, Survivor event, const Window&)
        : id(id), ingest(ingest), payload(event) {}

};

#include "scatter.hpp"
#include "group_sort.hpp"

EventValue<uint32_t> unpack(uint64_t value, const Window& window) {
    return {window.start + int64_t(value >> 32), uint32_t(value) & price_mask, uint32_t(value) >> 24};
}

template<class Category> EventValue<Category> unpack(EventValue<Category> value, const Window&) {
    return value;
}

#include "session_buckets.hpp"

template<class Category, class Row>
Event<Category> parse_event(const Row& fields, const Window& window, const Catalogs& catalogs,
                            Ids& ids, int64_t ingest SHOPPING_TIME(, unsigned weight)) {
    SHOPPING_TIME(timing::Scope validation(timing::Validate, weight);)
    if (ingest == bad_time) return {};
    const char *user, *product, *type, *ts;
    size_t user_size = fields.template field<1>(user);
    if (!user_size) return {};
    size_t product_size = fields.template field<5>(product);
    if (!product_size) return {};
    size_t type_size = fields.template field<4>(type);
    bool purchase = type_size == 8 && std::memcmp(type, "purchase", 8) == 0;
    if (!purchase && (type_size != 4 || (std::memcmp(type, "view", 4) && std::memcmp(type, "cart", 4))))
        return {};
    size_t ts_size = fields.template field<2>(ts);
    int64_t time = window.event_time(View(ts, ts_size));
    if (time < window.start || time >= window.end || time < ingest - day_ms) return {};
    uint32_t price = 0;
    Category category = 0;
    if (purchase) {
        const char* text;
        size_t size = fields.template field<6>(text);
        int64_t value = decimal(View(text, size), 10'000'000);
        if (value <= 0) return {};
        price = value;
        category = static_cast<Category>(catalogs.category(View(product, product_size)));
    }
    Person person;
    {
        SHOPPING_TIME(timing::Scope hash(timing::PersonHash, weight);)
        person = ids.key(View(user, user_size));
    }
    return {person, {time, price, category}};
}

template<class Candidate>
void read_events(const Input& input, unsigned threads, Trace& trace,
                 const Window& window, const Catalogs& catalogs, const Input& links, Ids& ids,
                 People& people, ScatterWorkers<Candidate>& entries, size_t buckets, size_t ingest_buckets,
                 ScatterWorkers<typename Candidate::Survivor>& records, size_t partitions, size_t alias_batch,
                 int alias_load, bool alias_bitmap, const ScatterSettings& scatter) {
    std::optional<Ids> event_ids(std::in_place, 'e');
    std::array<int64_t, 8> chunk_ends{};
    LinkTrace link_trace;
    const bool measure = trace.enabled();
    const int64_t ingest_start = measure ? LinkTrace::now() : 0;
    link_trace.begin = ingest_start;
    auto scan_events = [&](size_t worker, size_t task) {
        ScatterRoute route(*entries[worker], scatter.prefetch);
        SHOPPING_TIME(timing::Scope scanning(timing::Scan);)
        scan_fields<7>(input.chunks[task], [&](const auto& fields) {
            SHOPPING_TIME(unsigned weight = timing::sample(256);)
            const char* event_id;
            size_t event_id_size = fields.template field<0>(event_id);
            if (!event_id_size) return;
            uint64_t id;
            {
                SHOPPING_TIME(timing::Scope hash(timing::Hash, weight);)
                id = event_ids->key(View(event_id, event_id_size));
            }
            int64_t ingest;
            {
                SHOPPING_TIME(timing::Scope parse(timing::Parse, weight);)
                const char* clock;
                size_t size = fields.template field<3>(clock);
                ingest = window.ingest_time(View(clock, size));
            }
            auto event = parse_event<typename Candidate::Category>(fields, window, catalogs, ids, ingest
                                                                  SHOPPING_TIME(, weight));
            SHOPPING_TIME(timing::Scope route(timing::Route, weight);)
            route.append(id & (ingest_buckets - 1), Candidate(id, ingest, event, window));
        });
        route.finish();
        if (measure) chunk_ends[worker] = LinkTrace::now();
    };
    {
        LinkBuild build(people, links, ids, alias_batch, alias_load, alias_bitmap,
                        scatter.huge, threads, measure ? &link_trace : nullptr);
        std::atomic<size_t> next_event{0};
        parallel(threads, threads, [&](unsigned, size_t worker) {
            bool finished = false;
            try {
                for (;;) {
                    auto stage = build.stage();
                    if (stage == LinkBuild::Cancelled) return;
                    if (build.work(stage)) continue;
                    if (!finished) {
                        size_t task = next_event.fetch_add(1, std::memory_order_relaxed);
                        if (task < input.chunks.size()) {
                            scan_events(worker, task);
                            continue;
                        }
                        entries[worker]->finish();
                        finished = true;
                    }
                    if (stage == LinkBuild::Done) return;
                    build.wait(stage);
                }
            } catch (...) {
                build.cancel();
                throw;
            }
        });
    }
    event_ids.reset();
    trace.mark("event_ingest_links_catalogs");
    if (measure) link_trace.report(ingest_start, std::max(ingest_start, *std::max_element(chunk_ends.begin(), chunk_ends.end())));
    if (memory::settings().release == memory::Release::Early && !ids.has_input_views()) {
        std::vector<memory::Region> retired;
        input.take_mappings(retired);
        links.take_mappings(retired);
        release_parallel(retired, threads);
        trace.mark("input_release");
    }

    struct Table {
        std::vector<uint32_t> positions;
        std::vector<const Candidate*> winners;
    };
    std::vector<Table> tables(threads);
    const unsigned shift = std::countr_zero(buckets);
    auto dedup = [&](unsigned worker, size_t count, auto visit) {
        if (!count) return;
        ScatterRoute routing(*records[worker], scatter.prefetch);
        auto route = [&](const typename Candidate::Survivor& event) {
            routing.append(event.person & (partitions - 1), event);
        };
        PersonBatch<typename Candidate::Survivor, decltype(route)> resolved(people, alias_batch, route);
        auto emit = [&](const Candidate& winner) {
            if (winner.payload.person) resolved.append(winner.payload);
        };
        constexpr size_t budget = 512 << 10;
        constexpr size_t bytes_per_winner = std::bit_ceil(sizeof(Candidate)) + sizeof(const Candidate*);
        if (count > budget / bytes_per_winner ||
            std::bit_ceil(count + count / 2 + 1) * sizeof(uint32_t) > budget - count * bytes_per_winner) {
            auto order = [](const Candidate& a, const Candidate& b) {
                return a.id == b.id ? a.ingest < b.ingest : a.id < b.id;
            };
            SortedRuns<Candidate, decltype(order)> runs(order);
            visit([&](const auto& row) { runs.append(row); });
            runs.finish();
            auto rows = runs.cursor();
            uint64_t seen = 0;
            while (!rows.empty()) {
                if (rows.front().id != seen) {
                    seen = rows.front().id;
                    emit(rows.front());
                }
                rows.pop();
            }
            resolved.finish();
            routing.finish();
            return;
        }
        auto& table = tables[worker];
        table.winners.clear();
        table.winners.reserve(count);
        size_t slots = std::bit_ceil(count + count / 2 + 1);
        auto& positions = table.positions;
        positions.assign(slots, 0);
        size_t mask = slots - 1;
        constexpr uint32_t index_mask = UINT16_MAX;
        visit([&](const Candidate& entry) {
            uint64_t hash = entry.id >> shift;
            uint32_t tag = uint32_t(hash ^ (hash >> 32)) & ~index_mask;
            size_t i = hash & mask;
            for (uint32_t pos; (pos = positions[i]); i = (i + 1) & mask) {
                if ((pos & ~index_mask) != tag) continue;
                auto*& winner = table.winners[(pos & index_mask) - 1];
                if (winner->id != entry.id) continue;
                if (entry.ingest < winner->ingest) winner = &entry;
                return;
            }
            table.winners.push_back(&entry);
            positions[i] = tag | static_cast<uint32_t>(table.winners.size());
        });
        for (auto* winner : table.winners) emit(*winner);
        resolved.finish();
        routing.finish();
    };
    parallel(threads, ingest_buckets, [&](unsigned worker, size_t bucket) {
        auto visit = [&](auto consume) {
            for (const auto& batch : entries) batch->visit(bucket, consume);
        };
        size_t count = 0;
        for (const auto& batch : entries) count += batch->size(bucket);
        if (ingest_buckets == buckets) {
            dedup(worker, count, visit);
            return;
        }
        if (!count) return;
        unsigned first_bits = std::countr_zero(ingest_buckets);
        size_t parts = buckets / ingest_buckets;
        Scatter<const Candidate*> split(parts, scatter.huge, scatter.streaming, scatter.staging_lines);
        ScatterRoute route(split, scatter.prefetch);
        visit([&](const Candidate& row) { route.append((row.id >> first_bits) & (parts - 1), &row); });
        route.finish();
        split.finish();
        for (size_t i = 0; i < parts; ++i) {
            dedup(worker, split.size(i), [&](auto consume) {
                split.visit(i, [&](const Candidate* row) { consume(*row); });
            });
        }
    }, [&](unsigned worker) { records[worker]->finish(); });
    trace.mark("event_dedup_route");
    std::vector<memory::Region> retired;
    if (memory::settings().release != memory::Release::None)
        for (auto& entry : entries) entry->take_mappings(retired);
    release_parallel(retired, threads);
    entries.clear();
    trace.mark("candidate_release");
}

template<class Time, class Validate>
ScatterWorkers<Click<Time>> read_clicks(const Input& input, unsigned threads, Trace& trace,
                                        size_t partitions, const People& people, size_t alias_batch,
                                        const ScatterSettings& scatter, Validate validate) {
    auto records = scatter_workers<Click<Time>>(threads, partitions, scatter);
    parallel(threads, input.chunks.size(), [&](unsigned worker, size_t task) {
        ScatterRoute routing(*records[worker], scatter.prefetch);
        auto route = [&](const Click<Time>& click) {
            routing.append(click.person & (partitions - 1), click);
        };
        PersonBatch<Click<Time>, decltype(route)> resolved(people, alias_batch, route);
        scan_fields<4>(input.chunks[task], [&](const auto& fields) {
            const char* id;
            if (fields.template field<0>(id)) validate(id, fields, resolved);
        });
        resolved.finish();
        routing.finish();
    }, [&](unsigned worker) { records[worker]->finish(); });
    trace.mark("click_scan_validation");
    return records;
}

template<class T> class ScratchArray {
    memory::Array<T> data_;
    size_t capacity_ = 0;
public:
    T* resize(size_t count) {
        if (count > capacity_) {
            capacity_ = std::bit_ceil(count);
            data_ = memory::array<T>(capacity_);
        }
        return data_.get();
    }
    T* get() const { return data_.get(); }
};

struct PersonTableFull {};

template<class Value, class Time, class Index> struct Timelines {
    using Category = decltype(unpack(std::declval<Value>(), std::declval<const Window&>()).category);
    struct PersonRange { Person person; Index begin, end, click_begin, click_end; };
    struct Slot { Person person; Index index, count; };
    std::vector<PersonRange> people;
    ScratchArray<Value> values, staged;
    ScratchArray<ClickValue<Time>> clicks;
    ScratchArray<Index> local_ids, click_ids;
    ScratchArray<uint64_t> keys, by_bucket;
    memory::Vector<SessionPurchase<Category>> purchases;
    std::vector<Index> bucket_offsets;
    std::vector<Index> event_offsets;
    memory::Vector<Slot> table;
};

template<bool Bucketed, class Value, class Time, class Index>
void collect_timelines(Timelines<Value, Time, Index>& result,
                       const ScatterWorkers<PersonEvent<Value>>& sources,
                       const ScatterWorkers<Click<Time>>& click_sources,
                       size_t bucket, size_t total, size_t click_count, unsigned partition_bits,
                       const Window& window, const BucketLayout& layout, size_t few_clicks = 0) {
    auto& people = result.people;
    auto& table = result.table;
    auto& offsets = result.event_offsets;
    people.clear();
    offsets.clear();
    using Slot = typename Timelines<Value, Time, Index>::Slot;
    size_t max_people = Bucketed ? std::min(total, layout.person_limit()) : 768;
    size_t max_slots = std::max<size_t>(32, std::bit_ceil((max_people * 4 + 2) / 3));
    size_t table_size = std::clamp<size_t>(std::bit_ceil(total / 16 + 1), 32, max_slots);
    table.assign(table_size, Slot{});
    auto find = [partition_bits](auto& slots, Person person) -> Slot& {
        size_t mask = slots.size() - 1;
        size_t i = (person >> partition_bits) & mask;
        while (slots[i].person && slots[i].person != person) i = (i + 1) & mask;
        return slots[i];
    };
    auto add = [&](Person person) -> Slot& {
        auto* slot = &find(table, person);
        if (slot->person) return *slot;
        if (people.size() == max_people) throw PersonTableFull{};
        if (people.size() == table.size() * 3 / 4) {
            memory::Vector<Slot> grown(table.size() * 2);
            for (auto entry : table) if (entry.person) find(grown, entry.person) = entry;
            table.swap(grown);
            slot = &find(table, person);
        }
        *slot = {person, static_cast<Index>(people.size()), 0};
        people.push_back({person, 0, 0, 0, 0});
        return *slot;
    };

    auto* local_ids = Bucketed ? nullptr : result.local_ids.resize(total);
    auto* staged = Bucketed ? nullptr : result.staged.resize(total);
    auto* keys = Bucketed ? result.keys.resize(total) : nullptr;
    if constexpr (Bucketed) {
        result.bucket_offsets.assign(layout.count, 0);
        result.purchases.clear();
    }
    size_t row_index = 0;
    for (const auto& batch : sources) batch->visit(bucket, [&](const auto& row) {
        auto& slot = add(row.person);
        ++slot.count;
        if constexpr (Bucketed) {
            auto event = unpack(row.value, window);
            uint32_t time;
            if constexpr (std::is_same_v<Value, uint64_t>) time = row.value >> 32;
            else time = uint64_t(event.time) - uint64_t(window.start);
            uint32_t time_bucket = time / gap_ms;
            ++result.bucket_offsets[time_bucket];
            keys[row_index++] = layout.key(slot.index, time_bucket, time);
            if (event.price) result.purchases.push_back({(uint64_t(slot.index) << 32) | time,
                                                        event.price, event.category});
        } else {
            staged[row_index] = row.value;
            local_ids[row_index++] = slot.index;
        }
    });
    offsets.resize(people.size());
    for (auto slot : table) if (slot.person) offsets[slot.index] = slot.count;
    auto* click_ids = result.click_ids.resize(click_count);
    constexpr Index absent = std::numeric_limits<Index>::max();
    row_index = 0;
    for (const auto& batch : click_sources) batch->visit(bucket, [&](const auto& row) {
        auto& slot = find(table, row.person);
        click_ids[row_index++] = slot.person ? slot.index : absent;
        if (slot.person) ++people[slot.index].click_end;
    });

    size_t next = 0, next_click = 0;
    for (size_t i = 0; i < people.size(); ++i) {
        auto& range = people[i];
        range.begin = next;
        next += offsets[i];
        range.end = next;
        offsets[i] = range.begin;
        range.click_begin = next_click;
        next_click += range.click_end;
        range.click_end = range.click_begin;
    }
    if constexpr (Bucketed) {
        bucket_scatter(layout, keys, result.by_bucket.resize(total), total, result.bucket_offsets, offsets);
        std::sort(result.purchases.begin(), result.purchases.end(),
                  [](const auto& a, const auto& b) { return a.key < b.key; });
    } else {
        auto* values = result.values.resize(total);
        for (size_t i = 0; i < total; ++i) values[offsets[local_ids[i]]++] = staged[i];
        for (const auto& range : people) sort_timeline(values + range.begin, values + range.end);
    }
    auto* clicks = result.clicks.resize(next_click);
    row_index = 0;
    for (const auto& batch : click_sources) batch->visit(bucket, [&](const auto& row) {
        Index id = click_ids[row_index++];
        if (id != absent) clicks[people[id].click_end++] = row.value;
    });
    for (const auto& range : people) {
        if (range.click_end - range.click_begin > std::max<size_t>(1, few_clicks))
            std::sort(clicks + range.click_begin, clicks + range.click_end,
                      [](const auto& a, const auto& b) {
                          return a.time == b.time ? click_id_after(a.id, b.id) : a.time < b.time;
                      });
    }
}

template<class Category> struct UserDaily {
    Person user;
    int32_t day;
    Category category;
    int64_t sessions = 0, events = 0, revenue = 0;
};
template<class Category> class UserOutput;
struct CampaignDaily { int64_t sessions = 0, purchases = 0, revenue = 0, assists = 0; };

uint64_t campaign_key(int32_t day, uint32_t campaign) {
    return (uint64_t(uint32_t(day)) << 32) | campaign;
}

class CampaignTotals {
    int32_t start_;
    size_t campaigns_;
    memory::Vector<CampaignDaily> dense_;
    memory::Map<uint64_t, CampaignDaily> sparse_;

public:
    CampaignTotals(const Window& window, size_t campaigns)
        : start_(utc_day(window.start)), campaigns_(campaigns) {
        size_t days = int64_t(utc_day(window.end - 1)) - start_ + 1;
        if (days <= 131072 / campaigns) dense_.resize(days * campaigns);
    }

    CampaignDaily* day(int32_t date) {
        return dense_.empty() ? nullptr : dense_.data() + (size_t(date - start_) * campaigns_);
    }

    CampaignDaily& operator[](uint64_t key) { return sparse_[key]; }

    void merge(const CampaignTotals& other) {
        auto add = [](CampaignDaily& to, const CampaignDaily& from) {
            to.sessions += from.sessions;
            to.purchases += from.purchases;
            to.revenue += from.revenue;
            to.assists += from.assists;
        };
        for (size_t i = 0; i < dense_.size(); ++i) add(dense_[i], other.dense_[i]);
        for (const auto& [key, row] : other.sparse_) add(sparse_[key], row);
    }

    template<class Use> void each(Use use) const {
        for (size_t i = 0; i < dense_.size(); ++i) {
            const auto& row = dense_[i];
            if (row.sessions || row.assists)
                use(campaign_key(start_ + int64_t(i / campaigns_), i % campaigns_), row);
        }
        for (const auto& [key, row] : sparse_) use(key, row);
    }
};

template<class Category> struct AttributionScratch {
    static constexpr bool sparse_categories = std::is_same_v<Category, size_t>;
    std::conditional_t<sparse_categories, std::unordered_map<Category, int64_t>, std::vector<int64_t>> spend;
    std::vector<Category> bought;
    std::vector<size_t> counts;
    std::vector<uint32_t> positions, active;

    explicit AttributionScratch(const Catalogs& catalogs)
        : counts(catalogs.campaigns.size()), positions(counts.size()) {
        if constexpr (!sparse_categories) spend.resize(catalogs.categories.size());
    }
};

template<bool Few = false, class Sessions, class ClickCursor, class Category>
void aggregate_person(Person person, Sessions sessions, ClickCursor first, ClickCursor last,
                      const Catalogs& catalogs, const Window& window, CampaignTotals& totals,
                      UserOutput<Category>& users, AttributionScratch<Category>& scratch) {
    constexpr bool sparse_categories = AttributionScratch<Category>::sparse_categories;
    auto& [spend, bought, counts, positions, active] = scratch;
    using Time = decltype(last.front().time);
    SHOPPING_TIME(unsigned weight = timing::sample(64);)
    uint32_t latest = 0;
    do {
        int32_t day = utc_day(sessions.start());
        UserDaily<Category> daily{person, day, no_category<Category>};
        auto* day_campaigns = totals.day(day);
        do {
            int64_t start = sessions.start(), revenue = 0;
            daily.events += sessions.take([&](uint32_t price, Category category) {
                auto& category_spend = spend[category];
                if (!category_spend) bought.push_back(category);
                category_spend += price;
                revenue += price;
            });
            ++daily.sessions;
            daily.revenue += revenue;

            SHOPPING_TIME(timing::Scope attribution(timing::Attribution, weight);)
            const int64_t click_start = std::is_unsigned_v<Time> ? start - (window.start - week_ms) : start;
            uint32_t winner = 0;
            if constexpr (Few) {
                const char* best_id = nullptr;
                int64_t best_time = INT64_MIN;
                for (auto click = first; !click.empty(); click.pop()) {
                    const auto& row = click.front();
                    if (uint64_t(click_start) - uint64_t(row.time) > week_ms) continue;
                    if (!counts[row.campaign]) {
                        active.push_back(row.campaign);
                        counts[row.campaign] = 1;
                    }
                    if (row.time > best_time || (row.time == best_time && (!best_id || click_id_after(best_id, row.id)))) {
                        best_time = row.time;
                        best_id = row.id;
                        winner = row.campaign;
                    }
                }
            } else {
                const int64_t lower = click_start - week_ms;
                while (first.position() != last.position() && first.front().time < lower) {
                    uint32_t campaign = first.front().campaign;
                    first.pop();
                    if (--counts[campaign] == 0) {
                        uint32_t tail = active.back();
                        active[positions[campaign]] = tail;
                        positions[tail] = positions[campaign];
                        active.pop_back();
                    }
                }
                while (!last.empty() && last.front().time <= click_start) {
                    auto row = last.front();
                    last.pop();
                    if (row.time < lower) {
                        first.pop();
                        continue;
                    }
                    uint32_t campaign = latest = row.campaign;
                    if (counts[campaign]++ == 0) {
                        positions[campaign] = active.size();
                        active.push_back(campaign);
                    }
                }
                winner = first.position() != last.position() ? latest : 0;
            }
            auto& credit = day_campaigns ? day_campaigns[winner] : totals[campaign_key(day, winner)];
            ++credit.sessions;
            credit.purchases += revenue != 0;
            credit.revenue += revenue;
            for (uint32_t campaign : active) {
                if (day_campaigns) ++day_campaigns[campaign].assists;
                else ++totals[campaign_key(day, campaign)].assists;
                if constexpr (Few) counts[campaign] = 0;
            }
            if constexpr (Few) active.clear();
        } while (!sessions.empty() && utc_day(sessions.start()) == day);

        int64_t high = 0;
        for (Category category : bought) {
            int64_t revenue = spend[category];
            if (revenue > high || (revenue == high &&
                catalogs.categories[category] < catalogs.categories[daily.category])) {
                daily.category = category;
                high = revenue;
            }
            if constexpr (sparse_categories) spend.erase(category);
            else spend[category] = 0;
        }
        bought.clear();
        users.append(daily);
    } while (!sessions.empty());
    for (uint32_t campaign : active) counts[campaign] = 0;
    active.clear();
}

template<bool Bucketed, class Value, class Time, class Index, class Category>
void aggregate(const Timelines<Value, Time, Index>& timelines, const Catalogs& catalogs,
               const Window& window, CampaignTotals& totals, UserOutput<Category>& users,
               AttributionScratch<Category>& scratch, size_t few_clicks) {
    const auto* clicks = timelines.clicks.get();
    std::span<const SessionPurchase<Category>> purchases(timelines.purchases);
    for (size_t index = 0; index < timelines.people.size(); ++index) {
        auto [person, begin, end, click_begin, click_end] = timelines.people[index];
        SpanCursor<ClickValue<Time>> range{clicks, click_begin, click_end};
        auto consume = [&](auto sessions) {
            if (few_clicks && click_end - click_begin <= few_clicks)
                aggregate_person<true>(person, sessions, range, range, catalogs, window, totals, users, scratch);
            else
                aggregate_person(person, sessions, range, range, catalogs, window, totals, users, scratch);
        };
        if constexpr (Bucketed) {
            auto* keys = timelines.keys.get();
            consume(BucketSessions(std::span(keys + begin, end - begin), purchases, index, window.start));
        } else {
            SpanCursor<Value> events{timelines.values.get(), begin, end};
            consume(SortedSessions(events, window));
        }
    }
}

template<class Value, class Time, class Category>
void aggregate_runs(const ScatterWorkers<PersonEvent<Value>>& sources, const ScatterWorkers<Click<Time>>& clicks,
                    size_t bucket, const Catalogs& catalogs, const Window& window,
                    CampaignTotals& totals, UserOutput<Category>& users,
                    AttributionScratch<Category>& scratch, GroupTrace* group_trace, size_t few_clicks) {
    Person frequent = 0;
    size_t votes = 0;
    for (const auto& batch : sources) if (const auto* row = batch->first(bucket)) {
        if (!votes) frequent = row->person;
        if (row->person == frequent) ++votes;
        else --votes;
    }
    if (!votes) frequent = 0;
    auto event_value_less = [](const auto& a, const auto& b) { return time_key(a) < time_key(b); };
    auto click_value_less = [](const auto& a, const auto& b) {
        return a.time == b.time ? click_id_after(a.id, b.id) : a.time < b.time;
    };
    auto event_less = [&](const auto& a, const auto& b) {
        return a.person == b.person ? event_value_less(a.value, b.value) : a.person < b.person;
    };
    auto click_less = [&](const auto& a, const auto& b) {
        return a.person == b.person ? click_value_less(a.value, b.value) : a.person < b.person;
    };
    SortedRuns<PersonEvent<Value>, decltype(event_less)> event_runs(event_less);
    SortedRuns<Click<Time>, decltype(click_less)> click_runs(click_less);
    uint64_t span = uint64_t(window.end) - uint64_t(window.start);
    std::optional<TimeRuns<Value, decltype(event_value_less)>> frequent_events;
    std::optional<TimeRuns<ClickValue<Time>, decltype(click_value_less)>> frequent_clicks;
    if (frequent) frequent_events.emplace(span, event_value_less);
    struct ClickRange { size_t count = 0, end = 0; };
    memory::Map<Person, ClickRange> click_ranges;
    ScratchArray<ClickValue<Time>> direct_clicks;
    auto direct = [&](const ClickRange& range) { return few_clicks && range.count <= few_clicks; };
    {
        GroupTrace::Scope measured{group_trace};
        SHOPPING_TIME(timing::Scope group(timing::Group);)
        if (frequent) click_ranges.try_emplace(frequent);
        for (const auto& batch : sources) batch->visit(bucket, [&](const auto& row) {
            if (row.person == frequent) {
                auto offset = uint64_t(unpack(row.value, window).time) - uint64_t(window.start);
                frequent_events->append(offset, row.value);
            } else {
                event_runs.append(row);
                click_ranges.try_emplace(row.person);
            }
        });
        if (few_clicks) for (const auto& batch : clicks) batch->visit(bucket, [&](const auto& row) {
            if (auto info = click_ranges.find(row.person); info != click_ranges.end()) ++info->second.count;
        });
        size_t small_count = 0;
        for (auto& [person, range] : click_ranges) if (direct(range)) {
            range.end = small_count;
            small_count += range.count;
        }
        auto* small = direct_clicks.resize(small_count);
        if (frequent && !direct(click_ranges.at(frequent))) frequent_clicks.emplace(span + week_ms, click_value_less);
        for (const auto& batch : clicks) batch->visit(bucket, [&](const auto& row) {
            auto info = click_ranges.find(row.person);
            if (info == click_ranges.end()) return;
            if (direct(info->second)) {
                small[info->second.end++] = row.value;
                return;
            }
            if (row.person == frequent) {
                uint64_t offset = row.value.time;
                if constexpr (std::is_signed_v<Time>) offset -= uint64_t(window.start - week_ms);
                frequent_clicks->append(offset, row.value);
            } else {
                click_runs.append(row);
            }
        });
        event_runs.finish();
        click_runs.finish();
        if (frequent_events) frequent_events->finish();
        if (frequent_clicks) frequent_clicks->finish();
    }

    SHOPPING_TIME(timing::Scope sessions(timing::Sessions);)
    auto consume_few = [&](Person person, auto sessions, const ClickRange& info) {
        SpanCursor<ClickValue<Time>> range{direct_clicks.get(), info.end - info.count, info.end};
        aggregate_person<true>(person, sessions, range, range, catalogs, window, totals, users, scratch);
    };
    if (frequent) {
        auto sessions = SortedSessions(frequent_events->cursor(), window);
        if (frequent_clicks)
            aggregate_person(frequent, std::move(sessions), frequent_clicks->cursor(), frequent_clicks->cursor(),
                             catalogs, window, totals, users, scratch);
        else
            consume_few(frequent, std::move(sessions), click_ranges.at(frequent));
    }
    auto events = event_runs.cursor();
    auto first = click_runs.cursor(), last = click_runs.cursor();
    while (!events.empty()) {
        Person person = events.front().person;
        auto skip = [person](auto& cursor) {
            while (!cursor.empty() && cursor.front().person < person) cursor.pop();
        };
        auto sessions = SortedSessions(PersonCursor{events, person}, window);
        const auto& info = click_ranges.at(person);
        if (direct(info)) {
            consume_few(person, sessions, info);
        } else {
            skip(first);
            skip(last);
            aggregate_person(person, sessions, PersonCursor{first, person}, PersonCursor{last, person},
                             catalogs, window, totals, users, scratch);
        }
    }
}

using parquet::Column;
constexpr Column user_schema[] = {{"user_id", Column::Text, 0}, {"day", Column::Int32, 6},
    {"sessions", Column::Int64}, {"events", Column::Int64}, {"revenue_cents", Column::Int64},
    {"top_category", Column::Text, 0, true}};
constexpr Column campaign_schema[] = {{"campaign_id", Column::Text, 0}, {"channel", Column::Text, 0},
    {"day", Column::Int32, 6}, {"sessions", Column::Int64}, {"purchasing_sessions", Column::Int64},
    {"revenue_cents", Column::Int64}, {"assisted_sessions", Column::Int64}};

size_t longest_string(const memory::Vector<View>& strings) {
    size_t length = 0;
    for (View value : strings) length = std::max(length, value.size());
    return length;
}

template<class Category> class UserOutput {
    const Catalogs& catalogs_;
    const Ids& ids_;
    fs::path path_;
    size_t group_rows_, count_ = 0;
    std::array<parquet::Page, std::size(user_schema)> columns_;
    std::unique_ptr<parquet::File> file_;

    void flush() {
        if (!file_) file_ = std::make_unique<parquet::File>(path_, user_schema);
        file_->append(count_, columns_);
        count_ = 0;
    }

public:
    UserOutput(const Catalogs& catalogs, const Ids& ids, fs::path path, size_t group_rows)
        : catalogs_(catalogs), ids_(ids), path_(std::move(path)), group_rows_(group_rows) {}

    void append(const UserDaily<Category>& row) {
        using parquet::little_endian;
        using parquet::plain;
        char buffer[9];
        plain(columns_[0].values, ids_.name(row.user, buffer));
        little_endian(columns_[1].values, row.day);
        little_endian(columns_[2].values, row.sessions);
        little_endian(columns_[3].values, row.events);
        little_endian(columns_[4].values, row.revenue);
        auto& category = columns_[5];
        if ((count_ & 7) == 0) category.present.push_back(0);
        if (row.category != no_category<Category>) {
            category.present.back() |= uint8_t{1} << (count_ & 7);
            plain(category.values, catalogs_.categories[row.category]);
        }
        if (++count_ == group_rows_) flush();
    }

    bool empty() const { return !file_ && !count_; }

    void finish(bool force_empty) {
        if (count_ || force_empty) flush();
        if (file_) file_->finish();
    }
};

void write_campaigns(const fs::path& path, const CampaignTotals& totals, const Catalogs& catalogs) {
    SHOPPING_TIME(timing::Scope encode(timing::Encode);)
    using parquet::little_endian;
    using parquet::plain;
    memory::Vector<std::pair<uint64_t, const CampaignDaily*>> rows;
    totals.each([&](uint64_t key, const CampaignDaily& row) { rows.emplace_back(key, &row); });
    parquet::File file(path, campaign_schema);
    size_t group_rows = parquet::row_group_limit(
        std::max(longest_string(catalogs.campaigns), longest_string(catalogs.channels)));
    for (size_t offset = 0; offset < rows.size(); offset += group_rows) {
        auto group = std::span(rows).subspan(offset, std::min(group_rows, rows.size() - offset));
        std::array<parquet::Page, std::size(campaign_schema)> columns;
        for (auto [key, values] : group) {
            uint32_t campaign = key;
            plain(columns[0].values, catalogs.campaigns[campaign]);
            plain(columns[1].values, catalogs.channels[campaign]);
            little_endian(columns[2].values, static_cast<int32_t>(key >> 32));
            little_endian(columns[3].values, values->sessions);
            little_endian(columns[4].values, values->purchases);
            little_endian(columns[5].values, values->revenue);
            little_endian(columns[6].values, values->assists);
        }
        file.append(group.size(), columns);
    }
    file.finish();
}

std::string env(const char* key, const char* fallback) {
    const char* value = std::getenv(key);
    return value && *value ? value : fallback;
}

uint64_t number_env(const char* key, uint64_t fallback, uint64_t minimum = 1) {
    const char* setting = std::getenv(key);
    if (!setting || !*setting) return fallback;
    View text(setting);
    uint64_t value = 0;
    auto parsed = std::from_chars(text.data(), text.data() + text.size(), value);
    if (parsed.ec != std::errc() || parsed.ptr != text.data() + text.size() || value < minimum)
        throw std::runtime_error(std::string("invalid ") + key);
    return value;
}

template<class Candidate>
void process(const Window& window, const Input& links, const Catalogs& catalogs,
             const Input& event_input, const Input& click_input, const fs::path& out,
             unsigned threads, Trace& trace) {
    Ids ids('u');
    People people;
    using Survivor = typename Candidate::Survivor;
    size_t buckets = partition_override("SHOPPING_EVENT_BUCKETS",
        partition_size(event_input.bytes, 72,
                       number_env("SHOPPING_EVENT_BUCKET_ROWS", 192'000 / sizeof(Candidate))));
    size_t partitions = partition_override("SHOPPING_PERSON_PARTITIONS",
        partition_size(event_input.bytes + click_input.bytes, 72,
                       number_env("SHOPPING_PERSON_PARTITION_ROWS", 12'000)));
    size_t alias_batch = std::min<uint64_t>(max_alias_batch, number_env("SHOPPING_ALIAS_BATCH", 32, 0));
    auto alias_load = number_env("SHOPPING_ALIAS_LOAD", 50);
    if (alias_load != 50 && alias_load != 75) throw std::runtime_error("invalid SHOPPING_ALIAS_LOAD");
    bool alias_bitmap = setting("SHOPPING_ALIAS_BITMAP", false);
    auto staging_lines = number_env("SHOPPING_STAGING_LINES", 1);
    if (staging_lines != 1 && staging_lines != 2 && staging_lines != 4)
        throw std::runtime_error("invalid SHOPPING_STAGING_LINES");
    ScatterSettings scatter{setting("SHOPPING_HUGE", true), setting("SHOPPING_NT", true),
                            static_cast<unsigned>(staging_lines),
                            std::min<uint64_t>(max_staging_prefetch, number_env("SHOPPING_STAGING_PREFETCH", 0, 0))};
    size_t ingest_buckets = setting("SHOPPING_EVENT_TWO_LEVEL", false) ? std::min<size_t>(64, buckets) : buckets;
    auto entries = scatter_workers<Candidate>(threads, ingest_buckets, scatter);
    auto events = scatter_workers<Survivor>(threads, partitions, scatter);
    read_events(event_input, threads, trace, window, catalogs, links, ids, people, entries, buckets, ingest_buckets,
                events, partitions, alias_batch, alias_load, alias_bitmap, scatter);
    if (trace.enabled())
        std::fprintf(stderr, "partitions %zu %zu\n", buckets, partitions);
    using ClickTime = std::conditional_t<std::is_same_v<Candidate, CompactCandidate>, uint32_t, int64_t>;
    auto clicks = read_clicks<ClickTime>(click_input, threads, trace, partitions, people, alias_batch, scatter,
                                       [&](const char* id, const auto& f, auto& batch) {
        const char *user, *ts, *campaign_id;
        size_t user_size = f.template field<1>(user);
        if (!user_size) return;
        size_t ts_size = f.template field<3>(ts);
        int64_t time = window.event_time(View(ts, ts_size));
        if (time < window.start - week_ms || time >= window.end) return;
        size_t campaign_size = f.template field<2>(campaign_id);
        uint32_t campaign = catalogs.campaign(View(campaign_id, campaign_size));
        if (!campaign) return;
        Person person = ids.key(View(user, user_size));
        if constexpr (std::is_unsigned_v<ClickTime>) time -= window.start - week_ms;
        batch.append({person, {id, static_cast<ClickTime>(time), campaign}});
    });

    fs::create_directories(out / "user_daily");
    fs::create_directories(out / "campaign_daily");
    size_t user_group_rows = parquet::row_group_limit(
        std::max(ids.longest(), longest_string(catalogs.categories)));
    using Category = typename Candidate::Category;
    using Value = decltype(Survivor::value);
    auto layout = BucketLayout::for_span(uint64_t(window.end) - uint64_t(window.start));
    size_t bucket_minimum = number_env("SHOPPING_BUCKET_MIN_EVENTS", 128);
    size_t few_clicks = number_env("SHOPPING_FEW_CLICKS", 16, 0);
    bool bucket_sessions = layout.count && !setting("SHOPPING_FORCE_SORT", false);
    auto partition_bits = std::countr_zero(partitions);
    struct Worker {
        CampaignTotals campaigns;
        AttributionScratch<Category> attribution;
        Timelines<Value, ClickTime, uint32_t> timelines;
        UserOutput<Category> users;
        GroupTrace group;

        Worker(const Window& window, const Catalogs& catalogs, const Ids& ids, fs::path path, size_t group_rows)
            : campaigns(window, catalogs.campaigns.size()), attribution(catalogs),
              users(catalogs, ids, std::move(path), group_rows) {}
    };
    std::vector<Worker> workers;
    workers.reserve(threads);
    for (unsigned i = 0; i < threads; ++i)
        workers.emplace_back(window, catalogs, ids, out / "user_daily" / ("part-" + std::to_string(i) + ".parquet"),
                             user_group_rows);
    std::vector<size_t> jobs(partitions), event_counts(partitions), click_counts(partitions);
    for (size_t bucket = 0; bucket < partitions; ++bucket) {
        jobs[bucket] = bucket;
        for (const auto& batch : events) event_counts[bucket] += batch->size(bucket);
        for (const auto& batch : clicks) click_counts[bucket] += batch->size(bucket);
    }
    std::sort(jobs.begin(), jobs.end(),
              [&](size_t a, size_t b) { return event_counts[a] > event_counts[b]; });
    parallel(threads, partitions, [&](unsigned worker, size_t task) {
        size_t bucket = jobs[task], count = event_counts[bucket], click_count = click_counts[bucket];
        if (!count) return;
        auto& state = workers[worker];
        auto* group = trace.enabled() ? &state.group : nullptr;
        auto consume = [&]<bool Bucketed>() {
            {
                GroupTrace::Scope measured{group};
                SHOPPING_TIME(timing::Scope group(timing::Group);)
                collect_timelines<Bucketed>(state.timelines, events, clicks, bucket, count, click_count,
                                            partition_bits, window, layout, few_clicks);
            }
            SHOPPING_TIME(timing::Scope sessions(timing::Sessions);)
            aggregate<Bucketed>(state.timelines, catalogs, window, state.campaigns, state.users, state.attribution,
                                few_clicks);
        };
        constexpr size_t event_bytes = std::bit_ceil(sizeof(Survivor)) + 2 * sizeof(Value) + sizeof(uint32_t);
        constexpr size_t click_bytes = std::bit_ceil(sizeof(Click<ClickTime>)) + sizeof(ClickValue<ClickTime>) + sizeof(uint32_t);
        constexpr size_t record_budget = 448 << 10;
        auto oversized = [&] {
            aggregate_runs(events, clicks, bucket, catalogs, window, state.campaigns, state.users,
                           state.attribution, group, few_clicks);
        };
        bool bucketed = false;
        if (bucket_sessions && count >= bucket_minimum && count <= UINT32_MAX && click_count <= UINT32_MAX) {
            try {
                consume.template operator()<true>();
                bucketed = true;
            } catch (const PersonTableFull&) {
            }
        }
        if (!bucketed) {
            if (count > record_budget / event_bytes ||
                click_count > (record_budget - count * event_bytes) / click_bytes) {
                oversized();
            } else {
                try {
                    consume.template operator()<false>();
                } catch (const PersonTableFull&) {
                    oversized();
                }
            }
        }
        if (group) {
            group->bucket_partitions += bucketed;
            group->sorted_partitions += !bucketed;
            group->bucket_rows += bucketed ? count : 0;
        }
    });
    trace.mark("sessions_user_output");
    if (trace.enabled()) {
        GroupTrace group;
        for (const auto& worker : workers) {
            group.cpu_ns += worker.group.cpu_ns;
            group.bucket_partitions += worker.group.bucket_partitions;
            group.sorted_partitions += worker.group.sorted_partitions;
            group.bucket_rows += worker.group.bucket_rows;
        }
        std::fprintf(stderr, "group cpu_ms %.3f bucket_partitions %zu sorted_partitions %zu bucket_rows %zu\n",
                     double(group.cpu_ns) / 1e6, group.bucket_partitions, group.sorted_partitions, group.bucket_rows);
    }

    bool empty = std::all_of(workers.begin(), workers.end(), [](const auto& worker) { return worker.users.empty(); });
    parallel(threads, threads + 1, [&](unsigned, size_t task) {
        if (task) {
            SHOPPING_TIME(timing::Scope encode(timing::Encode);)
            workers[task - 1].users.finish(empty && task == 1);
            return;
        }
        auto& totals = workers.front().campaigns;
        for (unsigned i = 1; i < threads; ++i) totals.merge(workers[i].campaigns);
        write_campaigns(out / "campaign_daily" / "part-0.parquet", totals, catalogs);
    });
    trace.mark("finish");
    SHOPPING_TIME(timing::report();)
    if (__gcov_dump) __gcov_dump();
    if (__llvm_profile_write_file) __llvm_profile_write_file();
    if (setting("SHOPPING_UNMAP", false)) return;
    if (memory::settings().release != memory::Release::None) release_parallel(memory::take_all(), threads);
    trace.mark("exit_release");
    _exit(0);
}

int main(int argc, char** argv) try {
    Trace trace;
    unsigned threads = std::min<uint64_t>(8, number_env("NCPU", 8));
    fs::path data = argc > 1 ? argv[1] : env("DATA_DIR", "/data");
    fs::path out = argc > 2 ? argv[2] : env("OUTPUT_DIR", "/output");
    const Window window(data / "manifest.json");
    const Input links(data / "links", true), products(data / "products.csv", false),
        campaigns(data / "campaigns.csv", false), event_input(data / "events", true),
        click_input(data / "clicks", true);
    trace.mark("mappings");
    const Catalogs catalogs(products, campaigns);
    if (compact_events(window, catalogs))
        process<CompactCandidate>(window, links, catalogs, event_input, click_input, out, threads, trace);
    else if (catalogs.categories.size() <= no_category<uint32_t>)
        process<WideCandidate<uint32_t>>(window, links, catalogs, event_input, click_input, out, threads, trace);
    else
        process<WideCandidate<size_t>>(window, links, catalogs, event_input, click_input, out, threads, trace);
    return 0;
} catch (const std::exception& error) {
    std::fprintf(stderr, "%s\n", error.what());
    return 1;
}
