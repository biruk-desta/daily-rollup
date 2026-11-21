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
constexpr unsigned partition_count = 128;
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

template<class V> using StringMap = std::unordered_map<View, V, Hash>;
template<class T> using Partitions = std::array<std::vector<T>, partition_count>;

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
            data_ = View(static_cast<const char*>(address), size);
        } else {
            close(fd);
        }
    }
    Mapping(const Mapping&) = delete;
    Mapping& operator=(const Mapping&) = delete;
    ~Mapping() { if (!data_.empty()) munmap(const_cast<char*>(data_.data()), data_.size()); }
    View bytes() const { return data_; }
};

struct Input {
    std::vector<std::unique_ptr<Mapping>> maps;
    std::vector<View> chunks;

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
};


template<class Work> void parallel(unsigned threads, size_t tasks, Work work) {
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
    // XCR0 must enable SSE/YMM and, for AVX-512, the opmask and ZMM state.
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
    bool enabled_ = std::getenv("SHOPPING_TRACE") != nullptr;
    int64_t time_ = 0;
    rusage usage_{};
    SHOPPING_TIME(timing::Hardware hardware_;)

    static int64_t micros(timeval t) { return int64_t(t.tv_sec) * 1'000'000 + t.tv_usec; }

public:
    Trace() {
        if (!enabled_) return;
        constexpr const char* names[] = {"scalar", "NEON", "AVX2", "AVX-512BW"};
        std::fprintf(stderr, "CSV %s\n", names[static_cast<unsigned>(scanner)]);
        std::fprintf(stderr, "trace columns: phase, monotonic_ns, wall_s, cpu_s, minor_faults, major_faults\n");
        mark("start");
    }

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
        SHOPPING_TIME(hardware_.mark(phase);)
    }

};

// A block supplies the positions of commas, LF and quotes. Shift is zero for
// x86's one bit per byte and two for NEON's four bits per byte.
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
    // Consume only the prefix so the following loads stay within cache lines.
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

template<size_t N, class Row> void scan(View chunk, Row row) {
#if defined(__x86_64__)
    if constexpr (N >= 4) {
        if (scanner == Scanner::Avx512) return scan_indexed<N, csv_avx512>(chunk, row);
        if (scanner == Scanner::Avx2) return scan_indexed<N, csv_avx2>(chunk, row);
    }
    if (scanner == Scanner::Avx512) return scan_avx512<N>(chunk, row);
    if (scanner == Scanner::Avx2) return scan_avx2<N>(chunk, row);
#elif defined(__aarch64__)
    if constexpr (N >= 4) {
        if (scanner == Scanner::Neon) return scan_indexed<N, csv_neon>(chunk, row);
    }
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
    // The two colons must match exactly. Validate all six digits together.
    word ^= 0x30303a30303a3030ULL;
    if ((word | (word + 0x0606000606000606ULL)) & 0xf0f0fff0f0fff0f0ULL) return -1;
    uint64_t pairs = word * 10 + (word >> 8);
    unsigned h = uint8_t(pairs), m = uint8_t(pairs >> 24), s = uint8_t(pairs >> 48);
    if ((h > 23) | (m > 59) | (s > 59)) return -1;
    return h * 3600 + m * 60 + s;
}

int64_t ingest_timestamp(View s) {
    // Ingest is F1 by contract. Use the same calendar with unchecked clock digits.
    if (s.size() != 20) return bad_time;
    auto pair = [](const char* p) { return (p[0] - '0') * 10 + (p[1] - '0'); };
    const char* p = s.data();
    int y = pair(p) * 100 + pair(p + 2), m = pair(p + 5), d = pair(p + 8);
    if (unsigned(y) >= years.size() || unsigned(m - 1) >= 12) return bad_time;
    auto [year_days, leap] = years[y];
    int days = year_days + month_offsets[m - 1] + d - 1 + (m > 2 && leap);
    return int64_t(days) * day_ms + ((pair(p + 11) * 60 + pair(p + 14)) * 60 + pair(p + 17)) * 1000;
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
        if (text.size() == 20) {
            if (auto* date = cached_date(text.data())) {
                int seconds = clock_seconds(text.data() + 11);
                if (seconds >= 0) return int64_t(date->day) * day_ms + seconds * 1000;
            }
        }
        return ingest_timestamp(text);
    }
};

using Person = uint64_t;

class Ids {
    static constexpr uint64_t external = uint64_t{1} << 63;
    static constexpr uint64_t multiplier = 0x94d049bb133111ebULL;
    static constexpr uint64_t inverse = 0x319642b2d24d8ec3ULL;
    static_assert(multiplier * inverse == 1);
    struct Shard {
        std::mutex lock;
        StringMap<uint32_t> indices;
        // Fields point into input mappings that outlive this index.
        std::vector<View> names;
        size_t longest = 0;
    };
    std::array<Shard, 64> shards_;

    // Nonzero ASCII bytes take seven bits each. The zero padding distinguishes
    // different lengths. Other byte strings use the high-bit tag and a shard index.
    static uint64_t inline_id(View bytes) {
        if (bytes.size() == 8 || bytes.size() == 9) {
            uint64_t word;
            std::memcpy(&word, bytes.data(), 8);
            constexpr uint64_t high = 0x8080808080808080ULL;
            if ((word & high) || ((word - 0x0101010101010101ULL) & ~word & high)) return 0;
            uint64_t ninth = bytes.size() == 9 ? uint8_t(bytes[8]) : 0;
            if (bytes.size() == 9 && (ninth == 0 || ninth > 127)) return 0;
            word = (word & 0x007f007f007f007fULL) | ((word >> 1) & 0x3f803f803f803f80ULL);
            word = (word & 0x00003fff00003fffULL) | ((word >> 2) & 0x0fffc0000fffc000ULL);
            return (word & 0x0fffffffULL) | ((word >> 4) & 0x00fffffff0000000ULL) | (ninth << 56);
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

    // Called after the input stages have stopped interning IDs.
    View name(Person key, char (&buffer)[9]) const {
        uint64_t value = (key ^ (key >> 32)) * inverse;
        if (value & external) return shards_[(value >> 32) & 63].names[uint32_t(value)];
        uint64_t word = (value & 0x0fffffffULL) | ((value & 0x00fffffff0000000ULL) << 4);
        word = (word & 0x00003fff00003fffULL) | ((word & 0x0fffc0000fffc000ULL) << 2);
        word = (word & 0x007f007f007f007fULL) | ((word & 0x3f803f803f803f80ULL) << 1);
        std::memcpy(buffer, &word, 8);
        buffer[8] = value >> 56;
        return View(buffer, buffer[8] ? 9 : (std::bit_width(word) + 7) / 8);
    }
};

class People {
    struct Alias { Person key = 0, value = 0; };
    struct MemberIndexOverflow {};
    std::vector<Alias> aliases_ = std::vector<Alias>(32);
    std::vector<uint64_t> has_alias_;

    size_t alias_bit(Person key) const {
        return (key >> 12) & (has_alias_.size() * 64 - 1);
    }

    static size_t slot(const std::vector<Alias>& aliases, Person key) {
        size_t mask = aliases.size() - 1;
        size_t i = key & mask;
        while (aliases[i].key && aliases[i].key != key) i = (i + 1) & mask;
        return i;
    }

    template<class Index> void build(const Input& links, Ids& ids) {
        struct Member { View name; Person key; Index parent, size; };
        std::vector<Member> members;
        auto add = [&](View name) {
            Person key = ids.key(name);
            size_t i = slot(aliases_, key);
            if (aliases_[i].key) return static_cast<Index>(aliases_[i].value);
            // The largest possible component must also fit in Index.
            if (members.size() == std::numeric_limits<Index>::max()) throw MemberIndexOverflow{};
            if ((members.size() + 1) * 4 > aliases_.size() * 3) {
                std::vector<Alias> grown(aliases_.size() * 2);
                for (const auto& alias : aliases_)
                    if (alias.key) grown[slot(grown, alias.key)] = alias;
                aliases_.swap(grown);
                i = slot(aliases_, key);
            }
            auto index = static_cast<Index>(members.size());
            aliases_[i] = {key, index};
            members.push_back({name, key, index, 1});
            return index;
        };
        auto root = [&](Index i) {
            while (i != members[i].parent) {
                members[i].parent = members[members[i].parent].parent;
                i = members[i].parent;
            }
            return i;
        };
        for (View chunk : links.chunks) scan<2>(chunk, [&](View, const Fields<2>& fields) {
            if (fields[0].empty() || fields[1].empty()) return;
            Index a = root(add(fields[0])), b = root(add(fields[1]));
            if (a == b) return;
            if (members[a].size < members[b].size) std::swap(a, b);
            members[b].parent = a;
            members[a].size += members[b].size;
            if (members[b].name < members[a].name) {
                members[a].name = members[b].name;
                members[a].key = members[b].key;
            }
        });
        has_alias_.resize(std::max<size_t>(1, aliases_.size() / 8));
        for (auto& alias : aliases_) if (alias.key) {
            alias.value = members[root(alias.value)].key;
            if (alias.key != alias.value) {
                size_t bit = alias_bit(alias.key);
                has_alias_[bit / 64] |= uint64_t{1} << (bit % 64);
            }
        }
    }

public:
    People(const Input& links, Ids& ids) {
        try {
            build<uint32_t>(links, ids);
        } catch (const MemberIndexOverflow&) {
            aliases_ = std::vector<Alias>(32);
            build<size_t>(links, ids);
        }
    }

    Person name(Person key) const {
        size_t bit = alias_bit(key);
        if (!(has_alias_[bit / 64] & (uint64_t{1} << (bit % 64)))) return key;
        auto value = aliases_[slot(aliases_, key)].value;
        return value ? value : key;
    }
};


struct Catalogs {
    StringMap<size_t> products;
    StringMap<uint32_t> campaign_ids;
    std::vector<View> categories{"unknown"};
    std::vector<View> campaigns{"organic"}, channels{"organic"};

    Catalogs(const Input& product_input, const Input& campaign_input) {
        StringMap<size_t> category_ids;
        for (View chunk : product_input.chunks) scan<2>(chunk, [&](View, const Fields<2>& f) {
            auto [where, inserted] = category_ids.try_emplace(f[1], categories.size());
            if (inserted) categories.push_back(f[1]);
            products.emplace(f[0], where->second);
        });
        for (View chunk : campaign_input.chunks) scan<2>(chunk, [&](View, const Fields<2>& f) {
            campaign_ids.emplace(f[0], campaigns.size());
            campaigns.push_back(f[0]);
            channels.push_back(f[1]);
        });
    }
    size_t category(View product) const {
        auto result = products.find(product);
        return result == products.end() ? 0 : result->second;
    }
};

template<class Category> struct EventValue { int64_t time; uint32_t price; Category category; };
template<class Value> struct PersonEvent { Person person; Value value; };
template<class Category> using Event = PersonEvent<EventValue<Category>>;
using CompactEvent = PersonEvent<uint64_t>;
struct Click { Person person; View id; int64_t time; uint32_t campaign; };

constexpr unsigned event_buckets = 256;
constexpr uint32_t price_mask = (1u << 24) - 1;

bool compact_events(const Window& window, const Catalogs& catalogs) {
    return uint64_t(window.end) - uint64_t(window.start) <= UINT32_MAX &&
           catalogs.categories.size() <= 256;
}

struct CompactCandidate {
    using Survivor = CompactEvent;
    using Category = uint32_t;
    uint64_t id;
    int64_t ingest;
    CompactEvent payload;

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

    WideCandidate(uint64_t id, int64_t ingest, Survivor event, const Window&)
        : id(id), ingest(ingest), payload(event) {}

};

// First-touch storage for a scan worker. A skewed bucket can exceed its
// estimate without moving the records already in the mapping.
template<class Candidate> class CandidateArena {
    struct Bucket {
        size_t used = 0;
        std::vector<Candidate> overflow;
    };
    std::array<Bucket, event_buckets> buckets_;
    Candidate* rows_ = nullptr;
    size_t bytes_ = 0, capacity_ = 0;

public:
    explicit CandidateArena(size_t capacity) {
        if (!capacity) return;
        size_t bytes = capacity * event_buckets * sizeof(Candidate);
#if defined(__linux__) && defined(MADV_HUGEPAGE)
        constexpr size_t huge_page = 2 << 20;
        if (bytes >= huge_page) {
            // Linux can only promote the aligned, complete pages in this arena.
            size_t page = sysconf(_SC_PAGESIZE);
            bytes = (bytes + page - 1) / page * page;
            void* base = mmap(nullptr, bytes + huge_page, PROT_READ | PROT_WRITE,
                              MAP_PRIVATE | MAP_ANONYMOUS, -1, 0);
            if (base == MAP_FAILED) return;
            size_t prefix = (huge_page - (reinterpret_cast<uintptr_t>(base) % huge_page)) % huge_page;
            auto* address = static_cast<std::byte*>(base) + prefix;
            if (prefix) munmap(base, prefix);
            munmap(address + bytes, huge_page - prefix);
            madvise(address, bytes, MADV_HUGEPAGE);
            rows_ = reinterpret_cast<Candidate*>(address);
        } else
#endif
        {
            void* address = mmap(nullptr, bytes, PROT_READ | PROT_WRITE,
                                 MAP_PRIVATE | MAP_ANONYMOUS, -1, 0);
            if (address == MAP_FAILED) return;
            rows_ = static_cast<Candidate*>(address);
        }
        bytes_ = bytes;
        capacity_ = capacity;
    }

    CandidateArena(const CandidateArena&) = delete;
    CandidateArena& operator=(const CandidateArena&) = delete;
    ~CandidateArena() { if (rows_) munmap(rows_, bytes_); }

    void append(size_t index, uint64_t id, int64_t ingest, Event<typename Candidate::Category> event,
                const Window& window) {
        auto& bucket = buckets_[index];
        if (bucket.used < capacity_)
            std::construct_at(rows_ + index * capacity_ + bucket.used++, id, ingest, event, window);
        else
            bucket.overflow.emplace_back(id, ingest, event, window);
    }

    size_t size(size_t index) const {
        const auto& bucket = buckets_[index];
        return bucket.used + bucket.overflow.size();
    }

    std::array<std::span<const Candidate>, 2> blocks(size_t index) const {
        const auto& bucket = buckets_[index];
        const auto* first = rows_ ? rows_ + index * capacity_ : nullptr;
        return {std::span(first, bucket.used), std::span(bucket.overflow)};
    }
};

EventValue<uint32_t> unpack(uint64_t value, const Window& window) {
    return {window.start + int64_t(value >> 32), uint32_t(value) & price_mask, uint32_t(value) >> 24};
}

template<class Category> EventValue<Category> unpack(EventValue<Category> value, const Window&) {
    return value;
}

template<class Category>
Event<Category> parse_event(const Fields<7>& fields, const Window& window, const Catalogs& catalogs,
                            Ids& ids, int64_t ingest SHOPPING_TIME(, unsigned weight)) {
    SHOPPING_TIME(timing::Scope validation(timing::Validate, weight);)
    if (fields[1].empty() || fields[5].empty() || ingest == bad_time) return {};
    bool purchase = fields[4] == "purchase";
    if (!purchase && fields[4] != "view" && fields[4] != "cart") return {};
    int64_t time = window.event_time(fields[2]);
    if (time < window.start || time >= window.end || time < ingest - day_ms) return {};
    int64_t price = purchase ? decimal(fields[6], 10'000'000) : 0;
    if (purchase && price <= 0) return {};
    Person person;
    {
        SHOPPING_TIME(timing::Scope hash(timing::PersonHash, weight);)
        person = ids.key(fields[1]);
    }
    return {person, {time, static_cast<uint32_t>(price),
            purchase ? static_cast<Category>(catalogs.category(fields[5])) : 0}};
}

template<class Candidate>
std::vector<Partitions<typename Candidate::Survivor>>
read_events(const Input& input, unsigned threads, Trace& trace,
                                           const Window& window, const Catalogs& catalogs,
                                           const Input& links, Ids& ids,
                                           std::optional<People>& people) {
    Ids event_ids;
    size_t estimated_rows = 0;
    for (View chunk : input.chunks) estimated_rows += chunk.size() / 64;
    size_t capacity_per_bucket = estimated_rows / (threads * event_buckets);
    std::vector<std::unique_ptr<CandidateArena<Candidate>>> entries;
    for (unsigned i = 0; i < threads; ++i)
        entries.push_back(std::make_unique<CandidateArena<Candidate>>(capacity_per_bucket));
    parallel(threads, input.chunks.size() + 1, [&](unsigned worker, size_t task) {
        if (task == 0) {
            people.emplace(links, ids);
            return;
        }
        SHOPPING_TIME(timing::Scope scanning(timing::Scan);)
        scan<7>(input.chunks[task - 1], [&](View, const Fields<7>& fields) {
            SHOPPING_TIME(unsigned weight = timing::sample(256); timing::Scope row(timing::Rows, weight);)
            if (fields[0].empty()) return;
            uint64_t id;
            {
                SHOPPING_TIME(timing::Scope hash(timing::Hash, weight);)
                id = event_ids.key(fields[0]);
            }
            int64_t ingest;
            {
                SHOPPING_TIME(timing::Scope parse(timing::Parse, weight);)
                ingest = window.ingest_time(fields[3]);
            }
            // Keep invalid copies in the same competition for earliest ingest.
            auto event = parse_event<typename Candidate::Category>(fields, window, catalogs, ids, ingest
                                                                  SHOPPING_TIME(, weight));
            SHOPPING_TIME(timing::Scope route(timing::Route, weight);)
            entries[worker]->append(id % event_buckets, id, ingest, event, window);
        });
    });
    trace.mark("event_ingest_links_catalogs");

    std::vector<Partitions<typename Candidate::Survivor>> records(threads);
    parallel(threads, event_buckets, [&](unsigned worker, size_t bucket) {
        size_t count = 0;
        for (const auto& batch : entries) count += batch->size(bucket);
        if (!count) return;
        // Equal IDs share a slot. Collisions can only add candidates to exact deduplication.
        size_t filter_size = std::max<size_t>(64, std::bit_ceil(count * 8));
        std::vector<uint64_t> seen(filter_size / 64), repeated(filter_size / 64);
        auto filter_slot = [&](uint64_t id) { return (id / event_buckets) & (filter_size - 1); };
        size_t candidates = 0;
        auto visit = [&](auto use) {
            for (const auto& batch : entries)
                for (auto block : batch->blocks(bucket))
                    for (const Candidate& entry : block) use(entry);
        };
        visit([&](const Candidate& entry) {
            size_t i = filter_slot(entry.id);
            uint64_t bit = uint64_t{1} << (i % 64);
            bool collision = seen[i / 64] & bit;
            candidates += collision + (collision && !(repeated[i / 64] & bit));
            repeated[i / 64] |= seen[i / 64] & bit;
            seen[i / 64] |= bit;
        });
        auto emit = [&](const Candidate& entry) {
            auto event = entry.payload;
            if (!event.person) return;
            event.person = people->name(event.person);
            records[worker][event.person % partition_count].push_back(event);
        };
        size_t capacity = std::bit_ceil(candidates + candidates / 2 + 1);
        std::vector<const Candidate*> table(capacity);
        visit([&](const Candidate& entry) {
            size_t i = filter_slot(entry.id);
            if (!(repeated[i / 64] & (uint64_t{1} << (i % 64)))) {
                emit(entry);
                return;
            }
            size_t pos = (entry.id / event_buckets) & (capacity - 1);
            while (table[pos] && table[pos]->id != entry.id) pos = (pos + 1) & (capacity - 1);
            if (!table[pos] || entry.ingest < table[pos]->ingest) table[pos] = &entry;
        });
        for (auto* winner : table) if (winner) emit(*winner);
    });
    parallel(threads, entries.size(), [&](unsigned, size_t i) { entries[i].reset(); });
    trace.mark("event_dedup_route");
    return records;
}

template<class Validate>
std::vector<Partitions<Click>> read_clicks(const Input& input, unsigned threads, Trace& trace,
                                          Validate validate) {
    // Identical click copies have the same winner ordering and contribute one
    // active campaign regardless of their multiplicity.
    std::vector<Partitions<Click>> records(threads);
    parallel(threads, input.chunks.size(), [&](unsigned worker, size_t task) {
        scan<4>(input.chunks[task], [&](View, const Fields<4>& fields) {
            if (!fields[0].empty()) validate(fields, records[worker]);
        });
    });
    trace.mark("click_scan_validation");
    return records;
}

struct PersonRange { Person person; size_t begin, end, click_begin, click_end; };
struct ClickValue { View id; int64_t time; uint32_t campaign; };

template<class Value> struct Timelines {
    std::vector<PersonRange> people;
    std::unique_ptr<Value[]> values;
    std::unique_ptr<ClickValue[]> clicks;
};

template<class Index, class Value>
Timelines<Value> collect_timelines(std::vector<Partitions<PersonEvent<Value>>>& sources,
                                   std::vector<Partitions<Click>>& click_sources, size_t bucket, size_t total) {
    Timelines<Value> result;
    auto& people = result.people;
    struct Slot { Person person = 0; Index index; };
    std::vector<Slot> table(std::max<size_t>(32, std::bit_ceil(total / 4 + 1)));
    auto find = [](auto& slots, Person person) -> Slot& {
        size_t mask = slots.size() - 1;
        size_t i = (person / partition_count) & mask;
        while (slots[i].person && slots[i].person != person) i = (i + 1) & mask;
        return slots[i];
    };
    auto index = [&](Person person) {
        auto& slot = find(table, person);
        if (slot.person) return slot.index;
        if (people.size() == table.size() * 3 / 4) {
            std::vector<Slot> grown(table.size() * 2);
            for (auto entry : table) if (entry.person) find(grown, entry.person) = entry;
            table.swap(grown);
        }
        auto id = static_cast<Index>(people.size());
        find(table, person) = {person, id};
        people.push_back({person, 0, 0, 0, 0});
        return id;
    };

    auto local_ids = std::make_unique_for_overwrite<Index[]>(total);
    size_t row_index = 0;
    for (const auto& batch : sources) for (const auto& row : batch[bucket]) {
        Index id = index(row.person);
        local_ids[row_index++] = id;
        ++people[id].end;
    }
    size_t click_count = 0;
    for (const auto& batch : click_sources) click_count += batch[bucket].size();
    auto click_ids = std::make_unique_for_overwrite<Index[]>(click_count);
    constexpr Index absent = std::numeric_limits<Index>::max();
    row_index = 0;
    for (const auto& batch : click_sources) for (const auto& row : batch[bucket]) {
        auto& slot = find(table, row.person);
        click_ids[row_index++] = slot.person ? slot.index : absent;
        if (slot.person) ++people[slot.index].click_end;
    }

    size_t next = 0, next_click = 0;
    for (auto& range : people) {
        range.begin = next;
        next += range.end;
        range.end = range.begin;
        range.click_begin = next_click;
        next_click += range.click_end;
        range.click_end = range.click_begin;
    }
    result.values = std::make_unique_for_overwrite<Value[]>(total);
    row_index = 0;
    for (auto& batch : sources) {
        auto& rows = batch[bucket];
        for (const auto& row : rows)
            result.values[people[local_ids[row_index++]].end++] = row.value;
        std::vector<PersonEvent<Value>>().swap(rows);
    }
    result.clicks = std::make_unique_for_overwrite<ClickValue[]>(next_click);
    row_index = 0;
    for (auto& batch : click_sources) {
        auto& rows = batch[bucket];
        for (const auto& row : rows) {
            Index id = click_ids[row_index++];
            if (id != absent) result.clicks[people[id].click_end++] = {row.id, row.time, row.campaign};
        }
        std::vector<Click>().swap(rows);
    }
    auto* values = result.values.get();
    auto* clicks = result.clicks.get();
    for (auto range : people) {
        if constexpr (std::is_same_v<Value, uint64_t>) {
            std::sort(values + range.begin, values + range.end);
        } else {
            std::sort(values + range.begin, values + range.end,
                      [](const Value& a, const Value& b) { return a.time < b.time; });
        }
        if (range.click_end - range.click_begin > 1)
            std::sort(clicks + range.click_begin, clicks + range.click_end,
                      [](const ClickValue& a, const ClickValue& b) {
                          return a.time == b.time ? a.id > b.id : a.time < b.time;
                      });
    }
    return result;
}

template<class Category> struct UserDaily {
    Person user;
    int32_t day;
    Category category;
    int64_t sessions = 0, events = 0, revenue = 0;
};
struct CampaignDaily { int64_t sessions = 0, purchases = 0, revenue = 0, assists = 0; };

uint64_t campaign_key(int32_t day, uint32_t campaign) {
    return (uint64_t(uint32_t(day)) << 32) | campaign;
}

class CampaignTotals {
    int32_t start_;
    size_t campaigns_;
    std::vector<CampaignDaily> dense_;
    std::unordered_map<uint64_t, CampaignDaily> sparse_;

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

template<class Value, class Category>
void aggregate(const Timelines<Value>& timelines,
               const Catalogs& catalogs, const Window& window,
               CampaignTotals& totals, std::vector<UserDaily<Category>>& users) {
    constexpr bool sparse_categories = std::is_same_v<Category, size_t>;
    std::conditional_t<sparse_categories, std::unordered_map<Category, int64_t>, std::vector<int64_t>> spend;
    if constexpr (!sparse_categories) spend.resize(catalogs.categories.size());
    std::vector<Category> bought;
    std::vector<size_t> counts(catalogs.campaigns.size());
    std::vector<uint32_t> positions(counts.size()), active;
    const auto* clicks = timelines.clicks.get();
    auto at = [&](size_t event) { return unpack(timelines.values[event], window); };
    for (auto [person, event, end, begin, click] : timelines.people) {
        SHOPPING_TIME(unsigned weight = timing::sample(64);)
        size_t first = begin, last = begin;
        do {
            int32_t day = utc_day(at(event).time);
            UserDaily<Category> daily{person, day, no_category<Category>};
            auto* day_campaigns = totals.day(day);
            do {
                int64_t start = at(event).time;
                int64_t previous = start, revenue = 0;
                do {
                    const auto row = at(event++);
                    previous = row.time;
                    ++daily.events;
                    if (row.price) {
                        auto& category_spend = spend[row.category];
                        if (!category_spend) bought.push_back(row.category);
                        category_spend += row.price;
                        revenue += row.price;
                    }
                } while (event < end && uint64_t(at(event).time) - uint64_t(previous) <= gap_ms);
                ++daily.sessions;
                daily.revenue += revenue;

                SHOPPING_TIME(timing::Scope attribution(timing::Attribution, weight);)
                const int64_t lower = start - week_ms;
                while (first < last && clicks[first].time < lower) {
                    uint32_t campaign = clicks[first++].campaign;
                    if (--counts[campaign] == 0) {
                        uint32_t tail = active.back();
                        active[positions[campaign]] = tail;
                        positions[tail] = positions[campaign];
                        active.pop_back();
                    }
                }
                while (last < click && clicks[last].time <= start) {
                    uint32_t campaign = clicks[last].campaign;
                    if (clicks[last++].time < lower) {
                        ++first;
                        continue;
                    }
                    if (counts[campaign]++ == 0) {
                        positions[campaign] = active.size();
                        active.push_back(campaign);
                    }
                }
                uint32_t winner = first < last ? clicks[last - 1].campaign : 0;
                auto& credit = day_campaigns ? day_campaigns[winner] : totals[campaign_key(day, winner)];
                ++credit.sessions;
                credit.purchases += revenue != 0;
                credit.revenue += revenue;
                if (day_campaigns) {
                    for (uint32_t campaign : active) ++day_campaigns[campaign].assists;
                } else
                    for (uint32_t campaign : active) ++totals[campaign_key(day, campaign)].assists;
            } while (event < end && utc_day(at(event).time) == day);

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
            users.push_back(daily);
        } while (event < end);
        for (uint32_t campaign : active) counts[campaign] = 0;
        active.clear();
    }
}

using parquet::Column;
constexpr Column user_schema[] = {{"user_id", Column::Text, 0}, {"day", Column::Int32, 6},
    {"sessions", Column::Int64}, {"events", Column::Int64}, {"revenue_cents", Column::Int64},
    {"top_category", Column::Text, 0, true}};
constexpr Column campaign_schema[] = {{"campaign_id", Column::Text, 0}, {"channel", Column::Text, 0},
    {"day", Column::Int32, 6}, {"sessions", Column::Int64}, {"purchasing_sessions", Column::Int64},
    {"revenue_cents", Column::Int64}, {"assisted_sessions", Column::Int64}};

size_t longest_string(const std::vector<View>& strings) {
    size_t length = 0;
    for (View value : strings) length = std::max(length, value.size());
    return length;
}

template<class Category>
void write_users(parquet::File& file, std::span<const UserDaily<Category>> users, const Catalogs& catalogs,
                 const Ids& ids, size_t group_rows) {
    using parquet::little_endian;
    using parquet::plain;
    while (!users.empty()) {
        auto group = users.first(std::min(group_rows, users.size()));
        file.append(group.size(), [&](size_t column, parquet::Bytes& bytes) {
            if (column == 5) parquet::definitions(bytes, group.size(), [&](size_t i) {
                return group[i].category != no_category<Category>;
            });
            for (const auto& row : group) {
                switch (column) {
                    case 0: {
                        char buffer[9];
                        plain(bytes, ids.name(row.user, buffer));
                        break;
                    }
                    case 1: little_endian(bytes, row.day); break;
                    case 2: little_endian(bytes, row.sessions); break;
                    case 3: little_endian(bytes, row.events); break;
                    case 4: little_endian(bytes, row.revenue); break;
                    case 5:
                        if (row.category != no_category<Category>) plain(bytes, catalogs.categories[row.category]);
                        break;
                }
            }
        });
        users = users.subspan(group.size());
    }
}

void write_campaigns(const fs::path& path, const CampaignTotals& totals, const Catalogs& catalogs) {
    SHOPPING_TIME(timing::Scope encode(timing::Encode);)
    using parquet::little_endian;
    using parquet::plain;
    std::vector<std::pair<uint64_t, const CampaignDaily*>> rows;
    totals.each([&](uint64_t key, const CampaignDaily& row) { rows.emplace_back(key, &row); });
    parquet::File file(path, campaign_schema);
    size_t group_rows = parquet::row_group_limit(
        std::max(longest_string(catalogs.campaigns), longest_string(catalogs.channels)));
    for (size_t offset = 0; offset < rows.size(); offset += group_rows) {
        auto group = std::span(rows).subspan(offset, std::min(group_rows, rows.size() - offset));
        file.append(group.size(), [&](size_t column, parquet::Bytes& bytes) {
            for (auto [key, values] : group) {
                uint32_t campaign = key;
                const auto& row = *values;
                switch (column) {
                    case 0: plain(bytes, catalogs.campaigns[campaign]); break;
                    case 1: plain(bytes, catalogs.channels[campaign]); break;
                    case 2: little_endian(bytes, static_cast<int32_t>(key >> 32)); break;
                    case 3: little_endian(bytes, row.sessions); break;
                    case 4: little_endian(bytes, row.purchases); break;
                    case 5: little_endian(bytes, row.revenue); break;
                    case 6: little_endian(bytes, row.assists); break;
                }
            }
        });
    }
    file.finish();
}

std::string env(const char* key, const char* fallback) {
    const char* value = std::getenv(key);
    return value && *value ? value : fallback;
}

uint64_t positive_env(const char* key, const char* fallback) {
    auto text = env(key, fallback);
    uint64_t value = 0;
    auto parsed = std::from_chars(text.data(), text.data() + text.size(), value);
    if (parsed.ec != std::errc() || parsed.ptr != text.data() + text.size() || !value)
        throw std::runtime_error(std::string("invalid ") + key);
    return value;
}

template<class Candidate>
void process(const Window& window, const Input& links, const Catalogs& catalogs,
             const Input& event_input, const Input& click_input, const fs::path& out,
             unsigned threads, Trace& trace) {
    Ids ids;
    std::optional<People> people;
    auto events = read_events<Candidate>(event_input, threads, trace, window, catalogs, links, ids, people);
    auto clicks = read_clicks(click_input, threads, trace, [&](const Fields<4>& f, auto& partitions) {
        if (f[1].empty()) return;
        int64_t time = window.event_time(f[3]);
        if (time < window.start - week_ms || time >= window.end) return;
        auto campaign = catalogs.campaign_ids.find(f[2]);
        if (campaign == catalogs.campaign_ids.end()) return;
        Person person = people->name(ids.key(f[1]));
        partitions[person % partition_count].push_back({person, f[0], time, campaign->second});
    });

    fs::create_directories(out / "user_daily");
    fs::create_directories(out / "campaign_daily");
    size_t user_group_rows = parquet::row_group_limit(
        std::max(ids.longest(), longest_string(catalogs.categories)));
    std::vector<std::unique_ptr<parquet::File>> writers(threads);
    std::vector<CampaignTotals> campaign_totals;
    campaign_totals.reserve(threads);
    for (unsigned i = 0; i < threads; ++i) campaign_totals.emplace_back(window, catalogs.campaigns.size());
    std::array<size_t, partition_count> jobs{}, event_counts{};
    for (size_t bucket = 0; bucket < partition_count; ++bucket) {
        jobs[bucket] = bucket;
        for (const auto& batch : events) event_counts[bucket] += batch[bucket].size();
    }
    std::sort(jobs.begin(), jobs.end(),
              [&](size_t a, size_t b) { return event_counts[a] > event_counts[b]; });
    parallel(threads, partition_count, [&](unsigned worker, size_t task) {
        size_t bucket = jobs[task];
        if (!event_counts[bucket]) return;
        auto ev = [&] {
            SHOPPING_TIME(timing::Scope group(timing::Group);)
            if (event_counts[bucket] <= UINT32_MAX)
                return collect_timelines<uint32_t>(events, clicks, bucket, event_counts[bucket]);
            return collect_timelines<size_t>(events, clicks, bucket, event_counts[bucket]);
        }();
        using Category = typename Candidate::Category;
        std::vector<UserDaily<Category>> users;
        {
            SHOPPING_TIME(timing::Scope sessions(timing::Sessions);)
            aggregate(ev, catalogs, window, campaign_totals[worker], users);
        }
        SHOPPING_TIME(timing::Scope encode(timing::Encode);)
        auto& writer = writers[worker];
        if (!writer) writer = std::make_unique<parquet::File>(out / "user_daily" /
            ("part-" + std::to_string(worker) + ".parquet"), user_schema);
        write_users<Category>(*writer, users, catalogs, ids, user_group_rows);
    });
    trace.mark("sessions_user_output");

    if (std::none_of(writers.begin(), writers.end(), [](const auto& file) { return bool(file); }))
        writers[0] = std::make_unique<parquet::File>(out / "user_daily" / "part-0.parquet", user_schema);
    parallel(threads, threads + 1, [&](unsigned, size_t task) {
        if (task) {
            SHOPPING_TIME(timing::Scope encode(timing::Encode);)
            if (auto& file = writers[task - 1]) file->finish();
            return;
        }
        auto& totals = campaign_totals.front();
        for (unsigned i = 1; i < threads; ++i) totals.merge(campaign_totals[i]);
        write_campaigns(out / "campaign_daily" / "part-0.parquet", totals, catalogs);
    });
    trace.mark("finish");
    SHOPPING_TIME(timing::report();)
    // Same code in the profiling and the final build: the profile writers are
    // weak symbols that exist only when the binary is instrumented.
    if (__gcov_dump) __gcov_dump();
    if (__llvm_profile_write_file) __llvm_profile_write_file();
    _exit(0);
}

int main(int argc, char** argv) try {
    Trace trace;
    unsigned threads = std::min<uint64_t>(8, positive_env("NCPU", "8"));
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
} catch (const std::exception& error) {
    std::fprintf(stderr, "%s\n", error.what());
    return 1;
}
