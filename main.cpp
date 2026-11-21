#include "parquet.hpp"

#include <algorithm>
#include <array>
#include <atomic>
#include <charconv>
#include <cstdlib>
#include <deque>
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

namespace fs = std::filesystem;
using View = std::string_view;
constexpr int64_t day_ms = 86'400'000;
constexpr int64_t week_ms = 7 * day_ms;
constexpr int64_t gap_ms = 1'800'000;
constexpr int64_t bad_time = INT64_MIN;
constexpr unsigned partition_count = 128;
constexpr uint32_t no_category = UINT32_MAX;

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

    void block(const char* data, uint64_t ends, uint64_t quotes) {
        while (ends) {
            unsigned bit = std::countr_zero(ends);
            const char* separator = data + (bit >> Shift);
            uint64_t before = quotes & ((uint64_t{1} << bit) - 1);
            if (before) quotes_ = std::min(3u, quotes_ + unsigned(std::popcount(before)));
            quotes ^= before;
            field(separator);
            field_ = separator + 1;
            quotes_ = 0;
            if (*separator == '\n') {
                if (!malformed_ && column_ == N) row_(View(line_, separator - line_), fields_);
                line_ = field_;
                column_ = 0;
                malformed_ = false;
            }
            ends &= ends - 1;
        }
        if (quotes) quotes_ = std::min(3u, quotes_ + unsigned(std::popcount(quotes)));
    }

    void finish(const char* end) {
        if (line_ < end) scan_scalar<N>(View(line_, end - line_), row_);
    }
};

#if defined(__x86_64__)
template<size_t N, class Row> __attribute__((target("avx2,popcnt")))
void scan_avx2(View chunk, Row row) {
    const char* p = chunk.data();
    const char* end = p + chunk.size();
    Delimited<N, 0, Row> parser(p, row);
    for (; end - p >= 32; p += 32) {
        auto data = _mm256_loadu_si256(reinterpret_cast<const __m256i*>(p));
        auto comma = _mm256_cmpeq_epi8(data, _mm256_set1_epi8(','));
        auto newline = _mm256_cmpeq_epi8(data, _mm256_set1_epi8('\n'));
        auto quotes = _mm256_cmpeq_epi8(data, _mm256_set1_epi8('"'));
        parser.block(p, uint32_t(_mm256_movemask_epi8(_mm256_or_si256(comma, newline))),
                        uint32_t(_mm256_movemask_epi8(quotes)));
    }
    parser.finish(end);
}

template<size_t N, class Row> __attribute__((target("avx512bw,popcnt")))
void scan_avx512(View chunk, Row row) {
    const char* p = chunk.data();
    const char* end = p + chunk.size();
    Delimited<N, 0, Row> parser(p, row);
    for (; end - p >= 64; p += 64) {
        auto data = _mm512_loadu_si512(p);
        uint64_t comma = _mm512_cmpeq_epi8_mask(data, _mm512_set1_epi8(','));
        uint64_t newline = _mm512_cmpeq_epi8_mask(data, _mm512_set1_epi8('\n'));
        uint64_t quotes = _mm512_cmpeq_epi8_mask(data, _mm512_set1_epi8('"'));
        parser.block(p, comma | newline, quotes);
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
    int64_t start, end;
    explicit Window(const fs::path& path) {
        Mapping file(path);
        auto fields = ManifestJson(file.bytes()).window();
        start = timestamp(fields[0]);
        end = timestamp(fields[1]);
        if (start == bad_time || end == bad_time || start >= end)
            throw std::runtime_error("invalid manifest window");
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
        std::deque<std::string> names;
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
                shard.names.emplace_back(bytes);
                position = shard.indices.emplace(shard.names.back(), number).first;
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
    std::vector<Alias> aliases_ = std::vector<Alias>(32);

    static size_t slot(const std::vector<Alias>& aliases, Person key) {
        size_t mask = aliases.size() - 1;
        size_t i = key & mask;
        while (aliases[i].key && aliases[i].key != key) i = (i + 1) & mask;
        return i;
    }

public:
    People(const Input& links, Ids& ids) {
        struct Member { View name; Person key; uint32_t parent, size; };
        std::vector<Member> members;
        auto add = [&](View name) {
            Person key = ids.key(name);
            size_t i = slot(aliases_, key);
            if (aliases_[i].key) return uint32_t(aliases_[i].value);
            if ((members.size() + 1) * 4 > aliases_.size() * 3) {
                std::vector<Alias> grown(aliases_.size() * 2);
                for (const auto& alias : aliases_)
                    if (alias.key) grown[slot(grown, alias.key)] = alias;
                aliases_.swap(grown);
                i = slot(aliases_, key);
            }
            auto index = static_cast<uint32_t>(members.size());
            aliases_[i] = {key, index};
            members.push_back({name, key, index, 1});
            return index;
        };
        auto root = [&](uint32_t i) {
            while (i != members[i].parent) {
                members[i].parent = members[members[i].parent].parent;
                i = members[i].parent;
            }
            return i;
        };
        for (View chunk : links.chunks) scan<2>(chunk, [&](View, const Fields<2>& fields) {
            if (fields[0].empty() || fields[1].empty()) return;
            uint32_t a = root(add(fields[0])), b = root(add(fields[1]));
            if (a == b) return;
            if (members[a].size < members[b].size) std::swap(a, b);
            members[b].parent = a;
            members[a].size += members[b].size;
            if (members[b].name < members[a].name) {
                members[a].name = members[b].name;
                members[a].key = members[b].key;
            }
        });
        for (auto& alias : aliases_) if (alias.key) alias.value = members[root(alias.value)].key;
    }

    Person name(Person key) const {
        auto value = aliases_[slot(aliases_, key)].value;
        return value ? value : key;
    }
};


struct Catalogs {
    StringMap<uint32_t> products, campaign_ids;
    std::vector<View> categories{"unknown"};
    std::vector<View> campaigns{"organic"}, channels{"organic"};

    Catalogs(const Input& product_input, const Input& campaign_input) {
        StringMap<uint32_t> category_ids;
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
    uint32_t category(View product) const {
        auto result = products.find(product);
        return result == products.end() ? 0 : result->second;
    }
};

struct Event { Person person; int64_t time; uint32_t price, category; };
struct Click { Person person; View id; int64_t time; uint32_t campaign; };

constexpr unsigned event_buckets = 256;
constexpr uint32_t price_mask = (1u << 24) - 1;

bool compact_events(const Window& window, const Catalogs& catalogs) {
    return uint64_t(window.end) - uint64_t(window.start) <= UINT32_MAX &&
           catalogs.categories.size() <= 256;
}

struct CompactCandidate {
    uint64_t id;
    int64_t ingest;
    Person user;
    uint32_t time, purchase;

    CompactCandidate(uint64_t id, int64_t ingest, Event event, const Window& window)
        : id(id), ingest(ingest), user(event.person),
          time(event.person ? static_cast<uint32_t>(event.time - window.start) : 0),
          purchase(event.price | (event.category << 24)) {}

    Event event(const Window& window) const {
        return {user, window.start + time, purchase & price_mask, purchase >> 24};
    }
};

struct WideCandidate {
    uint64_t id;
    int64_t ingest;
    Event payload;

    WideCandidate(uint64_t id, int64_t ingest, Event event, const Window&)
        : id(id), ingest(ingest), payload(event) {}

    Event event(const Window&) const { return payload; }
};

Event parse_event(const Fields<7>& fields, const Window& window, const Catalogs& catalogs,
                  Ids& ids, int64_t ingest) {
    if (fields[1].empty() || fields[5].empty() || ingest == bad_time) return {};
    bool purchase = fields[4] == "purchase";
    if (!purchase && fields[4] != "view" && fields[4] != "cart") return {};
    int64_t time = timestamp(fields[2]);
    if (time < window.start || time >= window.end || time < ingest - day_ms) return {};
    int64_t price = purchase ? decimal(fields[6], 10'000'000) : 0;
    if (purchase && price <= 0) return {};
    return {ids.key(fields[1]), time, static_cast<uint32_t>(price),
            purchase ? catalogs.category(fields[5]) : 0};
}

template<class Candidate>
std::vector<Partitions<Event>> read_events(const Input& input, unsigned threads, Trace& trace,
                                           const Window& window, const Catalogs& catalogs,
                                           const Input& links, Ids& ids,
                                           std::optional<People>& people) {
    Ids event_ids;
    std::vector<std::array<std::vector<Candidate>, event_buckets>> entries(threads);
    parallel(threads, input.chunks.size() + 1, [&](unsigned worker, size_t task) {
        if (task == 0) {
            people.emplace(links, ids);
            return;
        }
        scan<7>(input.chunks[task - 1], [&](View, const Fields<7>& fields) {
            if (fields[0].empty()) return;
            uint64_t id = event_ids.key(fields[0]);
            int64_t ingest = ingest_timestamp(fields[3]);
            // Keep invalid copies in the same competition for earliest ingest.
            Event event = parse_event(fields, window, catalogs, ids, ingest);
            entries[worker][id % event_buckets].emplace_back(id, ingest, event, window);
        });
    });
    trace.mark("event_ingest_links_catalogs");

    std::vector<Partitions<Event>> records(threads);
    parallel(threads, event_buckets, [&](unsigned worker, size_t bucket) {
        size_t count = 0;
        for (const auto& batch : entries) count += batch[bucket].size();
        if (!count) return;
        // Equal IDs share a slot. Collisions can only add candidates to exact deduplication.
        size_t filter_size = std::max<size_t>(64, std::bit_ceil(count * 8));
        std::vector<uint64_t> seen(filter_size / 64), repeated(filter_size / 64);
        auto filter_slot = [&](uint64_t id) { return (id / event_buckets) & (filter_size - 1); };
        size_t candidates = 0;
        for (const auto& batch : entries) for (const Candidate& entry : batch[bucket]) {
            size_t i = filter_slot(entry.id);
            uint64_t bit = uint64_t{1} << (i % 64);
            bool collision = seen[i / 64] & bit;
            candidates += collision + (collision && !(repeated[i / 64] & bit));
            repeated[i / 64] |= seen[i / 64] & bit;
            seen[i / 64] |= bit;
        }
        auto emit = [&](const Candidate& entry) {
            Event event = entry.event(window);
            if (!event.person) return;
            event.person = people->name(event.person);
            records[worker][event.person % partition_count].push_back(event);
        };
        size_t capacity = std::bit_ceil(candidates + candidates / 2 + 1);
        std::vector<const Candidate*> table(capacity);
        for (const auto& batch : entries) for (const Candidate& entry : batch[bucket]) {
            size_t i = filter_slot(entry.id);
            if (!(repeated[i / 64] & (uint64_t{1} << (i % 64)))) {
                emit(entry);
                continue;
            }
            size_t pos = (entry.id / event_buckets) & (capacity - 1);
            while (table[pos] && table[pos]->id != entry.id) pos = (pos + 1) & (capacity - 1);
            if (!table[pos] || entry.ingest < table[pos]->ingest) table[pos] = &entry;
        }
        for (auto* winner : table) if (winner) emit(*winner);
        for (auto& batch : entries) std::vector<Candidate>().swap(batch[bucket]);
    });
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

template<class T> struct Rows {
    std::unique_ptr<T[]> storage;
    size_t count;
    std::span<const T> view() const { return {storage.get(), count}; }
    bool empty() const { return !count; }
};


template<class T, class Less>
Rows<T> collect_sorted(std::vector<Partitions<T>>& sources, size_t bucket, Less less) {
    size_t total = 0;
    for (const auto& batch : sources) total += batch[bucket].size();
    constexpr unsigned radix_size = 256;
    auto digit = [](Person person) { return unsigned(person >> 56); };
    auto result = std::make_unique_for_overwrite<T[]>(std::max<size_t>(1, total));
    T* begin = result.get();
    if (total < 512) {
        T* next = begin;
        for (auto& batch : sources) {
            auto& rows = batch[bucket];
            next = std::copy(rows.begin(), rows.end(), next);
            std::vector<T>().swap(rows);
        }
        std::sort(begin, next, less);
        return {std::move(result), total};
    }
    std::array<size_t, radix_size + 1> offsets{};
    for (const auto& batch : sources)
        for (const auto& row : batch[bucket]) ++offsets[digit(row.person) + 1];
    for (size_t i = 1; i < offsets.size(); ++i) offsets[i] += offsets[i - 1];
    auto next = offsets;
    for (auto& batch : sources) {
        auto& rows = batch[bucket];
        for (const auto& row : rows) result[next[digit(row.person)]++] = row;
        std::vector<T>().swap(rows);
    }
    for (size_t i = 0; i < radix_size; ++i) {
        T* first = begin + offsets[i];
        T* last = begin + offsets[i + 1];
        std::sort(first, last, less);
    }
    return {std::move(result), total};
}

struct UserDaily {
    Person user;
    int32_t day;
    uint32_t category;
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

void aggregate(std::span<const Event> events,
               std::span<const Click> clicks, const Catalogs& catalogs,
               CampaignTotals& totals, std::vector<UserDaily>& users) {
    std::vector<int64_t> spend(catalogs.categories.size());
    std::vector<uint32_t> bought;
    std::vector<size_t> counts(catalogs.campaigns.size());
    std::vector<uint32_t> positions(counts.size()), active;
    size_t event = 0, click = 0;
    while (event < events.size()) {
        Person person = events[event].person;
        while (click < clicks.size() && clicks[click].person < person) ++click;
        size_t begin = click;
        while (click < clicks.size() && clicks[click].person == person) ++click;
        size_t first = begin, last = begin;
        do {
            int32_t day = utc_day(events[event].time);
            UserDaily daily{person, day, no_category};
            auto* day_campaigns = totals.day(day);
            do {
                int64_t start = events[event].time;
                int64_t previous = start, revenue = 0;
                do {
                    const auto& row = events[event++];
                    previous = row.time;
                    ++daily.events;
                    if (row.price) {
                        if (!spend[row.category]) bought.push_back(row.category);
                        spend[row.category] += row.price;
                        revenue += row.price;
                    }
                } while (event < events.size() && events[event].person == person &&
                         events[event].time - previous <= gap_ms);
                ++daily.sessions;
                daily.revenue += revenue;

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
            } while (event < events.size() && events[event].person == person &&
                     utc_day(events[event].time) == day);

            int64_t high = 0;
            for (uint32_t category : bought) {
                int64_t revenue = spend[category];
                if (revenue > high || (revenue == high &&
                    catalogs.categories[category] < catalogs.categories[daily.category])) {
                    daily.category = category;
                    high = revenue;
                }
                spend[category] = 0;
            }
            bought.clear();
            users.push_back(daily);
        } while (event < events.size() && events[event].person == person);
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

void write_users(parquet::File& file, std::span<const UserDaily> users, const Catalogs& catalogs,
                 const Ids& ids, size_t group_rows) {
    using parquet::little_endian;
    using parquet::plain;
    while (!users.empty()) {
        auto group = users.first(std::min(group_rows, users.size()));
        file.append(group.size(), [&](size_t column, parquet::Bytes& bytes) {
            if (column == 5) parquet::definitions(bytes, group.size(), [&](size_t i) {
                return group[i].category != no_category;
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
                    case 5: if (row.category != no_category) plain(bytes, catalogs.categories[row.category]); break;
                }
            }
        });
        users = users.subspan(group.size());
    }
}

void write_campaigns(const fs::path& path, const CampaignTotals& totals, const Catalogs& catalogs) {
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

void cap_memory(uint64_t gb) {
    constexpr uint64_t gib = uint64_t{1} << 30;
    if (gb > std::numeric_limits<rlim_t>::max() / gib)
        throw std::runtime_error("invalid MEM_GB");
#if defined(__linux__)
    // The scorer also enforces a cgroup memory limit. Bound our own mapped
    // address space so an undersized budget fails through the allocation path.
    rlimit limit{};
    if (getrlimit(RLIMIT_AS, &limit) != 0) throw std::runtime_error("cannot read memory limit");
    rlim_t budget = gb * gib;
    if (budget < limit.rlim_cur) {
        limit.rlim_cur = budget;
        if (setrlimit(RLIMIT_AS, &limit) != 0) throw std::runtime_error("cannot set MEM_GB");
    }
#endif
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
    Ids ids;
    std::optional<People> people;
    const Catalogs catalogs(products, campaigns);
    auto events = compact_events(window, catalogs)
        ? read_events<CompactCandidate>(event_input, threads, trace, window, catalogs, links, ids, people)
        : read_events<WideCandidate>(event_input, threads, trace, window, catalogs, links, ids, people);
    auto clicks = read_clicks(click_input, threads, trace, [&](const Fields<4>& f, auto& partitions) {
        if (f[1].empty()) return;
        int64_t time = timestamp(f[3]);
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
    parallel(threads, partition_count, [&](unsigned worker, size_t bucket) {
        auto ev = collect_sorted(events, bucket, [](const Event& a, const Event& b) {
            return a.person == b.person ? a.time < b.time : a.person < b.person;
        });
        auto cl = collect_sorted(clicks, bucket, [](const Click& a, const Click& b) {
            if (a.person != b.person) return a.person < b.person;
            return a.time == b.time ? a.id > b.id : a.time < b.time;
        });
        if (ev.empty()) return;
        std::vector<UserDaily> users;
        aggregate(ev.view(), cl.view(), catalogs, campaign_totals[worker], users);
        auto& writer = writers[worker];
        if (!writer) writer = std::make_unique<parquet::File>(out / "user_daily" /
            ("part-" + std::to_string(worker) + ".parquet"), user_schema);
        write_users(*writer, users, catalogs, ids, user_group_rows);
    });
    trace.mark("sessions_user_output");

    if (std::none_of(writers.begin(), writers.end(), [](const auto& file) { return bool(file); }))
        writers[0] = std::make_unique<parquet::File>(out / "user_daily" / "part-0.parquet", user_schema);
    parallel(threads, threads + 1, [&](unsigned, size_t task) {
        if (task) {
            if (auto& file = writers[task - 1]) file->finish();
            return;
        }
        auto& totals = campaign_totals.front();
        for (unsigned i = 1; i < threads; ++i) totals.merge(campaign_totals[i]);
        write_campaigns(out / "campaign_daily" / "part-0.parquet", totals, catalogs);
    });
    trace.mark("finish");
    _exit(0);
} catch (const std::exception& error) {
    std::fprintf(stderr, "%s\n", error.what());
    return 1;
}
