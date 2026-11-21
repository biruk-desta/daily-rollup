#pragma once

struct CsvMarkers { uint64_t commas, lines, quotes; };

template<size_t... Slots>
[[gnu::always_inline]] inline void csv_positions(uint16_t*& out, unsigned base, uint64_t mask,
                                                 std::index_sequence<Slots...>) {
    unsigned count = std::popcount(mask);
    auto write = [&](size_t slot) {
        out[slot] = base + __builtin_ctzll(mask | (uint64_t{1} << 63));
        mask &= mask - 1;
    };
    (write(Slots), ...);
    asm volatile("" : : "r"(out) : "memory");
    auto* excess = out + sizeof...(Slots);
    out += count;
    for (; excess < out; ++excess) {
        *excess = base + __builtin_ctzll(mask);
        mask &= mask - 1;
    }
}

struct IndexedCsvRow {
    const char* data;
    const uint16_t* ends;
    unsigned begin;

    template<size_t I> size_t field(const char*& text) const {
        unsigned start = I ? ends[I - 1] + 1 : begin;
        unsigned quoted = data[start] == '"';
        text = data + start + quoted;
        return ends[I] - start - 2 * quoted;
    }
};

template<size_t N> struct ScalarCsvRow {
    const Fields<N>& fields;

    template<size_t I> size_t field(const char*& text) const {
        text = fields[I].data();
        return fields[I].size();
    }
};

class CsvIndex {
public:
    static constexpr size_t batch_bytes = 16 * 1024;

private:
    uint16_t ends_[batch_bytes + 12], lines_[batch_bytes + 4], errors_[batch_bytes + 1];
    uint16_t* end_ = ends_;
    uint16_t* line_ = lines_;
    uint16_t* error_ = errors_;
    bool after_separator_ = true, after_open_ = false, after_close_ = false, in_quoted_field_ = false;

    static void append(uint16_t*& out, unsigned base, uint64_t mask) {
        while (mask) {
            *out++ = base + std::countr_zero(mask);
            mask &= mask - 1;
        }
    }

public:
    template<size_t Ends, size_t Lines> [[gnu::always_inline]] void block(unsigned base, CsvMarkers marks) {
        uint64_t sep = marks.commas | marks.lines;
        uint64_t open = marks.quotes & ((sep << 1) | after_separator_);
        uint64_t close = marks.quotes ^ open;
        uint64_t after_open = (open << 1) | after_open_;
        uint64_t after_close = (close << 1) | after_close_;
        uint64_t remaining = sep - open;
        uint64_t wrapped = sep & ~(remaining - in_quoted_field_);
        uint64_t bad = (after_close & ~sep) | (after_open & sep) | ((after_close ^ wrapped) & sep);
        in_quoted_field_ = (sep < open) | (remaining < in_quoted_field_);
        after_separator_ = sep >> 63;
        after_open_ = open >> 63;
        after_close_ = close >> 63;
        csv_positions(end_, base, sep, std::make_index_sequence<Ends>{});
        csv_positions(line_, base, marks.lines, std::make_index_sequence<Lines>{});
        append(error_, base, bad);
    }

    template<size_t N, class Row> void parse(const char* data, Row& row) {
        std::fill_n(end_, N, UINT16_MAX);
        *error_ = UINT16_MAX;
        auto field = ends_;
        auto error = errors_;
        unsigned begin = 0;
        for (auto line = lines_; line != line_; ++line) {
            unsigned end = *line;
            if (field[N - 1] == end && *error > end) {
                row(IndexedCsvRow{data, field, begin});
                field += N;
            } else {
                while (*field <= end) ++field;
                while (*error <= end) ++error;
            }
            begin = end + 1;
        }
    }
};

#if defined(__x86_64__)
__attribute__((target("avx2")))
inline CsvMarkers csv_avx2(const char* p) {
    auto lo = avx2_markers(p), hi = avx2_markers(p + 32);
    return {lo.commas | (hi.commas << 32), lo.newlines | (hi.newlines << 32), lo.quotes | (hi.quotes << 32)};
}

__attribute__((target("avx512bw")))
inline CsvMarkers csv_avx512(const char* p) {
    auto marks = avx512_markers(p);
    return {marks.commas, marks.newlines, marks.quotes};
}
#elif defined(__aarch64__)
inline CsvMarkers csv_neon(const char* p) {
    auto* bytes = reinterpret_cast<const uint8_t*>(p);
    uint8x16_t x0 = vld1q_u8(bytes), x1 = vld1q_u8(bytes + 16);
    uint8x16_t x2 = vld1q_u8(bytes + 32), x3 = vld1q_u8(bytes + 48);
    const uint8x16_t weights{1, 2, 4, 8, 16, 32, 64, 128, 1, 2, 4, 8, 16, 32, 64, 128};
    auto mask = [&](uint8_t marker) {
        auto target = vdupq_n_u8(marker);
        auto a = vandq_u8(vceqq_u8(x0, target), weights);
        auto b = vandq_u8(vceqq_u8(x1, target), weights);
        auto c = vandq_u8(vceqq_u8(x2, target), weights);
        auto d = vandq_u8(vceqq_u8(x3, target), weights);
        auto pairs = vpaddq_u8(vpaddq_u8(a, b), vpaddq_u8(c, d));
        return vgetq_lane_u64(vreinterpretq_u64_u8(vpaddq_u8(pairs, pairs)), 0);
    };
    return {mask(','), mask('\n'), mask('"')};
}
#endif

inline size_t csv_prefetch_distance() {
    constexpr size_t default_distance = 1024;
    static const size_t distance = [] {
        const char* setting = std::getenv("SHOPPING_CSV_PREFETCH");
        if (!setting) return default_distance;
        size_t bytes = 0;
        const char* end = setting + std::strlen(setting);
        auto [last, error] = std::from_chars(setting, end, bytes);
        return error == std::errc{} && last == end && bytes <= 16384 ? bytes : default_distance;
    }();
    return distance;
}

template<auto Mark, size_t Ends, size_t Lines>
[[gnu::always_inline]] inline void csv_blocks_impl(CsvIndex& index, View input, size_t size) {
    unsigned offset = 0;
    size_t distance = csv_prefetch_distance();
    size_t prefetch_until = distance && distance < input.size() ? input.size() - distance : 0;
    for (; offset + 64 <= size; offset += 64) {
        if (offset < prefetch_until) __builtin_prefetch(input.data() + offset + distance, 0, 3);
        index.block<Ends, Lines>(offset, Mark(input.data() + offset));
    }
    if (offset < size) {
        alignas(64) char tail[64]{};
        std::memcpy(tail, input.data() + offset, size - offset);
        index.block<Ends, Lines>(offset, Mark(tail));
    }
}

template<auto Mark, size_t Ends, size_t Lines> struct CsvBlocks {
    static void run(CsvIndex& index, View input, size_t size) {
        csv_blocks_impl<Mark, Ends, Lines>(index, input, size);
    }
};

#if defined(__x86_64__)
template<size_t Ends, size_t Lines> struct CsvBlocks<csv_avx2, Ends, Lines> {
    __attribute__((target("avx2,popcnt"))) static void run(CsvIndex& index, View input, size_t size) {
        csv_blocks_impl<csv_avx2, Ends, Lines>(index, input, size);
    }
};

template<size_t Ends, size_t Lines> struct CsvBlocks<csv_avx512, Ends, Lines> {
    __attribute__((target("avx512bw,popcnt"))) static void run(CsvIndex& index, View input, size_t size) {
        csv_blocks_impl<csv_avx512, Ends, Lines>(index, input, size);
    }
};
#endif

template<auto Mark> inline void csv_blocks(CsvIndex& index, View input, size_t size) {
    static const auto run = [] {
        const char* setting = std::getenv("SHOPPING_CSV_WRITES");
        if (setting && std::strcmp(setting, "4/1") == 0) return &CsvBlocks<Mark, 4, 1>::run;
        if (setting && std::strcmp(setting, "12/4") == 0) return &CsvBlocks<Mark, 12, 4>::run;
        if (setting && std::strcmp(setting, "0/0") == 0) return &CsvBlocks<Mark, 0, 0>::run;
        return &CsvBlocks<Mark, 8, 2>::run;
    }();
    run(index, input, size);
}

template<size_t N, auto Mark, class Row> void scan_indexed(View chunk, Row row) {
    while (!chunk.empty()) {
        size_t size = std::min(chunk.size(), CsvIndex::batch_bytes);
        while (size && chunk[size - 1] != '\n') --size;
        if (!size) {
            size_t lf = chunk.find('\n');
            size = lf == View::npos ? chunk.size() : lf + 1;
            scan_scalar<N>(chunk.substr(0, size),
                           [&](View, const Fields<N>& fields) { row(ScalarCsvRow<N>{fields}); });
        } else {
            CsvIndex index;
            {
                SHOPPING_TIME(timing::Scope markers(timing::MarkerScan);)
                csv_blocks<Mark>(index, chunk, size);
            }
            index.parse<N>(chunk.data(), row);
        }
        chunk.remove_prefix(size);
    }
}
