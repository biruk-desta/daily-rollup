#pragma once

struct CsvMarkers { uint64_t commas, lines, quotes; };

// Index enough text to keep the separator streams and the subsequent field
// loads in cache. An extra slot at the end is a sentinel for malformed rows.
class CsvIndex {
public:
    static constexpr size_t batch_bytes = 16 * 1024;

private:
    uint16_t ends_[batch_bytes + 8], lines_[batch_bytes + 8], errors_[batch_bytes + 1];
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
    void block(unsigned base, CsvMarkers marks) {
        uint64_t sep = marks.commas | marks.lines;
        uint64_t open = marks.quotes & ((sep << 1) | after_separator_);
        uint64_t close = marks.quotes ^ open;
        uint64_t after_open = (open << 1) | after_open_;
        uint64_t after_close = (close << 1) | after_close_;
        // Borrow from each opening position through to its next separator.
        // A separator whose bit is consumed must follow a closing quote.
        uint64_t remaining = sep - open;
        uint64_t wrapped = sep & ~(remaining - in_quoted_field_);
        uint64_t bad = (after_close & ~sep) | (after_open & sep) | ((after_close ^ wrapped) & sep);
        in_quoted_field_ = (sep < open) | (remaining < in_quoted_field_);
        after_separator_ = sep >> 63;
        after_open_ = open >> 63;
        after_close_ = close >> 63;
        append(end_, base, sep);
        append(line_, base, marks.lines);
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
                Fields<N> fields;
                unsigned start = begin;
                for (size_t i = 0; i < N; ++i) {
                    unsigned quoted = data[start] == '"';
                    fields[i] = View(data + start + quoted, field[i] - start - 2 * quoted);
                    start = field[i] + 1;
                }
                row(View(data + begin, end - begin), fields);
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

template<auto Mark> [[gnu::always_inline]] inline void csv_blocks_impl(CsvIndex& index, View batch) {
    unsigned offset = 0;
    for (; offset + 64 <= batch.size(); offset += 64) index.block(offset, Mark(batch.data() + offset));
    if (offset < batch.size()) {
        alignas(64) char tail[64]{};
        std::memcpy(tail, batch.data() + offset, batch.size() - offset);
        index.block(offset, Mark(tail));
    }
}

template<auto Mark> void csv_blocks(CsvIndex& index, View batch) {
    csv_blocks_impl<Mark>(index, batch);
}

#if defined(__x86_64__)
template<> __attribute__((target("avx2"))) inline void csv_blocks<csv_avx2>(CsvIndex& index, View batch) {
    csv_blocks_impl<csv_avx2>(index, batch);
}

template<> __attribute__((target("avx512bw"))) inline void csv_blocks<csv_avx512>(CsvIndex& index, View batch) {
    csv_blocks_impl<csv_avx512>(index, batch);
}
#endif

template<size_t N, auto Mark, class Row> void scan_indexed(View chunk, Row row) {
    while (!chunk.empty()) {
        size_t size = std::min(chunk.size(), CsvIndex::batch_bytes);
        while (size && chunk[size - 1] != '\n') --size;
        if (!size) {
            size_t lf = chunk.find('\n');
            size = lf == View::npos ? chunk.size() : lf + 1;
            scan_scalar<N>(chunk.substr(0, size), row);
        } else {
            CsvIndex index;
            csv_blocks<Mark>(index, chunk.substr(0, size));
            index.parse<N>(chunk.data(), row);
        }
        chunk.remove_prefix(size);
    }
}
