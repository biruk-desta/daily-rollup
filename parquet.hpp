#pragma once

#include "detail_trace.hpp"
#include "large_memory.hpp"

#include <algorithm>
#include <bit>
#include <cerrno>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <filesystem>
#include <limits>
#include <span>
#include <stdexcept>
#include <string_view>
#include <utility>
#include <vector>

#include <fcntl.h>
#include <sys/uio.h>
#include <unistd.h>

namespace parquet {

using Bytes = memory::Vector<uint8_t>;
using View = std::string_view;

template<class T> void little_endian(Bytes& bytes, T value) {
    static_assert(std::endian::native == std::endian::little);
    size_t offset = bytes.size();
    bytes.resize(offset + sizeof(value));
    std::memcpy(bytes.data() + offset, &value, sizeof(value));
}

inline void varint(Bytes& bytes, uint64_t n) {
    while (n >= 128) {
        bytes.push_back(static_cast<uint8_t>(n) | 128);
        n >>= 7;
    }
    bytes.push_back(static_cast<uint8_t>(n));
}

struct Compact {
    enum Type { I32 = 5, I64 = 6, Binary = 8, List = 9, Struct = 12 };
    Bytes& bytes;
    int previous = 0;

    void field(int id, Type type) {
        int delta = id - previous;
        if (delta > 0 && delta < 16) {
            bytes.push_back(static_cast<uint8_t>((delta << 4) | type));
        } else {
            bytes.push_back(type);
            number(id);
        }
        previous = id;
    }
    void number(int64_t n) { varint(bytes, (uint64_t(n) << 1) ^ uint64_t(-(n < 0))); }
    void integer(int id, int64_t n, Type type = I32) { field(id, type); number(n); }
    void string(View s) {
        varint(bytes, s.size());
        bytes.insert(bytes.end(), s.begin(), s.end());
    }
    void string(int id, View s) { field(id, Binary); string(s); }
    void list(int id, Type type, size_t count) {
        field(id, List);
        bytes.push_back(static_cast<uint8_t>((std::min<size_t>(count, 15) << 4) | type));
        if (count >= 15) varint(bytes, count);
    }
    void end() { bytes.push_back(0); }
};

struct Column {
    enum Physical { Int32 = 1, Int64 = 2, Text = 6 };
    View name;
    Physical physical;
    int converted = -1;
    bool nullable = false;
};

inline void plain(Bytes& bytes, View text) {
    if (text.size() > size_t(std::numeric_limits<int32_t>::max()) - sizeof(uint32_t))
        throw std::runtime_error("Parquet string exceeds page limit");
    little_endian(bytes, static_cast<uint32_t>(text.size()));
    if (!text.empty()) bytes.insert(bytes.end(), text.begin(), text.end());
}

inline size_t row_group_limit(size_t longest_string) {
    constexpr size_t target_page_bytes = 8 << 20;
    constexpr size_t max_rows = 32768;
    size_t per_row = std::max<size_t>(8, longest_string + 5);
    return std::clamp<size_t>((target_page_bytes - 16) / per_row, 1, max_rows);
}

struct Page { Bytes values, present; };

class File {
    static constexpr int plain_encoding = 0;
    static constexpr int rle_encoding = 3;
    struct Chunk { int64_t offset, size; };
    struct RowGroup { int64_t rows, size; std::vector<Chunk> columns; };

    int fd_;
    std::span<const Column> schema_;
    int64_t offset_ = 0;
    int64_t rows_ = 0;
    memory::Vector<RowGroup> groups_;
    std::vector<Bytes> pending_;
    size_t pending_bytes_ = 0;

    void flush() {
        if (pending_.empty()) return;
        SHOPPING_TIME(timing::Scope writing(timing::Write);)
        std::vector<iovec> segments;
        segments.reserve(pending_.size());
        for (auto& part : pending_) segments.push_back({part.data(), part.size()});
        static const size_t limit = [] {
            long maximum = sysconf(_SC_IOV_MAX);
            return size_t(maximum > 0 ? maximum : 16);
        }();
        for (size_t at = 0; at < segments.size();) {
            int count = std::min(limit, segments.size() - at);
            ssize_t written = writev(fd_, segments.data() + at, count);
            if (written < 0 && errno == EINTR) continue;
            if (written <= 0) throw std::runtime_error("Parquet write failed");
            size_t bytes = written;
            while (at < segments.size() && bytes >= segments[at].iov_len) bytes -= segments[at++].iov_len;
            if (bytes) {
                auto& remainder = segments[at];
                remainder.iov_base = static_cast<uint8_t*>(remainder.iov_base) + bytes;
                remainder.iov_len -= bytes;
            }
        }
        pending_.clear();
        pending_bytes_ = 0;
    }

    void queue(Bytes bytes) {
        if (bytes.empty()) return;
        offset_ += bytes.size();
        pending_bytes_ += bytes.size();
        pending_.push_back(std::move(bytes));
    }

    static void schema_element(Bytes& bytes, const Column& column) {
        Compact element{bytes};
        element.integer(1, column.physical);
        element.integer(3, column.nullable ? 1 : 0);
        element.string(4, column.name);
        if (column.converted >= 0) element.integer(6, column.converted);
        element.end();
    }

    void row_group(Bytes& bytes, const RowGroup& group) const {
        Compact row{bytes};
        row.list(1, Compact::Struct, schema_.size());
        for (size_t i = 0; i < schema_.size(); ++i) {
            const auto& col = schema_[i];
            const auto& chunk = group.columns[i];
            Compact column{bytes};
            column.integer(2, 0, Compact::I64);
            column.field(3, Compact::Struct);
            Compact meta{bytes};
            meta.integer(1, col.physical);
            meta.list(2, Compact::I32, col.nullable ? 2 : 1);
            meta.number(plain_encoding);
            if (col.nullable) meta.number(rle_encoding);
            meta.list(3, Compact::Binary, 1);
            meta.string(col.name);
            meta.integer(4, 0);
            meta.integer(5, group.rows, Compact::I64);
            meta.integer(6, chunk.size, Compact::I64);
            meta.integer(7, chunk.size, Compact::I64);
            meta.integer(9, chunk.offset, Compact::I64);
            meta.end();
            column.end();
        }
        row.integer(2, group.size, Compact::I64);
        row.integer(3, group.rows, Compact::I64);
        row.end();
    }

public:
    File(const std::filesystem::path& path, std::span<const Column> schema)
        : schema_(schema) {
        SHOPPING_TIME(timing::Scope writing(timing::Write);)
        queue(Bytes{'P', 'A', 'R', '1'});
        fd_ = open(path.c_str(), O_WRONLY | O_CREAT | O_TRUNC, 0666);
        if (fd_ < 0) throw std::runtime_error("cannot open " + path.string());
    }
    File(const File&) = delete;
    File& operator=(const File&) = delete;
    ~File() { if (fd_ >= 0) close(fd_); }

    void append(int32_t count, std::span<Page> columns) {
        if (!count) return;
        RowGroup group{count, 0, {}};
        group.columns.reserve(schema_.size());
        for (size_t i = 0; i < schema_.size(); ++i) {
            auto& column = columns[i];
            Bytes levels;
            if (schema_[i].nullable) {
                uint64_t run = (uint64_t(column.present.size()) << 1) | 1;
                uint32_t length = column.present.size() + (std::bit_width(run) + 6) / 7;
                little_endian(levels, length);
                varint(levels, run);
            }
            size_t page_size = column.values.size() + levels.size() + column.present.size();
            if (page_size > uint32_t(std::numeric_limits<int32_t>::max()))
                throw std::runtime_error("Parquet page exceeds size limit");
            Bytes header;
            header.reserve(32);
            Compact page{header};
            page.integer(1, 0);
            page.integer(2, page_size);
            page.integer(3, page_size);
            page.field(5, Compact::Struct);
            Compact data{header};
            data.integer(1, count);
            data.integer(2, plain_encoding);
            data.integer(3, rle_encoding);
            data.integer(4, rle_encoding);
            data.end();
            page.end();
            int64_t start = offset_;
            queue(std::move(header));
            queue(std::move(levels));
            queue(std::exchange(column.present, {}));
            queue(std::exchange(column.values, {}));
            group.columns.push_back({start, offset_ - start});
            group.size += offset_ - start;
        }
        groups_.push_back(std::move(group));
        rows_ += count;
        if (pending_bytes_ >= (128 << 20)) flush();
    }

    void finish() {
        Bytes footer;
        Compact metadata{footer};
        metadata.integer(1, 1);
        metadata.list(2, Compact::Struct, schema_.size() + 1);
        Compact root{footer};
        root.string(4, "schema");
        root.integer(5, schema_.size());
        root.end();
        for (const auto& column : schema_) schema_element(footer, column);
        metadata.integer(3, rows_, Compact::I64);
        metadata.list(4, Compact::Struct, groups_.size());
        for (const auto& group : groups_) row_group(footer, group);
        metadata.end();
        auto size = static_cast<uint32_t>(footer.size());
        little_endian(footer, size);
        footer.insert(footer.end(), {'P', 'A', 'R', '1'});
        queue(std::move(footer));
        flush();
        SHOPPING_TIME(timing::Scope closing(timing::Write);)
        int error = close(fd_);
        fd_ = -1;
        if (error) throw std::runtime_error("Parquet close failed");
    }
};

}
