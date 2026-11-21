#pragma once

#include <algorithm>
#include <bit>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <filesystem>
#include <limits>
#include <span>
#include <stdexcept>
#include <string_view>
#include <vector>

namespace parquet {

using Bytes = std::vector<uint8_t>;
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
    // Bound PLAIN string lengths, definition levels and fixed-width columns.
    size_t per_row = std::max<size_t>(8, longest_string + 5);
    return std::clamp<size_t>((target_page_bytes - 16) / per_row, 1, max_rows);
}

class File {
    static constexpr int plain_encoding = 0;
    static constexpr int rle_encoding = 3;
    struct Chunk { int64_t offset, size; };
    struct RowGroup { int64_t rows, size; std::vector<Chunk> columns; };

    std::FILE* file_;
    std::span<const Column> schema_;
    int64_t offset_ = 0;
    int64_t rows_ = 0;
    std::vector<RowGroup> groups_;
    Bytes page_;

    void write(const void* data, size_t size) {
        if (size && std::fwrite(data, 1, size, file_) != size)
            throw std::runtime_error("Parquet write failed");
        offset_ += size;
    }

    void write(const Bytes& bytes) { write(bytes.data(), bytes.size()); }

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
            column.integer(2, 0, Compact::I64); // Deprecated ColumnChunk.file_offset.
            column.field(3, Compact::Struct);
            Compact meta{bytes};
            meta.integer(1, col.physical);
            meta.list(2, Compact::I32, col.nullable ? 2 : 1);
            meta.number(plain_encoding);
            if (col.nullable) meta.number(rle_encoding);
            meta.list(3, Compact::Binary, 1);
            meta.string(col.name);
            meta.integer(4, 0); // Uncompressed.
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
        : file_(std::fopen(path.c_str(), "wb")), schema_(schema) {
        if (!file_) throw std::runtime_error("cannot open " + path.string());
        if (std::fwrite("PAR1", 1, 4, file_) != 4) {
            std::fclose(file_);
            throw std::runtime_error("Parquet write failed");
        }
        offset_ = 4;
    }
    File(const File&) = delete;
    File& operator=(const File&) = delete;
    ~File() { if (file_) std::fclose(file_); }

    template<class Encode> void append(int32_t count, Encode encode) {
        if (!count) return;
        RowGroup group{count, 0, {}};
        group.columns.reserve(schema_.size());
        Bytes header;
        header.reserve(32);
        for (size_t i = 0; i < schema_.size(); ++i) {
            page_.clear();
            encode(i, page_);
            if (page_.size() > uint32_t(std::numeric_limits<int32_t>::max()))
                throw std::runtime_error("Parquet page exceeds size limit");
            header.clear();
            Compact page{header};
            page.integer(1, 0); // Data Page V1.
            page.integer(2, page_.size());
            page.integer(3, page_.size());
            page.field(5, Compact::Struct);
            Compact data{header};
            data.integer(1, count);
            data.integer(2, plain_encoding);
            data.integer(3, rle_encoding);
            data.integer(4, rle_encoding);
            data.end();
            page.end();
            int64_t start = offset_;
            write(header);
            write(page_);
            group.columns.push_back({start, offset_ - start});
            group.size += offset_ - start;
        }
        groups_.push_back(std::move(group));
        rows_ += count;
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
        write(footer);
        write("PAR1", 4);
        int error = std::fclose(file_);
        file_ = nullptr;
        if (error) throw std::runtime_error("Parquet close failed");
    }
};

// Optional flat column: one bit per row, with a hybrid RLE/bit-packed prefix.
template<class Present> void definitions(Bytes& bytes, size_t count, Present present) {
    uint32_t groups = (count + 7) / 8;
    bytes.resize(4);
    varint(bytes, (uint64_t(groups) << 1) | 1);
    for (size_t base = 0; base < count; base += 8) {
        uint8_t bits = 0;
        for (size_t i = 0; i < std::min<size_t>(8, count - base); ++i)
            bits |= uint8_t(present(base + i)) << i;
        bytes.push_back(bits);
    }
    uint32_t size = bytes.size() - 4;
    std::memcpy(bytes.data(), &size, 4);
}

} // namespace parquet
