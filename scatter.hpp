#pragma once

inline bool setting(const char* name, bool fallback) {
    const char* value = std::getenv(name);
    return value && *value ? std::string_view(value) != "0" : fallback;
}

constexpr size_t partition_size(size_t input_bytes, size_t estimated_bytes_per_row, size_t rows_per_bucket) {
    return std::bit_ceil(1 + input_bytes / estimated_bytes_per_row / rows_per_bucket);
}

inline size_t partition_override(const char* name, size_t fallback) {
    const char* text = std::getenv(name);
    if (!text || !*text) return fallback;
    size_t value = 0;
    const char* end = text + std::strlen(text);
    auto result = std::from_chars(text, end, value);
    if (result.ec != std::errc() || result.ptr != end || !std::has_single_bit(value))
        throw std::runtime_error(std::string("invalid ") + name);
    return value;
}

class ScatterArena {
    static constexpr size_t slab_bytes = 16 << 20;
    std::vector<void*> slabs_;
    std::byte* next_ = nullptr;
    size_t remaining_ = 0;
    bool huge_;

public:
    explicit ScatterArena(bool huge) : huge_(huge) {}
    ScatterArena(const ScatterArena&) = delete;
    ScatterArena& operator=(const ScatterArena&) = delete;
    ~ScatterArena() {
        for (void* base : slabs_) {
            memory::forget(base);
            munmap(base, slab_bytes);
        }
    }

    void take_mappings(std::vector<memory::Region>& regions, bool unregister) {
        regions.reserve(regions.size() + slabs_.size());
        for (void* base : slabs_) {
            regions.push_back({base, slab_bytes});
            if (unregister) memory::forget(base);
        }
        slabs_.clear();
    }

    void* block(size_t bytes) {
        if (remaining_ < bytes) {
            slabs_.push_back(nullptr);
            size_t padding = 0;
#if defined(__linux__) && defined(MADV_HUGEPAGE)
            if (huge_) padding = 2 << 20;
#endif
            void* base = mmap(nullptr, slab_bytes + padding, PROT_READ | PROT_WRITE,
                              MAP_PRIVATE | MAP_ANONYMOUS, -1, 0);
            if (base == MAP_FAILED) {
                slabs_.pop_back();
                throw std::bad_alloc();
            }
#if defined(__linux__) && defined(MADV_HUGEPAGE)
            if (padding) {
                size_t prefix = (-reinterpret_cast<uintptr_t>(base)) & (padding - 1);
                auto* aligned = static_cast<std::byte*>(base) + prefix;
                if (prefix) munmap(base, prefix);
                munmap(aligned + slab_bytes, padding - prefix);
                madvise(aligned, slab_bytes, MADV_HUGEPAGE);
                base = aligned;
            }
#endif
            try {
                memory::remember({base, slab_bytes});
            } catch (...) {
                munmap(base, slab_bytes);
                slabs_.pop_back();
                throw;
            }
            slabs_.back() = base;
            next_ = static_cast<std::byte*>(base);
            remaining_ = slab_bytes;
        }
        auto* address = next_;
        next_ += bytes;
        remaining_ -= bytes;
        return address;
    }
};

#if defined(__x86_64__)
inline bool scatter_avx_supported() {
    unsigned a, b, c, d;
    constexpr unsigned required = bit_AVX | bit_OSXSAVE;
    if (!__get_cpuid(1, &a, &b, &c, &d) || (c & required) != required) return false;
    unsigned os, high;
    asm volatile("xgetbv" : "=a"(os), "=d"(high) : "c"(0));
    return (os & 6) == 6;
}

[[gnu::target("avx"), gnu::noinline]] inline void stream_128(void* to, const void* from) {
    auto* destination = static_cast<__m256i*>(to);
    auto* source = static_cast<const __m256i*>(from);
    for (size_t i = 0; i < 4; ++i)
        _mm256_stream_si256(destination + i, _mm256_load_si256(source + i));
}
#endif

template<class T> class Scatter {
    static constexpr size_t line = 64;
    static_assert(sizeof(T) <= line && alignof(T) <= line);
    static_assert(std::is_trivially_copyable_v<T>);
    static constexpr size_t staging_rows = line / sizeof(T);
    static constexpr size_t block_lines = sizeof(T) == 16 ? 63 : sizeof(T) == 24 ? 31 : 127;
    struct alignas(line) Buffer { T rows[staging_rows]; };
    struct alignas(line) Block {
        Block* next;
        size_t used;
        Buffer lines[block_lines];
    };
    static_assert(sizeof(Block) == line * (block_lines + 1));
    struct alignas(32) Stream {
        Block* head = nullptr;
        Block* tail = nullptr;
        size_t count = 0;
        unsigned used = 0, pending = 0;
    };
    static_assert(sizeof(Stream) == 32);
    struct alignas(line) Cursors { uintptr_t next[line / sizeof(uintptr_t)]; };
    static constexpr size_t cursors_per_line = line / sizeof(uintptr_t);
    std::vector<Stream> streams_;
    std::unique_ptr<Cursors[]> cursors_;
    std::unique_ptr<void, decltype(&std::free)> buffers_;
    ScatterArena arena_;
    size_t staging_bytes_, block_rows_, staging_capacity_;
    bool streaming_, avx_;

    uintptr_t& cursor(size_t index) {
        return cursors_[index / cursors_per_line].next[index % cursors_per_line];
    }

    uintptr_t buffer(size_t index) const {
        return reinterpret_cast<uintptr_t>(buffers_.get()) + index * staging_bytes_;
    }

    [[gnu::noinline]] void advance(Stream& stream) {
        auto* block = ::new (arena_.block(sizeof(Block))) Block;
        block->next = nullptr;
        if (stream.tail) {
            stream.tail->used = block_rows_;
            stream.tail->next = block;
            stream.count += block_rows_;
        } else {
            stream.head = block;
        }
        stream.tail = block;
        stream.used = 0;
    }

    template<bool Cursor> void ensure_space(Stream& stream) {
        if (stream.used >= block_rows_ || (!Cursor && !stream.tail)) advance(stream);
    }

    template<bool Cursor> void write_line(Stream& stream, std::byte* source) {
        ensure_space<Cursor>(stream);
        auto* destination = stream.tail->lines + stream.used / staging_rows;
#if defined(__x86_64__)
        if (streaming_) {
            if constexpr (staging_rows * sizeof(T) < line)
                std::memset(source + staging_rows * sizeof(T), 0, line - staging_rows * sizeof(T));
#if defined(__AVX2__)
            auto* to = reinterpret_cast<__m256i*>(destination);
            auto* from = reinterpret_cast<const __m256i*>(source);
            _mm256_stream_si256(to, _mm256_load_si256(from));
            _mm256_stream_si256(to + 1, _mm256_load_si256(from + 1));
#else
            auto* to = reinterpret_cast<__m128i*>(destination);
            auto* from = reinterpret_cast<const __m128i*>(source);
            for (size_t i = 0; i < line / 16; ++i)
                _mm_stream_si128(to + i, _mm_load_si128(from + i));
#endif
        } else
#endif
            std::memcpy(destination->rows, source, staging_rows * sizeof(T));
        stream.used += staging_rows;
    }

    template<bool Cursor> void flush(Stream& stream, uintptr_t start) {
        if (staging_bytes_ == line) {
            write_line<Cursor>(stream, reinterpret_cast<std::byte*>(start));
            return;
        }
        ensure_space<Cursor>(stream);
        auto* destination = stream.tail->lines + stream.used / staging_rows;
        auto* source = reinterpret_cast<std::byte*>(start);
#if defined(__x86_64__)
        if (streaming_) {
            if constexpr (staging_rows * sizeof(T) < line)
                for (size_t i = 0; i < staging_bytes_; i += line)
                    std::memset(source + i + staging_rows * sizeof(T), 0, line - staging_rows * sizeof(T));
            if (staging_bytes_ == 128 && avx_) {
                stream_128(destination, source);
            } else {
                auto* to = reinterpret_cast<__m128i*>(destination);
                auto* from = reinterpret_cast<const __m128i*>(source);
                for (size_t i = 0; i < staging_bytes_ / 16; ++i)
                    _mm_stream_si128(to + i, _mm_load_si128(from + i));
            }
        } else
#endif
            for (size_t i = 0; i < staging_bytes_ / line; ++i)
                std::memcpy(destination[i].rows, source + i * line, staging_rows * sizeof(T));
        stream.used += (staging_bytes_ / line) * staging_rows;
    }

public:
    Scatter(size_t partitions, bool huge, bool streaming, size_t staging_bytes, bool avx, bool use_cursor)
        : streams_(partitions),
          cursors_(use_cursor ? std::make_unique_for_overwrite<Cursors[]>(
              (partitions + cursors_per_line - 1) / cursors_per_line) : nullptr),
          buffers_(std::aligned_alloc(staging_bytes, partitions * staging_bytes), &std::free),
          arena_(huge), staging_bytes_(staging_bytes),
          block_rows_((block_lines & ~(staging_bytes / line - 1)) * staging_rows),
          staging_capacity_(staging_bytes / line * staging_rows),
          streaming_(streaming), avx_(avx) {
        if (!buffers_) throw std::bad_alloc();
        if (cursors_)
            for (size_t i = 0; i < partitions; ++i) {
                cursor(i) = buffer(i);
                streams_[i].used = std::numeric_limits<unsigned>::max();
            }
    }

    void prefetch(size_t index) {
        if (cursors_) __builtin_prefetch(&cursor(index), 1, 3);
        auto start = buffer(index);
        for (size_t i = 0; i < staging_bytes_; i += line)
            __builtin_prefetch(reinterpret_cast<void*>(start + i), 1, 3);
    }

    void append(size_t index, const T& row) {
        if (!cursors_) {
            auto& stream = streams_[index];
            if (staging_bytes_ == line) {
                auto* start = reinterpret_cast<std::byte*>(buffer(index));
                std::memcpy(start + stream.pending++ * sizeof(T), &row, sizeof(T));
                if (stream.pending == staging_rows) {
                    write_line<false>(stream, start);
                    stream.pending = 0;
                }
            } else {
                auto start = buffer(index);
                auto pending = stream.pending++;
                auto offset = pending / staging_rows * line + pending % staging_rows * sizeof(T);
                std::memcpy(reinterpret_cast<void*>(start + offset), &row, sizeof(T));
                if (stream.pending == staging_capacity_) {
                    flush<false>(stream, start);
                    stream.pending = 0;
                }
            }
            return;
        }
        auto& write = cursor(index);
        uintptr_t next = write + sizeof(T);
        std::memcpy(reinterpret_cast<void*>(write), &row, sizeof(T));
        if constexpr (line % sizeof(T) != 0) {
            if ((next & (line - 1)) + sizeof(T) > line)
                next = (next + line - 1) & ~(line - 1);
        }
        if (!(next & (staging_bytes_ - 1))) {
            next -= staging_bytes_;
            flush<true>(streams_[index], next);
        }
        write = next;
    }

    void finish() {
        for (size_t i = 0; i < streams_.size(); ++i) {
            auto& stream = streams_[i];
            auto start = buffer(i);
            size_t bytes = cursors_ ? cursor(i) - start
                : stream.pending / staging_rows * line + stream.pending % staging_rows * sizeof(T);
            if (bytes) {
                if (!stream.tail || stream.used >= block_rows_) advance(stream);
                auto* destination = stream.tail->lines + stream.used / staging_rows;
                for (size_t offset = 0; offset < bytes; offset += line) {
                    size_t count = std::min(bytes - offset, line) / sizeof(T);
                    std::memcpy(destination->rows, reinterpret_cast<void*>(start + offset), count * sizeof(T));
                    ++destination;
                    stream.used += count;
                }
                if (cursors_) cursor(i) = start;
                else stream.pending = 0;
            }
            if (stream.tail) {
                stream.tail->used = stream.used;
                stream.count += stream.used;
            }
        }
#if defined(__x86_64__)
        if (streaming_) _mm_sfence();
#endif
    }

    size_t size(size_t index) const { return streams_[index].count; }
    void take_mappings(std::vector<memory::Region>& regions, bool unregister) {
        arena_.take_mappings(regions, unregister);
    }
    const T* first(size_t index) const {
        auto* block = streams_[index].head;
        return block ? block->lines[0].rows : nullptr;
    }

    template<class Visit> void visit(size_t index, Visit visit) const {
        for (auto* block = streams_[index].head; block; block = block->next) {
            if (block->next) __builtin_prefetch(block->next);
            for (size_t i = 0; i < block->used; ++i)
                visit(block->lines[i / staging_rows].rows[i % staging_rows]);
        }
    }
};

constexpr size_t max_staging_prefetch = 64;

template<class T> class ScatterRoute {
    struct Entry { size_t partition; T row; };
    Scatter<T>& scatter_;
    size_t depth_;
    std::array<Entry, max_staging_prefetch> pending_;
    size_t first_ = 0, next_ = 0;

    void emit() {
        const auto& entry = pending_[first_++ & (max_staging_prefetch - 1)];
        scatter_.append(entry.partition, entry.row);
    }

public:
    ScatterRoute(Scatter<T>& scatter, size_t depth) : scatter_(scatter), depth_(depth) {}

    void append(size_t partition, const T& row) {
        if (!depth_) {
            scatter_.append(partition, row);
            return;
        }
        scatter_.prefetch(partition);
        if (next_ - first_ == depth_) emit();
        pending_[next_++ & (max_staging_prefetch - 1)] = {partition, row};
    }

    void finish() { while (first_ != next_) emit(); }
};

struct ScatterSettings { bool huge, streaming; size_t staging_bytes, prefetch; bool avx, cursor; };

template<class T> using ScatterWorkers = std::vector<std::unique_ptr<Scatter<T>>>;

template<class T> ScatterWorkers<T> scatter_workers(unsigned threads, size_t partitions, const ScatterSettings& settings) {
    ScatterWorkers<T> workers;
    for (unsigned i = 0; i < threads; ++i)
        workers.push_back(std::make_unique<Scatter<T>>(partitions, settings.huge, settings.streaming,
                                                       settings.staging_bytes, settings.avx, settings.cursor));
    return workers;
}
