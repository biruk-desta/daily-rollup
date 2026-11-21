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

    void take_mappings(std::vector<memory::Region>& regions) {
        regions.reserve(regions.size() + slabs_.size());
        for (void* base : slabs_) {
            regions.push_back({base, slab_bytes});
            memory::forget(base);
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

template<class T> class Scatter {
    static constexpr size_t line = 64;
    static_assert(sizeof(T) <= line && alignof(T) <= line);
    static constexpr size_t staging_rows = line / sizeof(T);
    static constexpr size_t block_lines = sizeof(T) == 16 ? 63 : sizeof(T) == 24 ? 31 : 127;
    static constexpr size_t block_rows = staging_rows * block_lines;
    struct alignas(line) Buffer { T rows[staging_rows]; };
    struct alignas(line) Block {
        Block* next;
        size_t used;
        Buffer lines[block_lines];
    };
    static_assert(sizeof(Block) == line * (block_lines + 1));
    struct Stream {
        Block* head = nullptr;
        Block* tail = nullptr;
        size_t count = 0;
        unsigned used = 0, pending = 0;
    };
    std::unique_ptr<Buffer[]> buffers_;
    std::vector<Stream> streams_;
    ScatterArena arena_;
    bool streaming_;
    unsigned staging_shift_;
    size_t staging_capacity_;

    [[gnu::noinline]] void advance(Stream& stream) {
        auto* block = ::new (arena_.block(sizeof(Block))) Block;
        block->next = nullptr;
        if (stream.tail) {
            stream.tail->used = block_rows;
            stream.tail->next = block;
            stream.count += block_rows;
        } else {
            stream.head = block;
        }
        stream.tail = block;
        stream.used = 0;
    }

    void write_line(Stream& stream, Buffer& buffer) {
        if (stream.used == block_rows || !stream.tail) advance(stream);
        auto* destination = &stream.tail->lines[stream.used / staging_rows];
#if defined(__x86_64__)
        if (streaming_) {
            if constexpr (staging_rows * sizeof(T) < line)
                std::memset(reinterpret_cast<std::byte*>(&buffer) + staging_rows * sizeof(T), 0,
                            line - staging_rows * sizeof(T));
#if defined(__AVX2__)
            auto* to = reinterpret_cast<__m256i*>(destination);
            auto* from = reinterpret_cast<const __m256i*>(buffer.rows);
            _mm256_stream_si256(to, _mm256_load_si256(from));
            _mm256_stream_si256(to + 1, _mm256_load_si256(from + 1));
#else
            auto* to = reinterpret_cast<__m128i*>(destination);
            auto* from = reinterpret_cast<const __m128i*>(buffer.rows);
            for (size_t i = 0; i < line / 16; ++i)
                _mm_stream_si128(to + i, _mm_load_si128(from + i));
#endif
        } else
#endif
            std::memcpy(destination->rows, buffer.rows, staging_rows * sizeof(T));
        stream.used += staging_rows;
    }

public:
    Scatter(size_t partitions, bool huge, bool streaming, unsigned staging_lines)
        : buffers_(std::make_unique_for_overwrite<Buffer[]>(partitions * staging_lines)), streams_(partitions),
          arena_(huge), streaming_(streaming), staging_shift_(std::countr_zero(staging_lines)),
          staging_capacity_(staging_rows * staging_lines) {}

    void prefetch(size_t index) {
        auto* buffer = buffers_.get() + (index << staging_shift_);
        for (unsigned i = 0; i < (1u << staging_shift_); ++i) __builtin_prefetch(buffer + i, 1, 3);
    }

    void append(size_t index, const T& row) {
        auto& stream = streams_[index];
        if (staging_shift_ == 0) {
            auto& buffer = buffers_[index];
            buffer.rows[stream.pending++] = row;
            if (stream.pending == staging_rows) {
                write_line(stream, buffer);
                stream.pending = 0;
            }
            return;
        }
        auto* buffer = buffers_.get() + (index << staging_shift_);
        size_t pending = stream.pending++;
        buffer[pending / staging_rows].rows[pending % staging_rows] = row;
        if (stream.pending != staging_capacity_) return;
        for (unsigned i = 0; i < (1u << staging_shift_); ++i) write_line(stream, buffer[i]);
        stream.pending = 0;
    }

    void finish() {
        for (size_t i = 0; i < streams_.size(); ++i) {
            auto& stream = streams_[i];
            if (stream.pending) {
                auto* buffer = buffers_.get() + (i << staging_shift_);
                for (size_t pending = stream.pending; pending;) {
                    if (!stream.tail || stream.used == block_rows) advance(stream);
                    auto& destination = stream.tail->lines[stream.used / staging_rows];
                    size_t count = std::min(pending, staging_rows);
                    std::memcpy(destination.rows, (buffer++)->rows, count * sizeof(T));
                    stream.used += count;
                    pending -= count;
                }
                stream.pending = 0;
            }
            if (stream.tail) stream.tail->used = stream.used;
            stream.count += stream.used;
        }
#if defined(__x86_64__)
        if (streaming_) _mm_sfence();
#endif
    }

    size_t size(size_t index) const { return streams_[index].count; }
    void take_mappings(std::vector<memory::Region>& regions) { arena_.take_mappings(regions); }
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

struct ScatterSettings { bool huge, streaming; unsigned staging_lines; size_t prefetch; };

template<class T> using ScatterWorkers = std::vector<std::unique_ptr<Scatter<T>>>;

template<class T> ScatterWorkers<T> scatter_workers(unsigned threads, size_t partitions, const ScatterSettings& settings) {
    ScatterWorkers<T> workers;
    for (unsigned i = 0; i < threads; ++i)
        workers.push_back(std::make_unique<Scatter<T>>(partitions, settings.huge, settings.streaming, settings.staging_lines));
    return workers;
}
