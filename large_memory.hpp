#pragma once

#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <memory>
#include <mutex>
#include <new>
#include <type_traits>
#include <unordered_map>
#include <vector>

#include <sys/mman.h>
#include <unistd.h>

namespace memory {

enum class Release { None, Exit, Early };
enum class Method { Advise, Unmap };
struct Settings { Release release; Method method; };

#ifdef __linux__
inline constexpr Settings defaults{Release::Exit, Method::Advise};
#else
inline constexpr Settings defaults{Release::Exit, Method::Unmap};
#endif

inline const Settings& settings() {
    static const Settings selected = [] {
        auto result = defaults;
        if (const char* release = std::getenv("SHOPPING_RELEASE")) {
            if (std::strcmp(release, "none") == 0) result.release = Release::None;
            if (std::strcmp(release, "exit") == 0) result.release = Release::Exit;
            if (std::strcmp(release, "early") == 0) result.release = Release::Early;
        }
        if (const char* method = std::getenv("SHOPPING_RELEASE_METHOD")) {
            if (std::strcmp(method, "advise") == 0) result.method = Method::Advise;
            if (std::strcmp(method, "unmap") == 0) result.method = Method::Unmap;
        }
        return result;
    }();
    return selected;
}

struct Region { void* address; size_t bytes; };
inline std::mutex region_lock;
inline std::unordered_map<void*, size_t> live_regions;

inline void remember(Region region) {
    if (settings().release == Release::None) return;
    std::lock_guard lock(region_lock);
    live_regions.emplace(region.address, region.bytes);
}

inline void forget(void* address) {
    if (settings().release == Release::None) return;
    std::lock_guard lock(region_lock);
    live_regions.erase(address);
}

inline std::vector<Region> take_all() {
    std::lock_guard lock(region_lock);
    std::vector<Region> regions;
    regions.reserve(live_regions.size());
    for (auto [address, bytes] : live_regions) regions.push_back({address, bytes});
    live_regions.clear();
    return regions;
}

inline void release(Region region, Method method) {
    if (method == Method::Advise && madvise(region.address, region.bytes, MADV_DONTNEED) == 0) return;
    munmap(region.address, region.bytes);
}

template<class T> struct Allocator {
    using value_type = T;

    Allocator() = default;
    template<class U> Allocator(const Allocator<U>&) {}
    template<class U> bool operator==(const Allocator<U>&) const { return true; }

    static bool uses_map(size_t count) {
        return settings().release != Release::None && count >= ((1 << 20) + sizeof(T) - 1) / sizeof(T);
    }

    T* allocate(size_t count) {
        if (!uses_map(count)) return std::allocator<T>{}.allocate(count);
        if (count > size_t(-1) / sizeof(T)) throw std::bad_array_new_length();
        size_t bytes = count * sizeof(T);
        void* address = mmap(nullptr, bytes, PROT_READ | PROT_WRITE, MAP_PRIVATE | MAP_ANONYMOUS, -1, 0);
        if (address == MAP_FAILED) throw std::bad_alloc();
        try {
            remember({address, bytes});
        } catch (...) {
            munmap(address, bytes);
            throw;
        }
        return static_cast<T*>(address);
    }

    void deallocate(T* data, size_t count) {
        if (!uses_map(count)) return std::allocator<T>{}.deallocate(data, count);
        forget(data);
        munmap(data, count * sizeof(T));
    }
};

template<class T> using Vector = std::vector<T, Allocator<T>>;

class ScratchPool {
    struct FreeBlock { size_t bytes; FreeBlock* next; };
    static constexpr size_t alignment = alignof(std::max_align_t);
    static constexpr size_t quantum = std::max(alignment, sizeof(FreeBlock));
    std::vector<Region> mappings_;
    FreeBlock* free_ = nullptr;
    std::mutex mutex_;

    static uintptr_t address(const void* ptr) { return reinterpret_cast<uintptr_t>(ptr); }
    static size_t rounded(size_t bytes) { return (bytes + quantum - 1) / quantum * quantum; }

public:
    ~ScratchPool() {
        for (const auto& region : mappings_) {
            forget(region.address);
            munmap(region.address, region.bytes);
        }
    }

    void adopt(std::vector<Region> mappings) {
        mappings_ = std::move(mappings);
        std::sort(mappings_.begin(), mappings_.end(), [](const auto& a, const auto& b) {
            return address(a.address) < address(b.address);
        });
        long page = sysconf(_SC_PAGESIZE);
        if (page <= 0) return;
#ifdef __APPLE__
        std::vector<char> resident;
#else
        std::vector<unsigned char> resident;
#endif
        for (auto it = mappings_.rbegin(); it != mappings_.rend(); ++it) {
            auto* start = static_cast<std::byte*>(it->address);
            resident.resize(it->bytes / page);
            if (mincore(start, it->bytes, resident.data()) != 0) continue;
            for (size_t end = resident.size(); end;) {
                while (end && !(resident[end - 1] & 1)) --end;
                size_t begin = end;
                while (begin && (resident[begin - 1] & 1)) --begin;
                if (begin != end) free_ = ::new (start + begin * page) FreeBlock{(end - begin) * page, free_};
                end = begin;
            }
        }
    }

    void* allocate(size_t bytes, size_t align) {
        if (!bytes || align > alignment || bytes > size_t(-1) - quantum) return nullptr;
        bytes = rounded(bytes);
        std::lock_guard lock(mutex_);
        auto** link = &free_;
        while (*link && (*link)->bytes < bytes) link = &(*link)->next;
        auto* block = *link;
        if (!block) return nullptr;
        auto* next = block->next;
        if (block->bytes > bytes)
            next = ::new (reinterpret_cast<std::byte*>(block) + bytes) FreeBlock{block->bytes - bytes, next};
        *link = next;
        return block;
    }

    bool reclaim(void* ptr, size_t bytes) {
        uintptr_t start = address(ptr);
        auto owner = std::upper_bound(mappings_.begin(), mappings_.end(), start,
            [](uintptr_t at, const Region& region) { return at < address(region.address); });
        if (owner == mappings_.begin()) return false;
        --owner;
        if (start - address(owner->address) >= owner->bytes) return false;
        bytes = rounded(bytes);
        std::lock_guard lock(mutex_);
        auto** link = &free_;
        FreeBlock* previous = nullptr;
        while (*link && address(*link) < start) {
            previous = *link;
            link = &previous->next;
        }
        auto* next = *link;
        if (next && start + bytes == address(next)) {
            bytes += next->bytes;
            next = next->next;
        }
        if (previous && address(previous) + previous->bytes == start) {
            previous->bytes += bytes;
            previous->next = next;
        } else {
            *link = ::new (ptr) FreeBlock{bytes, next};
        }
        return true;
    }
};

template<class T, template<class> class Base = Allocator> struct ScratchAllocator {
    using value_type = T;
    using propagate_on_container_move_assignment = std::true_type;
    using propagate_on_container_swap = std::true_type;
    template<class U> struct rebind { using other = ScratchAllocator<U, Base>; };
    ScratchPool* pool = nullptr;

    ScratchAllocator() = default;
    explicit ScratchAllocator(ScratchPool* pool) : pool(pool) {}
    template<class U> ScratchAllocator(const ScratchAllocator<U, Base>& other) : pool(other.pool) {}
    bool operator==(const ScratchAllocator&) const = default;

    T* allocate(size_t count) {
        if (count > size_t(-1) / sizeof(T)) throw std::bad_array_new_length();
        if (pool) if (void* ptr = pool->allocate(count * sizeof(T), alignof(T))) return static_cast<T*>(ptr);
        return Base<T>{}.allocate(count);
    }

    void deallocate(T* data, size_t count) {
        if (!pool || !pool->reclaim(data, count * sizeof(T))) Base<T>{}.deallocate(data, count);
    }
};

template<class T, template<class> class Base = Allocator>
using ScratchVector = std::vector<T, ScratchAllocator<T, Base>>;

template<class K, class V, class Hash = std::hash<K>>
using Map = std::unordered_map<K, V, Hash, std::equal_to<K>, Allocator<std::pair<const K, V>>>;

template<class T, class Alloc = Allocator<T>> struct ArrayDeleter {
    size_t count = 0;
    [[no_unique_address]] Alloc allocator;
    void operator()(T* data) { allocator.deallocate(data, count); }
};

template<class T, class Alloc = Allocator<T>> using Array = std::unique_ptr<T[], ArrayDeleter<T, Alloc>>;

template<class T, class Alloc = Allocator<T>> Array<T, Alloc> array(size_t count, Alloc allocator = {}) {
    static_assert(std::is_trivially_default_constructible_v<T> && std::is_trivially_destructible_v<T>);
    auto* data = allocator.allocate(count);
    return Array<T, Alloc>(::new (data) T[count], {count, allocator});
}

template<class T> class HugeArray {
    Region region_{};

public:
    HugeArray(size_t count, bool huge) {
        static_assert(std::is_trivially_default_constructible_v<T> && std::is_trivially_destructible_v<T>);
        size_t padding = 0;
#if defined(__linux__) && defined(MADV_HUGEPAGE)
        if (huge && count >= (2u << 20) / sizeof(T)) padding = 2 << 20;
#else
        (void)huge;
#endif
        if (count > (size_t(-1) - padding) / sizeof(T)) throw std::bad_array_new_length();
        size_t bytes = count * sizeof(T);
        void* base = mmap(nullptr, bytes + padding, PROT_READ | PROT_WRITE,
                          MAP_PRIVATE | MAP_ANONYMOUS, -1, 0);
        if (base == MAP_FAILED) throw std::bad_alloc();
#if defined(__linux__) && defined(MADV_HUGEPAGE)
        if (padding) {
            size_t prefix = (-reinterpret_cast<uintptr_t>(base)) & (padding - 1);
            auto* aligned = static_cast<char*>(base) + prefix;
            if (prefix) munmap(base, prefix);
            munmap(aligned + bytes, padding - prefix);
            madvise(aligned, bytes, MADV_HUGEPAGE);
            base = aligned;
        }
#endif
        try {
            remember({base, bytes});
        } catch (...) {
            munmap(base, bytes);
            throw;
        }
        region_ = {::new (base) T[count], bytes};
    }

    HugeArray(const HugeArray&) = delete;
    HugeArray& operator=(const HugeArray&) = delete;

    ~HugeArray() {
        forget(region_.address);
        munmap(region_.address, region_.bytes);
    }

    T* data() { return static_cast<T*>(region_.address); }
    const T* data() const { return static_cast<const T*>(region_.address); }
    size_t size() const { return region_.bytes / sizeof(T); }
    T& operator[](size_t index) { return data()[index]; }
    const T& operator[](size_t index) const { return data()[index]; }
};

}
