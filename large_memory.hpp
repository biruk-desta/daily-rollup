#pragma once

#include <algorithm>
#include <cstdlib>
#include <cstring>
#include <memory>
#include <mutex>
#include <new>
#include <type_traits>
#include <unordered_map>
#include <vector>

#include <sys/mman.h>

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

template<class K, class V, class Hash = std::hash<K>>
using Map = std::unordered_map<K, V, Hash, std::equal_to<K>, Allocator<std::pair<const K, V>>>;

template<class T> struct ArrayDeleter {
    size_t count = 0;
    void operator()(T* data) const { Allocator<T>{}.deallocate(data, count); }
};

template<class T> using Array = std::unique_ptr<T[], ArrayDeleter<T>>;

template<class T> Array<T> array(size_t count) {
    static_assert(std::is_trivially_default_constructible_v<T> && std::is_trivially_destructible_v<T>);
    auto* data = Allocator<T>{}.allocate(count);
    return Array<T>(::new (data) T[count], ArrayDeleter<T>{count});
}

}
