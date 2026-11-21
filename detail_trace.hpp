#pragma once

#if defined(SHOPPING_DETAIL_TRACE) || defined(SHOPPING_PHASE_COUNTERS)
#include <algorithm>
#include <array>
#include <cstdio>
#include <cstdlib>
#include <mutex>
#include <time.h>

#ifdef __APPLE__
#include <mach/mach_time.h>
#endif

#ifdef __linux__
#include <linux/perf_event.h>
#include <sys/syscall.h>
#include <unistd.h>
#endif

namespace timing {
enum Part { Scan, Parse, Validate, Hash, PersonHash, Route, Group, Sessions, Attribution, Encode, Write, Count };

inline bool enabled() {
    static const bool on = std::getenv("SHOPPING_DETAIL") != nullptr;
    return on;
}

struct Time {
    long long cpu = 0, wall = 0;
    Time& operator+=(Time b) { cpu += b.cpu; wall += b.wall; return *this; }
    friend Time operator-(Time a, Time b) { return {a.cpu - b.cpu, a.wall - b.wall}; }
};

inline long long nanos(clockid_t clock) {
    timespec ts{};
    clock_gettime(clock, &ts);
    return ts.tv_sec * 1'000'000'000LL + ts.tv_nsec;
}
inline Time stamp() { return {nanos(CLOCK_THREAD_CPUTIME_ID), nanos(CLOCK_MONOTONIC)}; }

inline long long fine_wall() {
#ifdef __APPLE__
    static const auto scale = [] { mach_timebase_info_data_t info; mach_timebase_info(&info); return info; }();
    return mach_absolute_time() * scale.numer / scale.denom;
#else
    return nanos(CLOCK_MONOTONIC);
#endif
}

inline long long fine_overhead() {
    static const long long cost = [] {
        std::array<long long, 128> trials{};
        for (auto& trial : trials) {
            auto start = fine_wall();
            trial = fine_wall() - start;
        }
        std::sort(trials.begin(), trials.end());
        return trials[trials.size() / 2];
    }();
    return cost;
}

inline Time overhead() {
    static const Time cost = [] {
        Time best{1'000'000, 1'000'000};
        for (int i = 0; i < 128; ++i) {
            auto start = stamp();
            auto elapsed = stamp() - start;
            best.cpu = std::min(best.cpu, elapsed.cpu);
            best.wall = std::min(best.wall, elapsed.wall);
        }
        return best;
    }();
    return cost;
}

inline std::mutex lock;
inline std::array<Time, Count> global{};

struct Local {
    std::array<Time, Count> times{};
    Local* previous;
    static inline thread_local Local* current = nullptr;
    static inline thread_local unsigned samples = 0;
    Local() : previous(current) { if (enabled()) current = this; }
    ~Local() {
        if (!enabled()) return;
        current = previous;
        std::lock_guard guard(lock);
        for (size_t i = 0; i < Count; ++i) global[i] += times[i];
    }
};

inline unsigned sample(unsigned n) {
    return Local::current && (++Local::samples % n == 1) ? n : 0;
}

class Scope {
    Part part_;
    Local* local_;
    unsigned weight_;
    Time start_;
    Time nested_cost_{};
    Scope* parent_ = nullptr;
    static inline thread_local Scope* current_ = nullptr;
public:
    explicit Scope(Part part, unsigned weight = 1)
        : part_(part), local_(weight ? Local::current : nullptr), weight_(weight) {
        if (!local_) return;
        parent_ = current_;
        current_ = this;
        start_ = weight_ > 1 ? Time{0, fine_wall()} : stamp();
    }
    ~Scope() {
        if (!local_) return;
        auto end = weight_ > 1 ? Time{0, fine_wall()} : stamp();
        auto cost = weight_ > 1 ? Time{0, fine_overhead()} : overhead();
        auto elapsed = end - start_ - cost - nested_cost_;
        if (weight_ > 1) elapsed.wall = std::clamp(elapsed.wall, 0LL, 5'000LL);
        local_->times[part_] += {elapsed.cpu * weight_, elapsed.wall * weight_};
        current_ = parent_;
        if (parent_) {
            parent_->nested_cost_ += nested_cost_;
            parent_->nested_cost_ += {2 * cost.cpu, 2 * cost.wall};
        }
    }
};

inline void report() {
    if (!enabled()) return;
    constexpr const char* names[] = {"scan", "parse", "validate", "hash", "person hash", "route",
                                     "group", "sessions", "attribution", "encode", "write"};
    auto split = [](Part parent, std::initializer_list<Part> parts) {
        long long total = 0;
        for (auto part : parts) total += global[part].wall;
        auto& all = global[parent];
        if (total > all.wall) {
            for (auto part : parts) global[part].wall = global[part].wall * all.wall / total;
        }
        Time used{};
        for (auto part : parts) {
            auto& time = global[part];
            time.cpu = all.wall > 0 ? static_cast<long long>(double(time.wall) * all.cpu / all.wall) : 0;
            used += time;
        }
        all = all - used;
    };
    global[Validate].wall = std::max(0LL, global[Validate].wall - global[PersonHash].wall);
    global[Hash] += global[PersonHash];
    split(Scan, {Parse, Validate, Hash, Route});
    split(Sessions, {Attribution});
    global[Encode] = global[Encode] - global[Write];
    for (size_t i = 0; i < Count; ++i) {
        if (i == PersonHash) continue;
        std::fprintf(stderr, "detail %s %.3f %.3f\n", names[i],
                     global[i].cpu / 1e6, global[i].wall / 1e6);
    }
}

class Hardware {
#ifdef __linux__
    std::array<int, 4> files_{-1, -1, -1, -1};
    std::array<std::array<unsigned long long, 3>, 4> previous_{};

    void close_all() {
        for (int& fd : files_) if (fd >= 0) { close(fd); fd = -1; }
    }
#endif
public:
    Hardware() {
#ifdef __linux__
        if (!std::getenv("SHOPPING_PERF")) return;
        constexpr unsigned events[]{PERF_COUNT_HW_INSTRUCTIONS, PERF_COUNT_HW_CPU_CYCLES,
                                    PERF_COUNT_HW_CACHE_MISSES, PERF_COUNT_HW_BRANCH_MISSES};
        for (size_t i = 0; i < files_.size(); ++i) {
            perf_event_attr attr{};
            attr.type = PERF_TYPE_HARDWARE;
            attr.size = sizeof(attr);
            attr.config = events[i];
            attr.inherit = 1;
            attr.exclude_kernel = 1;
            attr.exclude_hv = 1;
            attr.read_format = PERF_FORMAT_TOTAL_TIME_ENABLED | PERF_FORMAT_TOTAL_TIME_RUNNING;
            files_[i] = syscall(SYS_perf_event_open, &attr, 0, -1, -1, PERF_FLAG_FD_CLOEXEC);
            if (files_[i] >= 0) continue;
            close_all();
            break;
        }
#endif
    }
    void mark([[maybe_unused]] const char* phase) {
#ifdef __linux__
        if (files_[0] < 0) return;
        std::array<unsigned long long, 4> delta{};
        for (size_t i = 0; i < files_.size(); ++i) {
            std::array<unsigned long long, 3> value{};
            if (read(files_[i], value.data(), sizeof(value)) != sizeof(value)) { close_all(); return; }
            const auto& previous = previous_[i];
            auto enabled = value[1] - previous[1], running = value[2] - previous[2];
            auto count = value[0] - previous[0];
            if (enabled && (!running || running < enabled / 2)) { close_all(); return; }
            delta[i] = running && running < enabled ? double(count) * enabled / running : count;
            previous_[i] = value;
        }
        std::fprintf(stderr, "perf %s %llu %llu %llu %llu\n", phase,
                     delta[0], delta[1], delta[2], delta[3]);
#endif
    }
};
} // namespace timing
#endif

#ifdef SHOPPING_DETAIL_TRACE
#define SHOPPING_TIME(...) __VA_ARGS__
#else
#define SHOPPING_TIME(...)
#endif

#ifdef SHOPPING_PHASE_COUNTERS
#define SHOPPING_COUNT(...) __VA_ARGS__
#else
#define SHOPPING_COUNT(...)
#endif
