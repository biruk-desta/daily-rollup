#pragma once

struct BucketLayout {
    uint32_t count;
    unsigned bits;

    static BucketLayout for_span(uint64_t span) {
        if (!span || span > (uint64_t{1} << 32)) return {};
        uint32_t count = (span - 1) / gap_ms + 1;
        return {count, unsigned(std::bit_width(count - 1))};
    }

    size_t person_limit() const { return size_t{1} << (32 - bits); }
    uint64_t key(uint32_t person, uint32_t bucket, uint32_t time) const {
        return (uint64_t(person) << (32 + bits)) | (uint64_t(bucket) << 32) | time;
    }
    uint32_t person(uint64_t key) const { return key >> (32 + bits); }
    uint32_t bucket(uint64_t key) const { return uint32_t(key >> 32) & ((uint32_t{1} << bits) - 1); }
};

template<class Category> struct SessionPurchase {
    uint64_t key;
    uint32_t price;
    Category category;
};

template<class Cursor> class SortedSessions {
    Cursor events_;
    const Window& window_;

public:
    SortedSessions(Cursor events, const Window& window) : events_(std::move(events)), window_(window) {}

    bool empty() const { return events_.empty(); }
    int64_t start() const { return unpack(events_.front(), window_).time; }

    template<class Buy> size_t take(Buy buy) {
        size_t count = 0;
        int64_t previous;
        do {
            auto row = unpack(events_.front(), window_);
            events_.pop();
            previous = row.time;
            ++count;
            if (row.price) buy(row.price, row.category);
        } while (!events_.empty() && uint64_t(start()) - uint64_t(previous) <= gap_ms);
        return count;
    }
};

template<class Category> class BucketSessions {
    struct Run { uint32_t low, high; size_t size; };
    std::span<const uint64_t> keys_;
    std::span<const SessionPurchase<Category>>& purchases_;
    int64_t window_start_;
    uint64_t person_;
    Run pending_{};

    Run run() {
        const uint64_t group = keys_[0] >> 32;
        uint32_t low = keys_[0], high = low;
        size_t count = 1;
        while (count < keys_.size() && keys_[count] >> 32 == group) {
            uint32_t time = keys_[count++];
            low = std::min(low, time);
            high = std::max(high, time);
        }
        keys_ = keys_.subspan(count);
        return {low, high, count};
    }

public:
    BucketSessions(std::span<const uint64_t> keys, std::span<const SessionPurchase<Category>>& purchases,
                   uint32_t person, int64_t window_start)
        : keys_(keys), purchases_(purchases), window_start_(window_start), person_(uint64_t(person) << 32) {
        if (!keys_.empty()) pending_ = run();
    }

    bool empty() const { return pending_.size == 0; }
    int64_t start() const { return window_start_ + pending_.low; }

    template<class Buy> size_t take(Buy buy) {
        uint32_t last = pending_.high;
        size_t count = pending_.size;
        pending_ = {};
        while (!keys_.empty()) {
            auto next = run();
            if (next.low - last > gap_ms) {
                pending_ = next;
                break;
            }
            last = next.high;
            count += next.size;
        }
        while (!purchases_.empty() && purchases_[0].key <= (person_ | last)) {
            buy(purchases_[0].price, purchases_[0].category);
            purchases_ = purchases_.subspan(1);
        }
        return count;
    }
};

template<class Index>
void bucket_scatter(const BucketLayout& layout, uint64_t* keys, uint64_t* scratch, size_t count,
                    std::span<Index> buckets, std::span<Index> people) {
    Index next = 0;
    for (auto& bucket : buckets) {
        auto size = bucket;
        bucket = next;
        next += size;
    }
    for (size_t i = 0; i < count; ++i) scratch[buckets[layout.bucket(keys[i])]++] = keys[i];
    for (size_t i = 0; i < count; ++i) keys[people[layout.person(scratch[i])]++] = scratch[i];
}
