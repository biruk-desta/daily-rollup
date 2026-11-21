#pragma once

#include <numeric>

template<class Value> uint64_t time_key(const Value& value) {
    if constexpr (std::is_same_v<Value, uint64_t>) return value >> 32;
    else return uint64_t(value.time) ^ (uint64_t{1} << 63);
}

template<class Value>
void sort_timeline(Value* first, Value* last) {
    constexpr size_t bins = 128;
    constexpr size_t sort_limit = 768;
    if (last - first < sort_limit) {
        if constexpr (std::is_same_v<Value, uint64_t>) std::sort(first, last);
        else std::sort(first, last, [](const Value& a, const Value& b) { return a.time < b.time; });
        return;
    }

    uint64_t low = time_key(*first), high = low;
    for (auto* p = first + 1; p != last; ++p) {
        auto key = time_key(*p);
        low = std::min(low, key);
        high = std::max(high, key);
    }
    if (low == high) return;
    int shift = std::max(0, int(std::bit_width(low ^ high)) - 7);
    auto digit = [shift](const Value& value) { return (time_key(value) >> shift) & (bins - 1); };
    std::array<size_t, bins + 1> boundaries{};
    for (auto* p = first; p != last; ++p) ++boundaries[digit(*p) + 1];
    std::partial_sum(boundaries.begin(), boundaries.end(), boundaries.begin());
    auto next = boundaries;
    for (size_t bin = 0; bin < bins; ++bin) {
        while (next[bin] < boundaries[bin + 1]) {
            auto i = next[bin];
            auto value = first[i];
            for (size_t dest; (dest = digit(value)) != bin;)
                std::swap(value, first[next[dest]++]);
            first[i] = value;
            ++next[bin];
        }
    }
    if (shift) for (size_t bin = 0; bin < bins; ++bin) {
        if (boundaries[bin + 1] - boundaries[bin] > 1)
            sort_timeline(first + boundaries[bin], first + boundaries[bin + 1]);
    }
}

template<class Row, class Less> class SortedRuns {
    struct Run { std::unique_ptr<Row[]> rows; size_t size = 0; };
    static constexpr size_t run_rows = (128 << 10) / sizeof(Row);
    std::vector<Run> runs_;
    Less less_;

    void sort(Run& run) {
        std::sort(run.rows.get(), run.rows.get() + run.size, less_);
    }

public:
    explicit SortedRuns(Less less) : less_(less) {}

    void append(const Row& row) {
        if (runs_.empty() || runs_.back().size == run_rows) {
            if (!runs_.empty()) sort(runs_.back());
            runs_.push_back({std::make_unique_for_overwrite<Row[]>(run_rows)});
        }
        auto& run = runs_.back();
        run.rows[run.size++] = row;
    }

    void finish() { if (!runs_.empty()) sort(runs_.back()); }
    bool empty() const { return runs_.empty(); }

    class Cursor {
        struct Span { const Row* next; const Row* last; };
        Less less_;
        std::vector<Span> heap_;
        size_t position_ = 0;

    public:
        explicit Cursor(const SortedRuns& source) : less_(source.less_) {
            heap_.reserve(source.runs_.size());
            for (const auto& run : source.runs_)
                heap_.push_back({run.rows.get(), run.rows.get() + run.size});
            std::make_heap(heap_.begin(), heap_.end(), [&](Span a, Span b) { return less_(*b.next, *a.next); });
        }

        bool empty() const { return heap_.empty(); }
        const Row& front() const { return *heap_.front().next; }
        size_t position() const { return position_; }

        void pop() {
            ++position_;
            auto run = heap_.front();
            if (++run.next == run.last) {
                run = heap_.back();
                heap_.pop_back();
                if (heap_.empty()) return;
            }
            size_t pos = 0;
            for (size_t child; (child = pos * 2 + 1) < heap_.size(); pos = child) {
                if (child + 1 < heap_.size() && less_(*heap_[child + 1].next, *heap_[child].next)) ++child;
                if (!less_(*heap_[child].next, *run.next)) break;
                heap_[pos] = heap_[child];
            }
            heap_[pos] = run;
        }
    };

    Cursor cursor() const { return Cursor(*this); }
};

template<class Row, class Less> class TimeRuns {
    static constexpr size_t count = 256;
    using Runs = SortedRuns<Row, Less>;
    std::vector<Runs> bins_;
    unsigned shift_;

public:
    TimeRuns(uint64_t span, Less less) : shift_(std::bit_width((span - 1) / count)) {
        bins_.reserve(count);
        for (size_t i = 0; i < count; ++i) bins_.emplace_back(less);
    }

    void append(uint64_t tick, const Row& row) { bins_[tick >> shift_].append(row); }
    void finish() { for (auto& bin : bins_) bin.finish(); }

    class Cursor {
        const std::vector<Runs>& bins_;
        std::optional<typename Runs::Cursor> current_;
        size_t bin_ = 0, position_ = 0;

        void next_bin() {
            current_.reset();
            while (bin_ < bins_.size()) {
                const auto& next = bins_[bin_++];
                if (!next.empty()) {
                    current_.emplace(next);
                    return;
                }
            }
        }

    public:
        explicit Cursor(const TimeRuns& source) : bins_(source.bins_) { next_bin(); }
        bool empty() const { return !current_; }
        const Row& front() const { return current_->front(); }
        size_t position() const { return position_; }
        void pop() {
            ++position_;
            current_->pop();
            if (current_->empty()) next_bin();
        }
    };

    Cursor cursor() const { return Cursor(*this); }
};

template<class T> struct SpanCursor {
    const T* values;
    size_t at, end;
    bool empty() const { return at == end; }
    const T& front() const { return values[at]; }
    size_t position() const { return at; }
    void pop() { ++at; }
};

template<class Cursor> struct PersonCursor {
    Cursor& sorted;
    Person person;
    bool empty() const { return sorted.empty() || sorted.front().person != person; }
    const auto& front() const { return sorted.front().value; }
    size_t position() const { return sorted.position(); }
    void pop() { sorted.pop(); }
};
