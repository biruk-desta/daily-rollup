#include <cassert>
#include <random>

#define main shopping_main
#include "main.cpp"
#undef main

struct SessionResult {
    int64_t start;
    int32_t day;
    size_t events;
    std::array<uint64_t, 4> revenue{};
    bool operator==(const SessionResult&) const = default;
};

template<class Cursor> std::vector<SessionResult> read_sessions(Cursor sessions, int64_t start) {
    std::vector<SessionResult> result;
    while (!sessions.empty()) {
        auto time = sessions.start();
        SessionResult row{time - start, utc_day(time), 0};
        row.events = sessions.take([&](uint32_t price, uint32_t category) { row.revenue[category] += price; });
        result.push_back(row);
    }
    return result;
}

using EventRows = std::vector<CompactEvent>;

void add(EventRows& rows, Person person, uint32_t time, uint32_t price = 0, uint32_t category = 0) {
    rows.push_back({person, (uint64_t(time) << 32) | (category << 24) | price});
}

auto compare(EventRows& rows, const Window& window, std::mt19937& random) {
    std::shuffle(rows.begin(), rows.end(), random);
    const ScatterSettings scatter{setting("SHOPPING_HUGE", true), setting("SHOPPING_NT", true), 1, 0};
    auto events = scatter_workers<CompactEvent>(1, 1, scatter);
    for (const auto& row : rows) events[0]->append(0, row);
    events[0]->finish();
    auto clicks = scatter_workers<Click<uint32_t>>(1, 1, scatter);
    Timelines<uint64_t, uint32_t, uint32_t> legacy, bucketed;
    auto layout = BucketLayout::for_span(window.end - window.start);
    collect_timelines<false>(legacy, events, clicks, 0, rows.size(), 0, 0, window, layout);
    collect_timelines<true>(bucketed, events, clicks, 0, rows.size(), 0, 0, window, layout);
    assert(legacy.people.size() == bucketed.people.size());
    std::span<const SessionPurchase<uint32_t>> purchases(bucketed.purchases);
    std::unordered_map<Person, std::vector<SessionResult>> results;
    for (size_t i = 0; i < legacy.people.size(); ++i) {
        const auto& a = legacy.people[i];
        const auto& b = bucketed.people[i];
        assert(a.person == b.person && a.begin == b.begin && a.end == b.end);
        auto expected = read_sessions(SortedSessions(SpanCursor{legacy.values.get(), size_t(a.begin), size_t(a.end)},
                                                     window), window.start);
        auto actual = read_sessions(BucketSessions(std::span(bucketed.keys.get() + b.begin, b.end - b.begin),
                                                   purchases, i, window.start), window.start);
        assert(actual == expected);
        results.emplace(a.person, std::move(actual));
    }
    assert(purchases.empty());
    return results;
}

void compare_timestamps(const Window& window) {
    using namespace std::chrono;
    year_month_day day{sys_days{days{utc_day(window.start)}}};
    char date[12];
    std::snprintf(date, sizeof(date), "%04d-%02u-%02u", int(day.year()), unsigned(day.month()), unsigned(day.day()));
    const std::array<std::string, 6> samples = {
        std::string(date) + "T23:59:59Z", std::string(date) + "T00:01:02+23:59",
        std::string(date) + "T12:34:56-23:59", std::string(date) + "T00:00:00-00:00",
        "1791071890000", "0000000000001791071890000"
    };
    for (auto sample : samples) {
        assert(window.event_time(sample) == timestamp(sample));
        for (auto& byte : sample) {
            char original = byte;
            for (unsigned value = 0; value < 256; ++value) {
                byte = value;
                assert(window.event_time(sample) == timestamp(sample));
            }
            byte = original;
        }
    }
}

int main(int argc, char** argv) {
    assert(argc == 2);
    Window window(argv[1]);
    compare_timestamps(window);
    window.start = day_ms - gap_ms / 2;
    window.end = window.start + day_ms * 31 + 1;
    std::mt19937 random(20261121);
    EventRows rows;
    add(rows, 1, 0, 11, 1);
    add(rows, 1, gap_ms - 1, 19, 2);
    add(rows, 1, gap_ms - 1, 23, 2);
    add(rows, 2, gap_ms - 1, 29, 3);
    add(rows, 2, 2 * gap_ms, 31, 3);
    add(rows, 3, 1);
    add(rows, 3, gap_ms + 1, 37, 1);
    add(rows, 4, 1);
    add(rows, 4, gap_ms + 2, 41, 1);
    add(rows, 5, 0);
    add(rows, 5, gap_ms);
    add(rows, 5, 2 * gap_ms + 1);
    for (uint32_t i = 0; i < 100'000; ++i) {
        add(rows, 6, uint32_t(random() % uint64_t(window.end - window.start)), i % 11, i % 4);
        add(rows, 7, 2 * gap_ms, i % 13, i % 4);
    }
    auto got = compare(rows, window, random);
    assert((got.at(1) == std::vector<SessionResult>{{0, 0, 3, {0, 11, 42, 0}}}));
    assert((got.at(2) == std::vector<SessionResult>{{gap_ms - 1, 1, 1, {0, 0, 0, 29}},
                                                   {2 * gap_ms, 1, 1, {0, 0, 0, 31}}}));
    assert((got.at(3) == std::vector<SessionResult>{{1, 0, 2, {0, 37, 0, 0}}}));
    assert((got.at(4) == std::vector<SessionResult>{{1, 0, 1}, {gap_ms + 2, 1, 1, {0, 41, 0, 0}}}));
    assert((got.at(5) == std::vector<SessionResult>{{0, 0, 2}, {2 * gap_ms + 1, 1, 1}}));
    assert(got.at(6).size() == 1 && got.at(6)[0].events == 100'000);
    assert(got.at(7).size() == 1 && got.at(7)[0].events == 100'000);

    for (uint32_t buckets : {1, 2, 3, 17, 273, 672, 1700, 2386}) {
        window.end = window.start + gap_ms * buckets - 11;
        for (int trial = 0; trial < 10; ++trial) {
            rows.clear();
            for (int i = 0; i < 2000; ++i) {
                uint32_t time = random() % uint64_t(window.end - window.start);
                add(rows, 1 + random() % 35, time, i % 5 ? 0 : random() % 1000 + 1, random() % 4);
            }
            compare(rows, window, random);
        }
    }
    constexpr uint64_t last = UINT32_MAX;
    auto layout = BucketLayout::for_span(last + 1);
    auto person = layout.person_limit() - 1;
    auto key = layout.key(person, last / gap_ms, last);
    assert(layout.person(key) == person && layout.bucket(key) == last / gap_ms && uint32_t(key) == last);
    assert(BucketLayout::for_span(last + 2).count == 0);
}
