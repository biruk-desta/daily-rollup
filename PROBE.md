# Round 7 scorer probe

The probe runs the original `b15008f` engine on the official dev input. It adds exactly five `user_daily` rows in `!r7.parquet`. Their IDs start with `!00` through `!04` so they appear first in the rejection report. Other generated output lives under a hidden trial directory and is cleaned up. The diagnostic cutoff is 132 seconds, the dev engine has until second 148, and the process has a 150-second watchdog.

`probe-base.patch` adds diagnostics to `b15008f`. `probe-candidate.patch` applies `95f8084` to the same base. Both builds use the normal GCC PGO build with its plain fallback. Worker counts are four for the primary and detailed measurements and eight for comparison with Round 5. Each pair alternates in order and reports three-run medians when the budget permits. The generator requests 20M events and 5M clicks with seed `20261120` and can retry at 5M/1.25M. The nominal generated event count appears in each report row's `revenue_cents`. `sessions` and `events` are the shipped and candidate total wall-ms, respectively, at that row's worker count (four except in row 4). A negative total marks an unavailable result.

Reconstruct each payload by removing the `!0x ` prefix from `user_id` and appending `top_category` verbatim when its value differs from `-`. The same payloads appear as `R7 !0x ...` in stderr. With the event count from `revenue_cents`, `python3 decode-probe.py EVENT_COUNT < stderr.log` prints labeled measurements and normalized work. `B` is the shipped engine and `N` is the candidate. Slash-delimited groups follow these orders.

| Row | Contents |
| --- | --- |
| 0 | Four-worker sample count and CPU.wall milliseconds for total, ingest, dedup, clicks, and sessions plus output |
| 1 | Four-worker sample count and CPU.wall milliseconds for scan, ingest clock parse, payload validation, ID hashing, and candidate route |
| 2 | Four-worker sample count and CPU.wall milliseconds for group/sort, session building, attribution, encoding, and file writes |
| 3 | Ingest, dedup, clicks, sessions plus output. Each group is base-36 instructions.cycles.cache misses.branch misses per 1,000 nominal input events |
| 4 | Eight-worker sample count and CPU.wall milliseconds in the same order as row 0 |

Whole-phase CPU time is user plus system time across threads. Detailed wall time is summed worker elapsed time because subphases overlap. Fine-grained ingest work and attribution are estimates from sampled elapsed time, calibrated with empty scopes and scaled by the encompassing measured worker CPU time. The candidate also times SIMD index construction directly. Very short operations can approach the sampling overhead, so use whole-phase CPU or PMU counters to assess work savings. Link building and task setup are outside the ingest subphase figures. Encoding and file writes include final output closure. Row 3 uses inherited Linux PMU events and omits a side if a complete counter sample is unavailable. Counters cover user-space work in the four main phases and exclude exit.
