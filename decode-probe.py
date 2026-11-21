import re
import sys


events = int(sys.argv[1])
phase = 'total ingest dedup clicks sessions+output'.split()
labels = {
    '4': phase,
    'I': 'scan parse validate hash route'.split(),
    'S': 'group/sort sessions attribution encode write'.split(),
    'P': phase[1:],
    '8': phase,
}

for line in sys.stdin:
    match = re.fullmatch(r'(?:R7 )?!0[0-4] ([4ISP8]) B (.+) N (.+)', line.strip())
    if not match:
        continue
    section, shipped, candidate = match.groups()
    for engine, payload in [('shipped', shipped), ('candidate', candidate)]:
        if section == 'P':
            if payload == '-':
                continue
            print(f'{engine} [4 workers] counters/event')
            for name, group in zip(labels[section], payload.split('/'), strict=True):
                values = [int(value, 36) / 1000 for value in group.split('.')]
                print(f'  {name:18} instructions {values[0]:8.3f}  cycles {values[1]:8.3f}'
                      f'  cache misses {values[2]:7.3f}  branch misses {values[3]:7.3f}')
            continue
        count, metrics = payload.split(maxsplit=1)
        print(f'{engine} [{section}] {count} samples')
        if not int(count):
            continue
        for name, pair in zip(labels[section], metrics.split('/'), strict=True):
            cpu, wall = map(int, pair.split('.'))
            work = f'  {cpu * 1e6 / events:7.2f} CPU ns/input event' if events else ''
            print(f'  {name:18} {cpu:5} CPU-ms  {wall:5} wall-ms{work}')
