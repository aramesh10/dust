"""Summarize a CUDA memory snapshot from benchmark.py --memory-snapshot as text.

    python memory_report.py runs/bench/dust-1m-p4096-memsnap.pickle

Replays the allocation trace to find the peak, then lists the tensors alive at that moment grouped by the line
in this repository that allocated them. Tensors allocated before recording started are reported as one total.
"""
import argparse
from collections import defaultdict
import pickle

MIB = 1024 ** 2
ESTIMATORS = ('dust.py', 'dust_optimized.py', 'dust_noise.py', 'dust_noise_optimized.py',
              *(f'stage{n}.py' for n in range(1, 7)))
SOURCES = ESTIMATORS + ('model.py', 'benchmark.py', 'backprop.py')


def origin(frames):
    for frame in frames:
        path = frame['filename'].replace('\\', '/')
        if path.endswith(SOURCES):
            name = '/'.join(path.split('/')[-2 if '/stages/' in path else -1:])  # dust/stageN.py or noise/stageN.py
            return f"{name}:{frame['line']} {frame['name']}"
    return '<outside repo>'


def function(frames):
    for frame in frames:
        if frame['filename'].replace('\\', '/').endswith(ESTIMATORS):
            return frame['name']
    return '<outside the estimator>'


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('snapshot')
    parser.add_argument('--top', type=int, default=25)
    parser.add_argument('--device', type=int, default=0)
    args = parser.parse_args()

    with open(args.snapshot, 'rb') as stream:
        snapshot = pickle.load(stream)
    trace = snapshot['device_traces'][args.device]
    active_at_dump = sum(
        block['size'] for segment in snapshot['segments'] if segment['device'] == args.device
        for block in segment['blocks'] if block['state'] == 'active_allocated'
    )

    def replay(stop=None):
        live, untraced_freed, current = {}, 0, 0
        peak, peak_index = 0, -1
        for index, event in enumerate(trace if stop is None else trace[:stop + 1]):
            if event['action'] == 'alloc':
                live[event['addr']] = event
                current += event['size']
            elif event['action'] == 'free_completed':
                if live.pop(event['addr'], None) is None:
                    untraced_freed += event['size']
                else:
                    current -= event['size']
            if current - untraced_freed > peak:
                peak, peak_index = current - untraced_freed, index
        return live, untraced_freed, current, peak, peak_index

    _, untraced_freed, current, peak, peak_index = replay()
    baseline = active_at_dump - current + untraced_freed
    live, _, _, _, _ = replay(peak_index)

    by_line, by_function = defaultdict(lambda: [0, 0]), defaultdict(int)
    for event in live.values():
        frames = event.get('frames', [])
        entry = by_line[origin(frames)]
        entry[0] += event['size']
        entry[1] += 1
        by_function[function(frames)] += event['size']

    print(f'events recorded: {len(trace)}')
    print(f'allocated when recording started: {baseline / MIB:9.1f} MiB')
    print(f'added at the peak:                {peak / MIB:9.1f} MiB  (event {peak_index})')
    print(f'peak total:                       {(baseline + peak) / MIB:9.1f} MiB')
    print(f'\nalive at the peak, by DUST method:')
    for name, size in sorted(by_function.items(), key=lambda kv: -kv[1]):
        print(f'  {size / MIB:9.1f} MiB  {name}')
    print(f'\nalive at the peak, by allocating line (top {args.top}):')
    for line, (size, count) in sorted(by_line.items(), key=lambda kv: -kv[1][0])[:args.top]:
        print(f'  {size / MIB:9.1f} MiB  {count:5d} tensors  {line}')


if __name__ == '__main__':
    main()
