import numpy as np

time_conversion = {
    'days': 86400,
    'hours': 3600,
    'minutes': 60,
    'seconds': 1,
}


def extract_stats(lines, keyword):
    clock_times = []
    process_times = []
    for line in lines:
        if keyword in line:
            index = line.index(keyword) + len(keyword)
            parts = line[index:].lstrip(':').strip().split()
            clock_time = float(parts[0])
            process_time = float(parts[1].lstrip('(').rstrip(')'))
            units = parts[2]
            clock_times.append(clock_time * time_conversion[units])
            process_times.append(process_time * time_conversion[units])
    return clock_times, process_times


if __name__ == '__main__':
    filename = 'D:/slides/EM04652-02_slice17_spaghettiandmeatballs2/stitched_hpc/muvis-align-EM04652.log'
    lines = open(filename).readlines()
    for keyword in ['Time register', 'Time fuse image', 'Time save fused image']:
        clock_time, process_time = extract_stats(lines, keyword)
        print(f'{keyword}: mean {np.mean(clock_time):.0f} sum {np.sum(clock_time):.0f} process {np.sum(process_time):.0f} seconds')
