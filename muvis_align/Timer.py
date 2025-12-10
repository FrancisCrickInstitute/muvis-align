# https://www.geeksforgeeks.org/time-process_time-function-in-python/

import logging
import time


class Timer(object):
    def __init__(self, title, auto_unit=True, verbose=True):
        self.title = title
        self.auto_unit = auto_unit
        self.verbose = verbose

    def __enter__(self):
        self.ptime_start = time.process_time()
        self.time_start = time.time()

    def __exit__(self, type, value, traceback):
        if self.verbose:
            ptime_end = time.process_time()
            time_end = time.time()
            pelapsed = ptime_end - self.ptime_start
            elapsed = time_end - self.time_start
            unit = 'seconds'
            if self.auto_unit and elapsed >= 60:
                pelapsed /= 60
                elapsed /= 60
                unit = 'minutes'
                if elapsed >= 60:
                    pelapsed /= 60
                    elapsed /= 60
                    unit = 'hours'
            logging.info(f'Time {self.title}: {elapsed:.1f} ({pelapsed:.1f}) {unit}')
