import time
from contextlib import contextmanager


@contextmanager
def timer(string, verbose=True):
    start = time.time()
    yield
    duration = time.time() - start
    if verbose:
        time_string = f'{int(1000 * duration)} ms'
        print(f'{time_string:>8} - {string.capitalize()}')
