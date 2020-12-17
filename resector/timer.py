import time
from contextlib import contextmanager

@contextmanager
def timer(string, verbose=True):
    start = time.time()
    yield
    duration = time.time() - start
    if verbose:
        print(f'{string.capitalize()}: {duration:.3f} seconds')
