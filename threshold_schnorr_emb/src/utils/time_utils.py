"""High‑resolution timing helpers."""
from __future__ import annotations

import time
from contextlib import contextmanager
from typing import Generator


@contextmanager
def timer() -> Generator[dict, None, None]:
    """Context manager that records wall‑clock elapsed seconds.

    Usage::

        with timer() as t:
            do_work()
        print(t["elapsed"])
    """
    bag: dict = {}
    start = time.perf_counter()
    try:
        yield bag
    finally:
        bag["elapsed"] = time.perf_counter() - start