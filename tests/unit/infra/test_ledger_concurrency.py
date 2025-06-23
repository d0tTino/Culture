import multiprocessing
import threading

import pytest

from src.infra.ledger import Ledger

pytestmark = pytest.mark.unit


def _thread_worker(db: str, count: int) -> None:
    ledger = Ledger(db)
    for _ in range(count):
        ledger.log_change("a1", 1.0, 0.0, "thread")


def _process_worker(db: str, count: int) -> None:
    ledger = Ledger(db)
    for _ in range(count):
        ledger.log_change("a1", 1.0, 0.0, "process")


def test_concurrent_updates(tmp_path) -> None:
    db = tmp_path / "ledger.sqlite"

    thread_count = 5
    process_count = 3
    ops_per_worker = 5

    threads = [
        threading.Thread(target=_thread_worker, args=(db, ops_per_worker))
        for _ in range(thread_count)
    ]

    for t in threads:
        t.start()
    for t in threads:
        t.join()

    processes = [
        multiprocessing.Process(target=_process_worker, args=(db, ops_per_worker))
        for _ in range(process_count)
    ]

    for p in processes:
        p.start()
    for p in processes:
        p.join()

    ledger = Ledger(db)
    ip, du = ledger.get_balance("a1")

    expected = (thread_count + process_count) * ops_per_worker
    assert ip == pytest.approx(expected)
    assert du == pytest.approx(0.0)
