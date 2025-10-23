import zemosaic_worker


class DummyQueue:
    def __init__(self):
        self.items = []

    def put(self, item):
        self.items.append(item)


def test_worker_inserts_callback_when_missing(monkeypatch):
    recorded = {}

    def fake_run(*args, **kwargs):
        recorded["args"] = args
        recorded["kwargs"] = kwargs

    monkeypatch.setattr(zemosaic_worker, "run_hierarchical_mosaic", fake_run)

    queue = DummyQueue()
    sentinel_args = tuple(f"arg{i}" for i in range(15))

    zemosaic_worker.run_hierarchical_mosaic_process(
        queue,
        *sentinel_args,
        solver_settings_dict={"foo": "bar"},
    )

    assert "args" in recorded
    received_args = recorded["args"]
    assert len(received_args) == len(sentinel_args) + 1
    for idx in range(10):
        assert received_args[idx] == sentinel_args[idx]
    assert callable(received_args[10])
    assert received_args[11] == sentinel_args[10]
    assert recorded["kwargs"].get("solver_settings") == {"foo": "bar"}
    assert queue.items[-1] == ("PROCESS_DONE", None, "INFO", {})
    assert not any(item[0] == "PROCESS_ERROR" for item in queue.items)


def test_worker_replaces_existing_callback(monkeypatch):
    recorded = {}

    def fake_run(*args, **kwargs):
        recorded["args"] = args
        recorded["kwargs"] = kwargs

    monkeypatch.setattr(zemosaic_worker, "run_hierarchical_mosaic", fake_run)

    queue = DummyQueue()
    original_callback = lambda *a, **k: None
    sentinel_args = tuple(f"arg{i}" for i in range(10)) + (original_callback,) + tuple(
        f"tail{i}" for i in range(5)
    )

    zemosaic_worker.run_hierarchical_mosaic_process(
        queue,
        *sentinel_args,
        solver_settings_dict=None,
    )

    assert "args" in recorded
    received_args = recorded["args"]
    assert len(received_args) == len(sentinel_args)
    for idx in range(10):
        assert received_args[idx] == sentinel_args[idx]
    assert callable(received_args[10])
    assert received_args[10] is not original_callback
    assert received_args[11] == sentinel_args[11]
    assert queue.items[-1] == ("PROCESS_DONE", None, "INFO", {})
    assert not any(item[0] == "PROCESS_ERROR" for item in queue.items)
