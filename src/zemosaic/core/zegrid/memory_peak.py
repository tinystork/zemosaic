"""Peak memory DECOMPOSITION: owned anonymous vs file-backed mapped pages.

ZM-ZEGRID-PEAK-ATTRIBUTION-20261113
===================================

Why this module exists
----------------------
``sweep.peak_rss_kib_with_source()`` answers "how large did this process get?"
with a TRUE peak (``ru_maxrss`` / ``peak_wset`` / ``VmHWM``). That number is
correct and stays unchanged. It is, however, a *working-set* number: on every
supported platform it also counts **file-backed mapped pages** — the external
spool tiles read through ``np.load(..., mmap_mode="r")`` and the source windows
mapped by the provider — which are not owned memory. The external host model
prices OWNED arrays. Comparing the two without decomposing them over-states the
model error (and, on a real M33 run, made an admitted 15.7 GB budget look
exceeded by a 17 340 MB working-set peak against a 13.7 GB owned-array model).

What this module measures
-------------------------
Per sample, on the best available source:

* **owned** — anonymous/private resident memory (executor arrays, allocator
  retention). Linux: ``RssAnon``. Windows: private COMMIT
  (``memory_info().private``), an UPPER BOUND of resident private bytes and
  therefore conservative. Other platforms: ``memory_full_info().uss`` when
  available, else not measured (reported, never guessed).
* **file-backed** — mapped file pages (spool tiles, mapped source windows).
  Linux: ``RssFile``. Windows: ``working_set - private_commit`` (derived, a
  bound). Other platforms: not measured unless USS is available.
* **shared / swap** — Linux ``RssShmem`` / ``VmSwap``; other platforms None.
* **total** — ``VmRSS`` (Linux) / working set (Windows). The TRUE total peak is
  additionally taken from the platform high-water mark (``VmHWM`` /
  ``peak_wset``) exactly like ``peak_rss_kib_with_source``.

The sampler keeps the peaks of each component AND the decomposition of the
instant of the total peak. The two are deliberately NOT added together: maxima
of different components can occur at different instants, so ``max(owned) +
max(file)`` is not a working set and must never be published as one.

Everything here is additive telemetry. Nothing in this module changes a model
constant, an admission ceiling, a class/batch policy, science, geometry or
payload.
"""

from __future__ import annotations

import os
import sys
import threading
import time

try:
    import psutil
except ImportError:  # pragma: no cover - psutil is a product dependency
    psutil = None

__all__ = [
    "MEMORY_COMPONENTS_SCHEMA",
    "DEFAULT_SAMPLE_INTERVAL_S",
    "platform_capability",
    "sample_memory_components",
    "PeakMemorySampler",
]

MEMORY_COMPONENTS_SCHEMA = "zemosaic-memory-peak-components-v1"
DEFAULT_SAMPLE_INTERVAL_S = 0.1

_LINUX_STATUS_FIELDS = (
    "VmRSS", "VmHWM", "RssAnon", "RssFile", "RssShmem", "VmSwap", "VmSize",
)


def _read_linux_status() -> dict:
    """Return the kB fields we care about from ``/proc/self/status`` (bytes)."""
    out: dict = {}
    try:
        with open("/proc/self/status", "r", encoding="ascii") as fh:
            for line in fh:
                name, sep, rest = line.partition(":")
                if not sep or name not in _LINUX_STATUS_FIELDS:
                    continue
                try:
                    out[name] = int(rest.split()[0]) * 1024
                except (ValueError, IndexError):
                    continue
    except OSError:
        return {}
    return out


def platform_capability() -> dict:
    """State exactly which components this platform can measure.

    ``missing_instruments`` is the honest list of what a reader must NOT assume
    was measured. It is empty only on Linux (full anonymous/file split).
    """
    plat = sys.platform
    caps = {
        "platform": plat,
        "schema": MEMORY_COMPONENTS_SCHEMA,
        "owned_measured": False,
        "file_backed_measured": False,
        "shared_measured": False,
        "swap_measured": False,
        "owned_source": None,
        "file_backed_source": None,
        "total_source": None,
        "true_total_peak_source": None,
        "owned_is_upper_bound": False,
        "notes": [],
        "missing_instruments": [],
    }
    if plat.startswith("linux"):
        caps.update({
            "owned_measured": True,
            "file_backed_measured": True,
            "shared_measured": True,
            "swap_measured": True,
            "owned_source": "proc_status_RssAnon",
            "file_backed_source": "proc_status_RssFile",
            "total_source": "proc_status_VmRSS",
            "true_total_peak_source": "proc_status_VmHWM",
            "notes": [
                "RssAnon is resident anonymous memory (owned arrays + allocator "
                "retention); RssFile is mapped file pages (spool tiles, mapped "
                "source windows); RssShmem is shared memory.",
            ],
        })
    elif plat.startswith("win"):
        caps.update({
            "owned_measured": psutil is not None,
            "file_backed_measured": psutil is not None,
            "owned_source": "psutil_private_commit",
            "file_backed_source": "derived_working_set_minus_private_commit",
            "total_source": "psutil_working_set_rss",
            "true_total_peak_source": "psutil_peak_wset",
            "owned_is_upper_bound": True,
            "notes": [
                "Private COMMIT is cheap to sample and is an upper bound of "
                "resident private bytes; it is reported as owned (conservative). "
                "File-backed = working_set - private_commit, a bound that also "
                "absorbs shareable non-file pages.",
                "Private WORKING SET (USS) would be exact but is not sampled: "
                "walking the working set of a multi-GB process every interval "
                "would perturb the reduction.",
            ],
        })
        caps["missing_instruments"] = [
            "exact_private_working_set_uss_peak",
        ]
    elif plat == "darwin" and psutil is not None:
        caps.update({
            "owned_measured": True,
            "file_backed_measured": False,
            "owned_source": "psutil_uss",
            "total_source": "psutil_rss",
            "true_total_peak_source": None,
            "notes": [
                "USS is exact private resident memory but expensive; file-backed "
                "pages are not separable with psutil on macOS.",
            ],
        })
        caps["missing_instruments"] = [
            "file_backed_mapped_pages",
            "true_total_peak_high_water_mark",
        ]
    else:
        caps["notes"] = ["no per-component memory API on this platform"]
        caps["missing_instruments"] = [
            "owned_anonymous_peak",
            "file_backed_mapped_peak",
            "true_total_peak_high_water_mark",
        ]
    return caps


def sample_memory_components() -> dict:
    """Take ONE component sample (bytes). Unmeasured components are ``None``.

    Never substitutes a different quantity for a missing one: a component that
    cannot be measured is ``None`` and is named in ``platform_capability()``.
    """
    caps = platform_capability()
    plat = caps["platform"]
    sample = {
        "total_bytes": None,
        "owned_bytes": None,
        "file_backed_bytes": None,
        "shared_bytes": None,
        "swap_bytes": None,
        "sources": {
            "owned": caps["owned_source"],
            "file_backed": caps["file_backed_source"],
            "total": caps["total_source"],
        },
    }
    if plat.startswith("linux"):
        st = _read_linux_status()
        if not st:
            return sample
        sample["total_bytes"] = st.get("VmRSS")
        sample["owned_bytes"] = st.get("RssAnon")
        sample["file_backed_bytes"] = st.get("RssFile")
        sample["shared_bytes"] = st.get("RssShmem")
        sample["swap_bytes"] = st.get("VmSwap")
        return sample
    if psutil is None:
        return sample
    try:
        proc = psutil.Process()
        info = proc.memory_info()
    except Exception:  # pragma: no cover - defensive
        return sample
    total = int(getattr(info, "rss", 0)) or None
    sample["total_bytes"] = total
    if plat.startswith("win"):
        private = getattr(info, "private", None)
        if private is not None:
            sample["owned_bytes"] = int(private)
            if total is not None:
                sample["file_backed_bytes"] = max(0, int(total) - int(private))
        return sample
    if plat == "darwin":
        try:
            full = proc.memory_full_info()
            uss = getattr(full, "uss", None)
            if uss is not None:
                sample["owned_bytes"] = int(uss)
        except Exception:  # pragma: no cover - defensive
            pass
    return sample


class PeakMemorySampler:
    """Background component sampler with phase bucketing.

    Usage::

        sampler = PeakMemorySampler()
        sampler.start()
        try:
            run(..., memory_probe=sampler.mark)
        finally:
            sampler.stop()
        payload = sampler.result()

    ``mark(label)`` is a phase transition: it records a sample at the boundary
    and attributes subsequent samples to ``label``. Sampling is passive (reads
    only), so it cannot change the reduction.
    """

    def __init__(self, interval_s: float = DEFAULT_SAMPLE_INTERVAL_S) -> None:
        try:
            interval = float(interval_s)
        except (TypeError, ValueError):
            interval = DEFAULT_SAMPLE_INTERVAL_S
        self.interval_s = max(0.005, interval)
        self._capability = platform_capability()
        self._lock = threading.Lock()
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None
        self._started_at = 0.0
        self._phase = "idle"
        self._phase_started = 0.0
        self._samples: list[dict] = []
        # The platform high-water mark is read BEFORE sampling and AFTER stop so
        # the caller can decide whether to reset it (tests do; product does not).
        self._hwm_before = self._read_true_total_peak()

    # -- platform high-water mark -------------------------------------------
    def _read_true_total_peak(self) -> int | None:
        plat = self._capability["platform"]
        if plat.startswith("linux"):
            st = _read_linux_status()
            return st.get("VmHWM")
        if plat.startswith("win") and psutil is not None:
            try:
                peak = getattr(psutil.Process().memory_info(), "peak_wset", None)
                return None if peak is None else int(peak)
            except Exception:  # pragma: no cover - defensive
                return None
        return None

    # -- lifecycle ----------------------------------------------------------
    def start(self) -> None:
        if self._thread is not None:
            return
        self._started_at = time.perf_counter()
        self._phase_started = self._started_at
        self._stop.clear()
        self._thread = threading.Thread(
            target=self._run, name="memory-peak-sampler", daemon=True
        )
        self._thread.start()

    def stop(self) -> None:
        self._stop.set()
        thread = self._thread
        if thread is not None:
            thread.join(timeout=5.0)
        self._thread = None
        # A final sample closes the window opened by the last mark().
        self._record()

    def _record(self) -> None:
        now = time.perf_counter()
        sample = sample_memory_components()
        sample["t_s"] = round(now - self._started_at, 6)
        sample["phase"] = self._phase
        with self._lock:
            self._samples.append(sample)

    def _run(self) -> None:
        while not self._stop.is_set():
            self._record()
            self._stop.wait(self.interval_s)

    # -- phases -------------------------------------------------------------
    def mark(self, label) -> None:
        """Record a phase boundary; subsequent samples belong to ``label``."""
        now = time.perf_counter()
        self._record()
        with self._lock:
            self._phase = str(label)
        self._phase_started = now

    # -- results ------------------------------------------------------------
    def _peak(self, key: str) -> int | None:
        values = [
            s[key] for s in self._samples
            if s.get(key) is not None
        ]
        return max(values) if values else None

    def result(self) -> dict:
        samples = list(self._samples)
        hwm_after = self._read_true_total_peak()
        hwm = None
        for value in (self._hwm_before, hwm_after):
            if value is not None and (hwm is None or value > hwm):
                hwm = value
        total_peak = self._peak("total_bytes")
        if hwm is not None and (total_peak is None or hwm > total_peak):
            total_peak = int(hwm)
        owned_peak = self._peak("owned_bytes")
        file_peak = self._peak("file_backed_bytes")
        shared_peak = self._peak("shared_bytes")
        swap_peak = self._peak("swap_bytes")

        # Decomposition AT the instant of the sampled total peak. Never the sum
        # of independent maxima (those can come from different instants).
        at_peak = None
        if samples:
            top = max(
                samples,
                key=lambda s: -1 if s.get("total_bytes") is None
                else int(s["total_bytes"]),
            )
            if top.get("total_bytes") is not None:
                at_peak = {
                    "t_s": top.get("t_s"),
                    "phase": top.get("phase"),
                    "total_bytes": int(top["total_bytes"]),
                    "owned_bytes": top.get("owned_bytes"),
                    "file_backed_bytes": top.get("file_backed_bytes"),
                    "shared_bytes": top.get("shared_bytes"),
                }
                accounted = sum(
                    int(at_peak[k]) for k in
                    ("owned_bytes", "file_backed_bytes", "shared_bytes")
                    if at_peak[k] is not None
                )
                at_peak["unaccounted_bytes"] = (
                    int(at_peak["total_bytes"]) - accounted
                    if at_peak["owned_bytes"] is not None
                    else None
                )

        phases: dict = {}
        for s in samples:
            label = str(s.get("phase"))
            slot = phases.setdefault(label, {
                "samples": 0,
                "total_peak_bytes": None,
                "owned_peak_bytes": None,
                "file_backed_peak_bytes": None,
                "shared_peak_bytes": None,
                "started_s": s.get("t_s"),
                "last_s": s.get("t_s"),
            })
            slot["samples"] += 1
            slot["last_s"] = s.get("t_s")
            for key, out_key in (
                ("total_bytes", "total_peak_bytes"),
                ("owned_bytes", "owned_peak_bytes"),
                ("file_backed_bytes", "file_backed_peak_bytes"),
                ("shared_bytes", "shared_peak_bytes"),
            ):
                value = s.get(key)
                if value is None:
                    continue
                current = slot[out_key]
                slot[out_key] = int(value) if current is None else max(
                    int(current), int(value)
                )
        for slot in phases.values():
            if slot["started_s"] is not None and slot["last_s"] is not None:
                slot["elapsed_s"] = round(
                    float(slot["last_s"]) - float(slot["started_s"]), 6
                )

        return {
            "schema": MEMORY_COMPONENTS_SCHEMA,
            "total_peak_bytes": None if total_peak is None else int(total_peak),
            "true_total_peak_bytes": None if hwm is None else int(hwm),
            "true_total_peak_source": self._capability["true_total_peak_source"],
            "owned_peak_bytes": owned_peak,
            "file_backed_peak_bytes": file_peak,
            "shared_peak_bytes": shared_peak,
            "swap_peak_bytes": swap_peak,
            "at_total_peak": at_peak,
            "phases": phases,
            "samples": len(samples),
            "interval_s": float(self.interval_s),
            "capability": dict(self._capability),
            "missing_instruments": list(self._capability["missing_instruments"]),
            "additivity": (
                "owned_peak_bytes and file_backed_peak_bytes are INDEPENDENT "
                "maxima and must not be summed; at_total_peak is the single "
                "instant that the total peak was observed."
            ),
        }
