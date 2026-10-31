"""ZM-ZEGRID-RESILIENT-RUN-RESUME M5-B — coherent multi-file publication.

Plan §6.4/§6.5/§6.7. ``os.replace`` per file is NOT a multi-file transaction:
a crash between the science and the coverage rename would expose a mosaic whose
manifest describes bytes that are not there. This module publishes ONE COMPLETE
GENERATION of the product outputs and only then flips one atomic pointer:

* every member (raw SCI, optional aesthetic, optional uint16, coverage,
  manifest, optional run-log copy) is written inside a run-owned generation
  directory on the SAME local filesystem as ``output_dir``;
* the generation is verified before publication: regular files (never
  symlinks), nonzero size, streamed sha256, the manifest's declared output
  names agree with the staged members, and the FITS axis layout matches the
  declared canvas/channels;
* ``publication.json`` (inside the generation) records the digest of every
  member; ``CURRENT.json`` (in the run publication root) is written LAST with a
  single ``os.replace`` and carries the digest of ``publication.json``;
* readers/resume/product returns resolve ONE generation and re-verify every
  member digest. A missing/corrupt companion is an explicit refusal, never a
  ``finalized_noop``;
* the PREVIOUS generation stays untouched and authoritative until the pointer
  moves. A lost acknowledgement (publication durable, state update failed) is
  reconciled from the pointer, never recomputed;
* two different contents under the same validated identity are refused
  (content collision), while re-committing identical bytes is idempotent;
* cleanup is confined to run-owned staging generations that carry OUR staging
  marker and are NOT referenced by the current pointer. Foreign files, user
  results and older valid generations are never deleted.

Budgets: RAM (finishing + serialization working copies) and disk (members +
manifest bound + transactional margin) are admitted BEFORE any finishing or
serialization allocation. The model is conservative and documented; it is not
a universal guarantee, and an unobserved RAM probe is recorded as such rather
than presented as headroom.

Explicit non-claims: no Windows execution was performed on this host; no
network/cloud filesystem is qualified; no science, dtype, halo or finishing
algorithm is changed; no default activation. Windows uses junction/symlink
refusal and the same atomic-rename contract where NTFS provides it — the
protocol is documented in the mission report, not claimed as executed.
"""

from __future__ import annotations

import hashlib
import json
import os
import re
import shutil
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

from . import run_store as zrs
from . import run_resume as zres

SCHEMA = "zemosaic.zegrid.run_publication.v1"
POINTER_SCHEMA = "zemosaic.zegrid.run_publication_pointer.v1"
STAGING_SCHEMA = "zemosaic.zegrid.run_publication_staging.v1"

# ZM-ZEGRID-VISIBLE-ARTIFACTS-20261031: the work/publication directories are
# VISIBLE (open-source transparency). The old dot-prefixed names are read-only
# legacy fallbacks: never moved, renamed, deleted or written by a new run.
PUBLICATION_DIRNAME = "zegrid_publications"
LEGACY_PUBLICATION_DIRNAME = ".zegrid_publications"
PUBLICATION_DIRNAMES = (PUBLICATION_DIRNAME, LEGACY_PUBLICATION_DIRNAME)
LEGACY_CACHE_DIRNAME = "__zegrid_cache__"
CACHE_DIRNAME = "zegrid-cache"
PUBLICATION_FILENAME = "publication.json"
POINTER_FILENAME = "CURRENT.json"
STAGING_FILENAME = ".staging.json"
GENERATION_PREFIX = "gen-"
# ZM-ZEGRID-ACTIVE-RESUME-PARITY r4: VISIBLE discovery entry point at the output
# root. The atomic generations stay where they are (hidden directory); these two
# small files name the CURRENT immutable generation so a completed run is not
# invisible to the user. Never a copy of a FITS, never a symlink.
PRODUCTS_JSON_FILENAME = "zemosaic_products.json"
PRODUCTS_TEXT_FILENAME = "PRODUCTS.txt"
PRODUCTS_SCHEMA = "zemosaic.zegrid.products.v1"

#: Free-space kept untouched so an atomic pointer/temp write can always be
#: rolled back. This is a real floor, never lowered to make a test pass.
MIN_DISK_MARGIN_BYTES = 64 * 2 ** 20
#: Smallest host-RAM reserve kept outside the modelled finishing/serialization
#: peak. It is a model margin, not an OS reservation (plan §5.8 TOCTOU).
MIN_RAM_RESERVE_BYTES = 128 * 2 ** 20
#: Working-copy/BLAS margin applied to the modelled RAM peak (same spirit as
#: the M5-A dense-solve margin; conservative, not measured).
MODEL_MARGIN_PERCENT = 25
#: Bound for the JSON manifest member (scales with cells, capped conservatively).
MANIFEST_BOUND_BASE_BYTES = 4 * 2 ** 20
MANIFEST_BOUND_PER_CELL_BYTES = 2048
#: Bound for the optional run-log copy (the live log is bounded by the product).
RUN_LOG_BOUND_BYTES = 8 * 2 ** 20
#: FITS header/alignment overhead per member.
FITS_OVERHEAD_BYTES = 64 * 1024
#: Streamed copy/hash block (never a whole-file buffer).
IO_CHUNK_BYTES = 1 * 2 ** 20

RESERVED_MEMBER_NAMES = frozenset(
    {PUBLICATION_FILENAME, POINTER_FILENAME, STAGING_FILENAME}
)

ROLE_SCIENCE = "science"
ROLE_AESTHETIC = "aesthetic"
ROLE_COVERAGE = "coverage"
ROLE_UINT16 = "uint16"
ROLE_MANIFEST = "manifest"
ROLE_RUN_LOG = "run_log"
REQUIRED_ROLES = (ROLE_SCIENCE, ROLE_COVERAGE, ROLE_MANIFEST)


# ---------------------------------------------------------------------------
# Errors
# ---------------------------------------------------------------------------


class PublicationError(RuntimeError):
    """Base class: an explicit publication refusal (never a silent success)."""


class PublicationCapacityError(PublicationError):
    """RAM or disk admission failed BEFORE any expensive allocation."""

    def __init__(self, message: str, *, plan: Mapping[str, Any] | None = None):
        super().__init__(message)
        self.plan = dict(plan or {})


class PublicationPathError(PublicationError):
    """Unsafe member/generation path (traversal, absolute, symlink, junction)."""


class PublicationIntegrityError(PublicationError):
    """A generation is missing a member, is corrupt, or is not self-consistent."""


class PublicationCollisionError(PublicationError):
    """Same validated identity, different content: refused, never overwritten."""


class PublicationNotFound(PublicationError):
    """No published generation (or no pointer) exists for this run."""


# ---------------------------------------------------------------------------
# Strict helpers
# ---------------------------------------------------------------------------


def _req_int(value: Any, label: str, *, minimum: int | None = None) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise PublicationError(f"{label} must be an int")
    if minimum is not None and value < minimum:
        raise PublicationError(f"{label} must be >= {minimum}")
    return int(value)


def member_filename(name: Any, label: str = "member name") -> str:
    """Validate ONE output member file name (a bare basename, never a path)."""
    if not isinstance(name, str) or not name:
        raise PublicationPathError(f"{label} must be a non-empty string")
    if "\x00" in name:
        raise PublicationPathError(f"{label} contains a NUL byte")
    normalised = name.replace("\\", "/")
    if "/" in normalised or normalised in (".", ".."):
        raise PublicationPathError(f"{label} must be a bare basename: {name!r}")
    if len(name) > 200:
        raise PublicationPathError(f"{label} is too long")
    if name in RESERVED_MEMBER_NAMES:
        raise PublicationPathError(f"{label} is reserved: {name!r}")
    return name


def _is_link(path: Path) -> bool:
    """True for symlinks and (where detectable) Windows junctions."""
    try:
        if path.is_symlink():
            return True
    except OSError:
        return True
    if os.name == "nt":  # pragma: no cover - Windows NOT_RUN on this host
        try:
            import stat as _stat

            st = os.lstat(str(path))
            if getattr(st, "st_file_attributes", 0) & getattr(
                _stat, "FILE_ATTRIBUTE_REPARSE_POINT", 0
            ):
                return True
        except OSError:
            return True
    return False


def _assert_plain_dir(path: Path, label: str) -> Path:
    if _is_link(path):
        raise PublicationPathError(f"{label} is a symlink/junction: {path}")
    if not path.is_dir():
        raise PublicationPathError(f"{label} is not a directory: {path}")
    return path


def _plain_file(path: Path, label: str) -> Path:
    if _is_link(path):
        raise PublicationPathError(f"{label} is a symlink/junction: {path}")
    if not path.is_file():
        raise PublicationIntegrityError(f"{label} is missing or not a file: {path}")
    return path


def _stream_sha256(path: Path) -> tuple[str, int]:
    h = hashlib.sha256()
    total = 0
    fd = os.open(str(path), os.O_RDONLY | int(getattr(os, "O_BINARY", 0)))
    try:
        while True:
            block = os.read(fd, IO_CHUNK_BYTES)
            if not block:
                break
            h.update(block)
            total += len(block)
    finally:
        os.close(fd)
    return h.hexdigest(), int(total)


def _canonical_json_bytes(payload: Mapping[str, Any]) -> bytes:
    return zrs.canonical_json(payload)


def _sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def _read_json_bytes(path: Path, label: str) -> tuple[dict, bytes]:
    raw = path.read_bytes()
    try:
        value = json.loads(raw.decode("utf-8"))
    except Exception as exc:
        raise PublicationIntegrityError(f"{label} is not readable JSON: {exc}") from exc
    if not isinstance(value, dict):
        raise PublicationIntegrityError(f"{label} must be an object")
    return value, raw


def _atomic_write(path: Path, data: bytes, *, tmp_dir: Path) -> None:
    if _is_link(path):
        raise PublicationPathError(f"refusing to replace a symlink: {path}")
    zrs.atomic_write_bytes(Path(path), data, tmp_dir=Path(tmp_dir))


# ---------------------------------------------------------------------------
# Budget models (metadata only; no array is allocated)
# ---------------------------------------------------------------------------


def _margin(total: int) -> int:
    return int(int(total) * MODEL_MARGIN_PERCENT // 100)


def finishing_peak_bytes(
    shape_hw: Sequence[int],
    *,
    channels: int = 3,
    finishing_config: Mapping[str, Any] | None = None,
) -> dict:
    """Price the finishing working copies from metadata ONLY.

    The model is deliberately conservative: the legacy DBE path keeps one
    float32 channel copy plus fitted-plane buffers, RGB equalize and hole fill
    each produce a cube copy, and the uint16 render materialises a float64 cube
    plus the uint16 cube. ``science`` itself is never mutated (an output copy is
    returned), so the input cube stays live next to the output.
    """
    h, w = int(shape_hw[0]), int(shape_hw[1])
    if h < 1 or w < 1:
        raise PublicationError("shape_hw must be positive")
    ch = int(channels)
    if ch < 1:
        raise PublicationError("channels must be positive")
    cfg = dict(finishing_config or {})
    cube32 = int(h * w * ch * 4)
    cube64 = int(h * w * ch * 8)
    plane32 = int(h * w * 4)
    dbe = bool(cfg.get("dbe_enabled", False))
    rgb_eq = bool(cfg.get("rgb_equalize", False))
    hole = bool(cfg.get("hole_fill_enabled", False))
    u16 = bool(cfg.get("save_uint16", False))
    terms = {
        "output_cube_float32": cube32,
        "dbe_channel_and_plane_work": (cube32 + 3 * plane32) if dbe else 0,
        "rgb_equalize_copy": cube32 if rgb_eq else 0,
        "hole_fill_copy": (cube32 + h * w) if hole else 0,
        "uint16_float64_scale": cube64 if u16 else 0,
        "uint16_cube": int(h * w * ch * 2) if u16 else 0,
    }
    base = int(sum(terms.values()))
    margin = _margin(base)
    return {
        "shape_hw": [h, w],
        "channels": ch,
        "terms": terms,
        "model_margin_percent": MODEL_MARGIN_PERCENT,
        "model_margin_bytes": margin,
        "peak_bytes": int(base + margin),
        "enabled": bool(dbe or rgb_eq or hole or u16),
        "note": (
            "conservative working-copy model built from metadata only; the "
            "input assembled cube stays live next to the finishing output"
        ),
    }


def serialization_peak_bytes(
    shape_hw: Sequence[int],
    *,
    channels: int = 3,
    export_aesthetic: bool = False,
    uint16: bool = False,
) -> dict:
    """Price the FITS serialization working copies (before any buffer exists).

    ``_write_outputs`` builds a contiguous CHW copy of the finished cube, a
    sanitized copy and a second CHW copy of the raw science; each FITS write
    also materialises one member-sized temporary file inside the generation
    (transactional temp). Coverage is serialized as a contiguous float32 plane.
    """
    h, w = int(shape_hw[0]), int(shape_hw[1])
    ch = int(channels)
    cube32 = int(h * w * ch * 4)
    plane32 = int(h * w * 4)
    terms = {
        "finished_chw_copy": cube32,
        "finished_sanitize_copy": cube32,
        "raw_chw_copy": cube32,
        "raw_sanitize_copy": cube32,
        "coverage_contiguous_copy": plane32,
        "aesthetic_chw_copy": cube32 if export_aesthetic else 0,
        "uint16_chw_copy": int(h * w * ch * 2) if uint16 else 0,
        "largest_fits_temp_file": max(
            cube32,
            int(h * w * ch * 2) if uint16 else 0,
        ),
    }
    base = int(sum(terms.values()))
    margin = _margin(base)
    return {
        "shape_hw": [h, w],
        "channels": ch,
        "terms": terms,
        "model_margin_percent": MODEL_MARGIN_PERCENT,
        "model_margin_bytes": margin,
        "peak_bytes": int(base + margin),
        "note": (
            "serialization copies plus one member-sized transactional temp; "
            "no giant contiguous full-size compressed buffer is created"
        ),
    }


def publication_disk_bytes(
    shape_hw: Sequence[int],
    *,
    channels: int = 3,
    export_aesthetic: bool = False,
    uint16: bool = False,
    n_cells: int = 0,
) -> dict:
    """Price the complete generation + manifest + transactional margin."""
    h, w = int(shape_hw[0]), int(shape_hw[1])
    ch = int(channels)
    cube32 = int(h * w * ch * 4)
    members = {
        ROLE_SCIENCE: cube32 + FITS_OVERHEAD_BYTES,
        ROLE_COVERAGE: int(h * w * 4) + FITS_OVERHEAD_BYTES,
        ROLE_AESTHETIC: (cube32 + FITS_OVERHEAD_BYTES) if export_aesthetic else 0,
        ROLE_UINT16: (int(h * w * ch * 2) + FITS_OVERHEAD_BYTES) if uint16 else 0,
        ROLE_MANIFEST: int(
            min(
                MANIFEST_BOUND_BASE_BYTES
                + max(0, int(n_cells)) * MANIFEST_BOUND_PER_CELL_BYTES,
                64 * 2 ** 20,
            )
        ),
        ROLE_RUN_LOG: int(min(RUN_LOG_BOUND_BYTES, 8 * 2 ** 20)),
    }
    members_total = int(sum(members.values()))
    # One member-sized temp can coexist with the members already written.
    temp = int(max(members[ROLE_SCIENCE], members[ROLE_AESTHETIC], members[ROLE_UINT16]))
    margin = int(max(MIN_DISK_MARGIN_BYTES, members_total // 20))
    required = int(members_total + temp + margin)
    return {
        "shape_hw": [h, w],
        "channels": ch,
        "member_bounds": members,
        "members_total_bytes": members_total,
        "largest_temp_bytes": temp,
        "transactional_margin_bytes": margin,
        "required_bytes": required,
        "note": (
            "generation members + one same-directory temp + a real free-space "
            "margin; the durable per-cell store is a separate disk line (M2)"
        ),
    }


def size_finishing_and_publication(
    *,
    output_dir: str | os.PathLike[str],
    shape_hw: Sequence[int],
    channels: int = 3,
    finishing_config: Mapping[str, Any] | None = None,
    n_cells: int = 0,
    disk_free_bytes: int | None = None,
) -> dict:
    """PURE metadata-only sizing (no admission decision, no allocation).

    Read-only callers (retry/reattach, diagnostics, tests) may use this without
    claiming that any budget was observed. The returned ``admitted`` is always
    ``False`` and the admission mode is explicitly ``sizing_only``: an
    unobserved budget is NEVER labelled admitted.
    """
    cfg = dict(finishing_config or {})
    fin = finishing_peak_bytes(shape_hw, channels=channels, finishing_config=cfg)
    ser = serialization_peak_bytes(
        shape_hw,
        channels=channels,
        export_aesthetic=bool(cfg.get("export_aesthetic_fits", False)),
        uint16=bool(cfg.get("save_uint16", False)),
    )
    disk = publication_disk_bytes(
        shape_hw,
        channels=channels,
        export_aesthetic=bool(cfg.get("export_aesthetic_fits", False)),
        uint16=bool(cfg.get("save_uint16", False)),
        n_cells=int(n_cells),
    )
    ram_required = int(fin["peak_bytes"] + ser["peak_bytes"] + MIN_RAM_RESERVE_BYTES)
    disk_free: int | None
    disk_error: str | None = None
    if disk_free_bytes is None:
        try:
            disk_free = int(shutil.disk_usage(str(Path(output_dir))).free)
        except Exception as exc:
            disk_free = None
            disk_error = f"{type(exc).__name__}: {exc}"
    else:
        disk_free = int(disk_free_bytes)
    return {
        "schema": SCHEMA,
        "finishing": fin,
        "serialization": ser,
        "ram": {
            "required_bytes": ram_required,
            "available_bytes": None,
            "observed": False,
            "reserve_bytes": MIN_RAM_RESERVE_BYTES,
            "fits": None,
            "note": (
                "model bound + reserve, never an OS reservation; MemoryError "
                "and native failures remain possible (TOCTOU)"
            ),
        },
        "disk": {
            **disk,
            "free_bytes": disk_free,
            "fits": (
                None
                if disk_free is None
                else bool(disk["required_bytes"] <= disk_free)
            ),
            **({"observation_error": disk_error} if disk_error else {}),
        },
        "admitted": False,
        "admission": {"mode": "sizing_only", "admitted": False},
    }


def admit_finishing_and_publication(
    *,
    output_dir: str | os.PathLike[str],
    shape_hw: Sequence[int],
    channels: int = 3,
    finishing_config: Mapping[str, Any] | None = None,
    available_ram_bytes: int | None,
    n_cells: int = 0,
    disk_free_bytes: int | None = None,
    ram_probe_error: str | None = None,
) -> dict:
    """Fail-closed production gate BEFORE finishing/serialization allocations.

    An UNOBSERVED (``None``), invalid or failed RAM probe is an explicit
    refusal carrying the complete honest plan — never admitted, never sold as
    headroom. No ``allow_unknown_ram`` override and no new configuration exist.
    """
    plan = size_finishing_and_publication(
        output_dir=output_dir,
        shape_hw=shape_hw,
        channels=channels,
        finishing_config=finishing_config,
        n_cells=n_cells,
        disk_free_bytes=disk_free_bytes,
    )
    ram = plan["ram"]
    required = int(ram["required_bytes"])
    if available_ram_bytes is None:
        ram["probe_error"] = ram_probe_error or "unobserved"
        raise PublicationCapacityError(
            "publication RAM budget is UNOBSERVED — refusing to admit "
            f"{required} B of finishing/serialization without a fresh "
            "observation; the durable store remains resumable",
            plan=plan,
        )
    if (
        isinstance(available_ram_bytes, bool)
        or not isinstance(available_ram_bytes, int)
        or int(available_ram_bytes) < 0
    ):
        ram["probe_error"] = f"invalid observation: {available_ram_bytes!r}"
        raise PublicationCapacityError(
            "publication RAM observation is invalid — refusing admission",
            plan=plan,
        )
    ram["available_bytes"] = int(available_ram_bytes)
    ram["observed"] = True
    ram["fits"] = bool(required <= int(available_ram_bytes))
    if required > int(available_ram_bytes):
        raise PublicationCapacityError(
            "finishing/publication does not fit the observed RAM budget: "
            f"required {required} B (finishing "
            f"{plan['finishing']['peak_bytes']} B + serialization "
            f"{plan['serialization']['peak_bytes']} B + reserve "
            f"{MIN_RAM_RESERVE_BYTES} B) > available "
            f"{available_ram_bytes} B; the durable store remains resumable",
            plan=plan,
        )
    disk = plan["disk"]
    if disk["free_bytes"] is None:
        disk["fits"] = False
        raise PublicationCapacityError(
            "publication free-space observation FAILED — refusing admission "
            "(previous generation and durable commits are preserved)",
            plan=plan,
        )
    if disk["required_bytes"] > int(disk["free_bytes"]):
        disk["fits"] = False
        raise PublicationCapacityError(
            "publication generation does not fit the free disk space: "
            f"required {disk['required_bytes']} B > free {disk['free_bytes']} B "
            "(previous generation and durable commits are preserved)",
            plan=plan,
        )
    disk["fits"] = True
    plan["admitted"] = True
    plan["admission"] = {"mode": "production", "admitted": True}
    return plan


# ---------------------------------------------------------------------------
# Published generation
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class PublishedGeneration:
    run_id: str
    generation: str
    generation_dir: Path
    pointer_path: Path
    publication_path: Path
    publication_sha256: str
    members: Mapping[str, dict]
    required_roles: tuple[str, ...]
    optional_roles: Mapping[str, bool]
    created_unix_ns: int
    identity_sha256: str | None
    verified: bool = True

    def path(self, role: str) -> Path:
        try:
            row = self.members[role]
        except KeyError:
            raise PublicationError(f"generation has no member role {role!r}") from None
        return self.generation_dir / row["file"]

    def outputs_state(self) -> dict:
        """State block: generation-resolved reader contract (never a mixed set)."""
        state: dict[str, Any] = {
            "generation": self.generation,
            "generation_dir": str(self.generation_dir),
            "publication": PUBLICATION_FILENAME,
            "publication_sha256": self.publication_sha256,
            "verified": bool(self.verified),
        }
        for role, row in self.members.items():
            state[role] = {
                "file": row["file"],
                "sha256": row["sha256"],
                "nbytes": int(row["nbytes"]),
                "generation": self.generation,
            }
        return state

    def as_report(self) -> dict:
        return {
            "schema": SCHEMA,
            "run_id": self.run_id,
            "generation": self.generation,
            "generation_dir": str(self.generation_dir),
            "pointer": str(self.pointer_path),
            "publication_sha256": self.publication_sha256,
            "members": {role: dict(row) for role, row in self.members.items()},
            "required_roles": list(self.required_roles),
            "optional_roles": dict(self.optional_roles),
            "verified": bool(self.verified),
        }


# ---------------------------------------------------------------------------
# Publication session
# ---------------------------------------------------------------------------


def publication_root(output_dir: str | os.PathLike[str]) -> Path:
    """The WRITE path of the publication container (always the visible name)."""
    return Path(output_dir) / PUBLICATION_DIRNAME


def discover_publication_root(
    output_dir: str | os.PathLike[str],
) -> tuple[Path, bool]:
    """Resolve the publication container for READING.

    Returns ``(path, legacy)``: the visible name when it exists (or when neither
    exists), else the legacy hidden name when ONLY it exists. Nothing is moved
    or created here; a legacy layout is a read-only fallback.
    """
    root = Path(output_dir)
    visible = root / PUBLICATION_DIRNAME
    legacy = root / LEGACY_PUBLICATION_DIRNAME
    if visible.exists() or visible.is_symlink():
        return visible, False
    if legacy.exists() or legacy.is_symlink():
        return legacy, True
    return visible, False


def legacy_publication_in_use(
    output_dir: str | os.PathLike[str],
) -> Path | None:
    """The legacy publication root when it is the one that will be READ."""
    _path, legacy = discover_publication_root(output_dir)
    if not legacy:
        return None
    return _path


# ---------------------------------------------------------------------------
# Previous-publication / foreign-content detection (shared CLI + GUI semantics)
# ---------------------------------------------------------------------------

ZEGRID_OUTPUT_FILE_RE = re.compile(r"^mosaic_grid.*\.fits$", re.IGNORECASE)


def _dir_publication_evidence(root: Path) -> list[str]:
    """Publication evidence directly inside one directory (no recursion)."""
    evidence: list[str] = []
    try:
        if (root / "zegrid_manifest.json").is_file():
            evidence.append("zegrid_manifest.json")
        for path in sorted(root.glob("mosaic_grid*.fits")):
            if path.is_file():
                evidence.append(path.name)
        for _name in PUBLICATION_DIRNAMES:
            if (root / _name).is_dir():
                evidence.append(_name + "/")
    except OSError:
        pass
    return evidence


def _manifest_is_zegrid(path: Path) -> bool:
    try:
        payload = json.loads(path.read_bytes().decode("utf-8"))
        return isinstance(payload, dict) and payload.get("engine") == "zegrid"
    except Exception:
        return False


def _is_plain_file(path: Path) -> bool:
    """A real regular file, never a symlink/junction."""
    return path.is_file() and not _is_link(path)


def _drop_empty_scratch(tmp: Path) -> None:
    """Remove our staging dir only when it is a plain, EMPTY directory.

    Safe scratch lifecycle: ``rmdir`` alone never deletes content, so a user
    file, a symlink or a nested directory always survives.
    """
    try:
        if tmp.is_dir() and not _is_link(tmp) and not any(tmp.iterdir()):
            tmp.rmdir()
    except OSError:
        pass


def _own_products_sidecars(root: Path) -> bool:
    """True when BOTH visible product sidecars are provably OURS.

    Strong ownership evidence (all required):
      * ``zemosaic_products.json`` and ``PRODUCTS.txt`` are plain files, never
        symlinks/junctions;
      * the JSON carries OUR schema and a run/generation identity;
      * ``generation_dir`` resolves EXACTLY to this output's immutable
        generation directory, which must be a plain existing directory;
      * a valid, plain CURRENT pointer must exist and name the same run,
        generation and ``publication_sha256``;
      * ``PRODUCTS.txt`` must describe that same run/generation/digest.

    Anything forged, partial, stale or linked is treated as foreign.
    """
    json_path = root / PRODUCTS_JSON_FILENAME
    text_path = root / PRODUCTS_TEXT_FILENAME
    if not _is_plain_file(json_path) or not _is_plain_file(text_path):
        return False
    try:
        payload = json.loads(json_path.read_bytes().decode("utf-8"))
    except Exception:
        return False
    if not isinstance(payload, dict) or payload.get("schema") != PRODUCTS_SCHEMA:
        return False
    run_id = payload.get("run_id")
    generation = payload.get("generation")
    gen_dir_raw = payload.get("generation_dir")
    pub_sha = payload.get("publication_sha256")
    if not all(
        isinstance(value, str) and value
        for value in (run_id, generation, gen_dir_raw, pub_sha)
    ):
        return False
    try:
        _validate_generation_token(generation)
    except PublicationPathError:
        return False
    container, _legacy = discover_publication_root(root)
    expected = container / f"run-{_run_token(run_id)}" / generation
    try:
        if Path(gen_dir_raw).resolve() != expected.resolve():
            return False
    except OSError:
        return False
    if not expected.is_dir() or _is_link(expected):
        return False
    try:
        pointer = _read_pointer(expected.parent)
    except Exception:
        return False
    # C1 (rework-2): ownership requires a VALID, plain CURRENT pointer that
    # commits to this exact run/generation/digest. Without it there is no
    # proven commitment at all — two sidecars agreeing with each other (or a
    # surviving publication.json) is not authority. A malformed/linked pointer
    # raised above and already failed closed.
    if pointer is None:
        return False
    if str(pointer.get("run_id")) != str(run_id):
        return False
    if str(pointer.get("generation")) != generation:
        return False
    if str(pointer.get("publication_sha256")) != pub_sha:
        return False
    try:
        text = text_path.read_bytes().decode("utf-8")
    except Exception:
        return False
    return (
        f"run_id: {run_id}" in text
        and f"generation: {generation} (immutable)" in text
        and f"generation_dir: {gen_dir_raw}" in text
        and f"publication_sha256: {pub_sha}" in text
    )


def _own_empty_scratch(path: Path) -> bool:
    """True only for an EMPTY plain ``tmp`` staging dir (never a link)."""
    if _is_link(path) or not path.is_dir():
        return False
    try:
        return not any(path.iterdir())
    except OSError:
        return False


def detect_previous_publication(
    output_dir: str | os.PathLike[str],
    *,
    ignore_names: Iterable[str] = (),
) -> dict:
    """Detect a previous ZeGrid publication and foreign content.

    Publication evidence (C1 contract): ``zegrid_manifest.json``,
    ``mosaic_grid*.fits``, or a ``.zegrid_publications/`` root/pointer. Direct
    child directories are inspected too (mount-segregated publications).
    ``ignore_names`` lets a caller declare its OWN non-publication artifacts
    (GUI logs, telemetry, its durable store folder) without weakening the
    foreign-file refusal for everything else.
    """
    root = Path(output_dir)
    ignore = {str(name) for name in ignore_names}
    evidence: list[str] = []
    foreign: list[str] = []
    zegrid = False
    if not root.is_dir():
        return {"found": False, "zegrid": False, "evidence": [], "foreign": []}
    # Our own visible sidecars are recognised only with strong evidence; a
    # forged/partial/foreign pair stays foreign and still refuses —overwrite.
    products_own = _own_products_sidecars(root)

    root_evidence = _dir_publication_evidence(root)
    evidence.extend(root_evidence)
    if "zegrid_manifest.json" in root_evidence:
        zegrid = bool(_manifest_is_zegrid(root / "zegrid_manifest.json"))
    if any(name.startswith("mosaic_grid") for name in root_evidence):
        zegrid = True
    if any(name + "/" in root_evidence for name in PUBLICATION_DIRNAMES):
        zegrid = True

    try:
        entries = sorted(root.iterdir())
    except OSError:
        entries = []
    for entry in entries:
        name = entry.name
        if name in ignore:
            continue
        if name in {
            "zegrid_manifest.json", "zegrid_run.log",
            LEGACY_CACHE_DIRNAME, CACHE_DIRNAME,
            *PUBLICATION_DIRNAMES,
        }:
            continue
        if entry.is_file() and ZEGRID_OUTPUT_FILE_RE.match(name):
            continue
        if name in (PRODUCTS_JSON_FILENAME, PRODUCTS_TEXT_FILENAME) and products_own:
            continue
        if name == "tmp" and products_own and _own_empty_scratch(entry):
            continue
        if entry.is_dir():
            child = _dir_publication_evidence(entry)
            if child:
                evidence.extend(f"{name}/{item}" for item in child)
                zegrid = True
                continue
        foreign.append(name)

    return {
        "found": bool(evidence),
        "zegrid": bool(zegrid),
        "evidence": sorted(set(evidence)),
        "foreign": sorted(foreign),
    }


def _run_token(run_id: str) -> str:
    return zrs.safe_name(str(run_id), "run_id")


def run_publication_root(
    output_dir: str | os.PathLike[str],
    run_id: str,
    *,
    for_write: bool = False,
) -> Path:
    """One run's publication root.

    ``for_write=True`` always uses the visible name (new runs never write into a
    legacy hidden container). Readers resolve the legacy container when the
    visible one is absent.
    """
    base = (
        publication_root(output_dir)
        if for_write
        else discover_publication_root(output_dir)[0]
    )
    return base / f"run-{_run_token(run_id)}"


def _validate_generation_token(generation: str) -> str:
    if not isinstance(generation, str) or not generation.startswith(
        GENERATION_PREFIX
    ):
        raise PublicationPathError(f"invalid generation token: {generation!r}")
    suffix = generation[len(GENERATION_PREFIX):]
    if not suffix.isdigit() or int(suffix) < 1:
        raise PublicationPathError(f"invalid generation token: {generation!r}")
    return generation


def _read_pointer(run_root: Path) -> dict | None:
    path = run_root / POINTER_FILENAME
    if not path.exists():
        return None
    if _is_link(path):
        raise PublicationPathError(f"publication pointer is a link: {path}")
    payload, _ = _read_json_bytes(path, "publication pointer")
    if payload.get("schema") != POINTER_SCHEMA:
        raise PublicationIntegrityError(
            f"unsupported publication pointer schema: {payload.get('schema')!r}"
        )
    return payload


def _referenced_generation(run_root: Path) -> str | None:
    pointer = _read_pointer(run_root)
    if pointer is None:
        return None
    return str(pointer.get("generation"))


def inspect_same_run_staging(
    output_dir: str | os.PathLike[str], run_id: str
) -> dict:
    """READ-ONLY ownership proof for a same-run crash staging generation.

    Used by callers (the Qt GUI preflight) that must decide whether a resume may
    be OFFERED when there is no ``CURRENT.json`` yet. It never creates, deletes,
    repairs or cleans anything, and it never treats staged bytes as COMPLETE.

    ``ok`` is True only when ALL of the following hold:
    * the publication container (``zegrid_publications``, or the legacy
      ``.zegrid_publications`` fallback) exists as a plain directory and
      contains ONLY the selected run's publication root (an unrelated/other-run
      entry is refused);
    * the run root is a plain directory and contains only ``tmp`` and ``gen-N``
      directories (any other entry is foreign);
    * there is NO ``CURRENT.json`` (a pointer belongs to the committed path and
      is verified by ``resolve_current`` instead);
    * every ``gen-N`` is a plain directory carrying a schema-valid staging
      marker for the SAME run/generation with a DEAD owner pid;
    * present members are declared plain files directly inside the generation;
      absent members are allowed for incomplete staging. No extra children are
      allowed except the staging marker and a plain ``publication.json`` record;
      if that record exists, all declared members must exist too. Staging is
      never reported as COMPLETE by this ownership check;
    * no symlink/junction is used anywhere on the checked path.

    A marker-less directory, a live owner, a foreign/other-run entry, a path
    escape or a malformed marker is a refusal with a concrete reason. The engine
    remains the authority for identity, commits and cleanup after resume.
    """
    out = Path(output_dir)
    pub_root, _legacy = discover_publication_root(out)
    run_root = run_publication_root(out, run_id)
    result: dict = {"ok": False, "reason": None, "generations": []}

    def _refuse(reason: str) -> dict:
        result["ok"] = False
        result["reason"] = str(reason)
        return result

    try:
        if not pub_root.exists():
            return _refuse("no .zegrid_publications directory exists")
        if _is_link(pub_root) or not pub_root.is_dir():
            return _refuse("the publication root is not a plain directory")
        try:
            pub_entries = sorted(pub_root.iterdir(), key=lambda p: p.name)
        except OSError as exc:
            return _refuse(f"the publication root cannot be inspected: {exc}")
        other_run_dirs = [
            entry.name for entry in pub_entries if entry.name != run_root.name
        ]
        if other_run_dirs:
            return _refuse(
                "the publication root also contains unrelated run content "
                f"({', '.join(other_run_dirs)})"
            )
        if not run_root.is_dir() or _is_link(run_root):
            return _refuse("the selected run publication root is not a plain directory")
        if (run_root / POINTER_FILENAME).exists():
            return _refuse("CURRENT.json exists; use the committed-publication verifier")
        try:
            run_entries = sorted(run_root.iterdir(), key=lambda p: p.name)
        except OSError as exc:
            return _refuse(f"the run publication root cannot be inspected: {exc}")

        generations: list[str] = []
        for entry in run_entries:
            if entry.name == "tmp":
                if _is_link(entry) or not entry.is_dir():
                    return _refuse("the publication temp entry is not a plain directory")
                continue
            try:
                _validate_generation_token(entry.name)
            except PublicationPathError:
                return _refuse(
                    f"foreign entry in the selected run publication root: {entry.name!r}"
                )
            if _is_link(entry) or not entry.is_dir():
                return _refuse(f"generation {entry.name!r} is not a plain directory")
            generations.append(entry.name)
        if not generations:
            return _refuse("no staging generation exists for the selected run")

        for generation in generations:
            gen_dir = run_root / generation
            marker = gen_dir / STAGING_FILENAME
            if marker.is_file() and not _is_link(marker):
                try:
                    payload, _raw = _read_json_bytes(marker, "staging marker")
                except PublicationError as exc:
                    return _refuse(
                        f"staging marker of {generation!r} is unreadable: {exc}"
                    )
                if payload.get("schema") != STAGING_SCHEMA:
                    return _refuse(
                        f"staging marker of {generation!r} has an unsupported schema"
                    )
                if str(payload.get("run_id")) != str(run_id):
                    return _refuse(
                        f"staging marker of {generation!r} belongs to another run"
                    )
                if str(payload.get("generation")) != str(generation):
                    return _refuse(
                        f"staging marker of {generation!r} names another generation"
                    )
                pid = payload.get("pid")
                try:
                    pid_int = int(pid)
                except (TypeError, ValueError):
                    return _refuse(
                        f"staging marker of {generation!r} has no provable owner pid"
                    )
                if zrs.pid_alive(pid_int):
                    return _refuse(
                        f"staging generation {generation!r} is owned by a LIVE "
                        f"writer (pid {pid_int}); not a crash artifact"
                    )
                declared = payload.get("members")
                if not isinstance(declared, list) or not declared:
                    return _refuse(
                        f"staging marker of {generation!r} does not declare members"
                    )
                names: set[str] = set()
                for item in declared:
                    try:
                        names.add(member_filename(item, "staging member"))
                    except PublicationError as exc:
                        return _refuse(
                            f"staging marker of {generation!r} declares an unsafe "
                            f"member: {exc}"
                        )
                allowed = names | {STAGING_FILENAME, PUBLICATION_FILENAME}
                try:
                    children = sorted(gen_dir.iterdir(), key=lambda p: p.name)
                except OSError as exc:
                    return _refuse(
                        f"generation {generation!r} cannot be inspected: {exc}"
                    )
                for child in children:
                    if child.name not in allowed:
                        return _refuse(
                            f"generation {generation!r} holds foreign content "
                            f"({child.name!r})"
                        )
                    if _is_link(child) or not child.is_file():
                        return _refuse(
                            f"generation {generation!r} member {child.name!r} is "
                            "not a plain file"
                        )
                    if (
                        child.name != STAGING_FILENAME
                        and child.name != PUBLICATION_FILENAME
                        and child.name not in names
                    ):  # pragma: no cover - redundant with 'allowed'
                        return _refuse(
                            f"generation {generation!r} member {child.name!r} is "
                            "not declared by the marker"
                        )
                # F1 (REWORK-4): ownership is NOT completeness. A real crash may
                # happen right after ``begin`` (zero members) or mid-writing (a
                # subset/truncated plain file). Missing declared members are
                # therefore tolerated here; every PRESENT child is still checked
                # above (declared/plain/no extra/no link). Staged bytes are never
                # reported as COMPLETE, and the engine remains the authority for
                # identity, commits and cleanup.
                if (gen_dir / PUBLICATION_FILENAME).is_file():
                    missing = [
                        name
                        for name in sorted(names)
                        if not (gen_dir / name).is_file()
                        or _is_link(gen_dir / name)
                    ]
                    if missing:
                        return _refuse(
                            f"generation {generation!r} has a publication record "
                            f"but is missing declared member(s) {missing!r}; "
                            "the record is not trusted"
                        )
                result.setdefault("present_members", {})[generation] = sum(
                    1 for name in names if (gen_dir / name).is_file()
                )
                continue

            # No staging marker: a complete-looking record without a pointer is
            # NOT provably OURS. Refuse conservatively and preserve it; never
            # delete a publication record to force a resume.
            if (gen_dir / PUBLICATION_FILENAME).is_file():
                return _refuse(
                    f"generation {generation!r} has a publication record but no "
                    "staging marker; ownership is not provable, so it is left "
                    "untouched and not resumed"
                )
            return _refuse(
                f"generation {generation!r} has no staging marker; ownership is "
                "not provable"
            )

        result["ok"] = True
        result["generations"] = generations
        return result
    except Exception as exc:  # fail closed on any inspection error
        return _refuse(f"publication staging inspection failed: {type(exc).__name__}: {exc}")


def inspect_publication_root_ownership(
    output_dir: str | os.PathLike[str], run_id: str
) -> dict:
    """READ-ONLY ownership check of the publication container for ONE run.

    Applies to BOTH the committed-pointer and the staged-only resume branches:
    the publication root must contain ONLY the selected run's root, and that
    root may contain only ``CURRENT.json``, ``tmp`` and ``gen-N`` directories
    (plain, never links). Any other run/foreign entry is a refusal, even when a
    valid CURRENT exists for the selected run. Nothing is created or modified.
    """
    out = Path(output_dir)
    pub_root, _legacy = discover_publication_root(out)
    run_root = run_publication_root(out, run_id)
    result: dict = {"ok": False, "reason": None}

    def _refuse(reason: str) -> dict:
        result["ok"] = False
        result["reason"] = str(reason)
        return result

    try:
        if not pub_root.is_dir() or _is_link(pub_root):
            return _refuse("the publication root is not a plain directory")
        try:
            entries = sorted(pub_root.iterdir(), key=lambda p: p.name)
        except OSError as exc:
            return _refuse(f"the publication root cannot be inspected: {exc}")
        others = [entry.name for entry in entries if entry.name != run_root.name]
        if others:
            return _refuse(
                "the publication root contains unrelated run/foreign content "
                f"({', '.join(others)})"
            )
        if not run_root.is_dir() or _is_link(run_root):
            return _refuse("the selected run publication root is not a plain directory")
        try:
            run_entries = sorted(run_root.iterdir(), key=lambda p: p.name)
        except OSError as exc:
            return _refuse(f"the run publication root cannot be inspected: {exc}")
        for entry in run_entries:
            if entry.name == POINTER_FILENAME:
                if _is_link(entry) or not entry.is_file():
                    return _refuse("CURRENT.json is not a plain file")
                continue
            if entry.name == "tmp":
                if _is_link(entry) or not entry.is_dir():
                    return _refuse("the publication temp entry is not a plain directory")
                continue
            try:
                _validate_generation_token(entry.name)
            except PublicationPathError:
                return _refuse(
                    f"foreign entry in the selected run publication root: {entry.name!r}"
                )
            if _is_link(entry) or not entry.is_dir():
                return _refuse(f"generation {entry.name!r} is not a plain directory")
        result["ok"] = True
        return result
    except Exception as exc:
        return _refuse(
            f"publication ownership inspection failed: {type(exc).__name__}: {exc}"
        )


def cleanup_orphan_generations(
    output_dir: str | os.PathLike[str],
    run_id: str,
    *,
    keep_generation: str | None = None,
    _run_root: Path | None = None,
) -> list[str]:
    """Remove ONLY our own uncommitted staging generations.

    A candidate must be a ``gen-N`` directory directly under the run publication
    root, must carry a staging marker whose ``run_id`` matches, and must not be
    the generation referenced by the current pointer (or explicitly kept).
    Everything else (foreign files, other runs, previous committed generations)
    is left untouched.
    """
    run_root = (
        Path(_run_root)
        if _run_root is not None
        else run_publication_root(output_dir, run_id)
    )
    if not run_root.is_dir():
        return []
    _assert_plain_dir(run_root, "run publication root")
    keep = set()
    referenced = _referenced_generation(run_root)
    if referenced is not None:
        keep.add(referenced)
    if keep_generation is not None:
        keep.add(keep_generation)
    removed: list[str] = []
    for entry in sorted(run_root.iterdir(), key=lambda p: p.name):
        if entry.name in (POINTER_FILENAME, "tmp"):
            continue
        if not entry.name.startswith(GENERATION_PREFIX):
            continue
        try:
            _validate_generation_token(entry.name)
        except PublicationPathError:
            continue
        if entry.name in keep:
            continue
        if not entry.is_dir() or _is_link(entry):
            continue
        # F3: a publication record is proof of a complete (and possibly already
        # committed) generation. A residual staging marker after a crash or a
        # failed unlink MUST NOT be read as evidence of disposable scratch.
        if (entry / PUBLICATION_FILENAME).exists():
            continue
        marker = entry / STAGING_FILENAME
        if not marker.is_file() or _is_link(marker):
            continue
        try:
            payload, _ = _read_json_bytes(marker, "staging marker")
        except PublicationError:
            continue
        if payload.get("schema") != STAGING_SCHEMA:
            continue
        if str(payload.get("run_id")) != str(run_id):
            continue
        # A live owner may still be staging: never delete its scratch.
        try:
            if zrs.pid_alive(int(payload.get("pid"))):
                continue
        except Exception:
            continue
        declared = payload.get("members")
        if not isinstance(declared, list) or not declared:
            # A marker that cannot enumerate its own members is not provably
            # disposable scratch (older/foreign marker): preserve it.
            continue
        allowed = {str(v) for v in declared} | {STAGING_FILENAME}
        try:
            entries = list(entry.iterdir())
        except OSError:
            continue
        disposable = True
        for child in entries:
            if child.name not in allowed or _is_link(child) or not child.is_file():
                disposable = False
                break
        if not disposable:
            continue
        shutil.rmtree(entry)
        removed.append(entry.name)
    if removed:
        zrs.fsync_dir(run_root)
    return removed


class PublicationSession:
    """Stage a complete generation, verify it, then flip ONE atomic pointer."""

    def __init__(
        self,
        *,
        run_id: str,
        run_root: Path,
        generation: str,
        generation_dir: Path,
        members: Mapping[str, str],
        plan: Mapping[str, Any],
        identity_sha256: str | None,
        tmp_dir: Path,
        session_nonce: str,
        created_unix_ns: int,
        output_dir: Path | None = None,
    ) -> None:
        self.run_id = str(run_id)
        self.run_root = Path(run_root)
        self.generation = generation
        self.generation_dir = Path(generation_dir)
        self.members = {str(k): str(v) for k, v in members.items()}
        self.plan = dict(plan)
        self.identity_sha256 = identity_sha256
        self.tmp_dir = Path(tmp_dir)
        self.session_nonce = session_nonce
        self.created_unix_ns = int(created_unix_ns)
        self.output_dir = Path(output_dir) if output_dir is not None else None

    # -- opening ---------------------------------------------------------
    @classmethod
    def begin(
        cls,
        output_dir: str | os.PathLike[str],
        run_id: str,
        *,
        members: Mapping[str, str],
        shape_hw: Sequence[int],
        channels: int = 3,
        finishing_config: Mapping[str, Any] | None = None,
        available_ram_bytes: int | None,
        n_cells: int = 0,
        identity_sha256: str | None = None,
        generation: str | None = None,
        ram_probe_error: str | None = None,
    ) -> "PublicationSession":
        if not run_id:
            raise PublicationError("publication requires a run_id")
        out = Path(output_dir)
        if not out.is_dir():
            raise PublicationPathError(f"output_dir is not a directory: {out}")
        # Budget FIRST: a refused admission creates no generation directory.
        plan = admit_finishing_and_publication(
            output_dir=out,
            shape_hw=shape_hw,
            channels=channels,
            finishing_config=finishing_config,
            available_ram_bytes=available_ram_bytes,
            n_cells=n_cells,
            ram_probe_error=ram_probe_error,
        )
        normalised: dict[str, str] = {}
        for role, filename in members.items():
            if not isinstance(role, str) or not role:
                raise PublicationPathError("member role must be a non-empty string")
            normalised[str(role)] = member_filename(filename, f"member {role}")
        if len(set(normalised.values())) != len(normalised.values()):
            raise PublicationPathError("two member roles share the same file name")
        run_root = run_publication_root(out, run_id, for_write=True)
        pub_root = run_root.parent
        if pub_root.exists() and _is_link(pub_root):
            raise PublicationPathError(
                f"publication root is a symlink/junction: {pub_root}"
            )
        # ZM-ZEGRID-VISIBLE-ARTIFACTS-20261031: the visible container must never
        # be written into when it already holds unrelated content. Only our own
        # run roots (``run-*`` plain directories) are tolerated here.
        if pub_root.exists():
            if not pub_root.is_dir():
                raise PublicationPathError(
                    f"publication root is not a plain directory: {pub_root}"
                )
            try:
                entries = sorted(entry.name for entry in pub_root.iterdir())
            except OSError as exc:
                raise PublicationPathError(
                    f"publication root cannot be inspected ({pub_root}): {exc}"
                ) from exc
            foreign = [
                name for name in entries
                if not (name.startswith("run-") and (pub_root / name).is_dir())
            ]
            if foreign:
                raise PublicationPathError(
                    "refusing to write into the publication folder "
                    f"{pub_root}: it already contains unrelated content "
                    f"({', '.join(foreign[:5])}); nothing was written"
                )
        run_root.mkdir(parents=True, exist_ok=True)
        _assert_plain_dir(run_root, "run publication root")
        # Cleanup is confined to OUR uncommitted staging generations; the
        # generation referenced by CURRENT.json and every foreign file stay.
        cleanup_orphan_generations(out, run_id, _run_root=run_root)
        tmp_dir = run_root / "tmp"
        tmp_dir.mkdir(parents=True, exist_ok=True)
        _assert_plain_dir(tmp_dir, "publication temp dir")
        if generation is not None:
            generation = _validate_generation_token(generation)
        else:
            highest = 0
            for entry in run_root.iterdir():
                if entry.name.startswith(GENERATION_PREFIX):
                    try:
                        highest = max(highest, int(entry.name[len(GENERATION_PREFIX):]))
                    except ValueError:
                        continue
            generation = f"{GENERATION_PREFIX}{highest + 1}"
        gen_dir = run_root / generation
        try:
            gen_dir.mkdir()
        except FileExistsError:
            raise PublicationCollisionError(
                f"generation {generation!r} already exists for run {run_id!r}; "
                "a retry must reattach and prove identical content"
            ) from None
        _assert_plain_dir(gen_dir, "generation dir")
        nonce = hashlib.sha256(
            f"{os.getpid()}-{time.time_ns()}-{generation}".encode("ascii")
        ).hexdigest()[:32]
        created = time.time_ns()
        marker = {
            "schema": STAGING_SCHEMA,
            "run_id": str(run_id),
            "generation": generation,
            "session_nonce": nonce,
            "pid": int(os.getpid()),
            "created_unix_ns": int(created),
            "members": sorted(normalised.values()),
            "note": "uncommitted staging; never authoritative until CURRENT.json",
        }
        _atomic_write(
            gen_dir / STAGING_FILENAME,
            _canonical_json_bytes(marker),
            tmp_dir=tmp_dir,
        )
        return cls(
            run_id=run_id,
            run_root=run_root,
            generation=generation,
            generation_dir=gen_dir,
            members=normalised,
            plan=plan,
            identity_sha256=None if identity_sha256 is None else str(identity_sha256),
            tmp_dir=tmp_dir,
            session_nonce=nonce,
            created_unix_ns=created,
            output_dir=out,
        )

    @classmethod
    def reattach(
        cls,
        output_dir: str | os.PathLike[str],
        run_id: str,
        generation: str,
        *,
        members: Mapping[str, str],
        plan: Mapping[str, Any] | None = None,
        identity_sha256: str | None = None,
        finishing_config: Mapping[str, Any] | None = None,
        shape_hw: Sequence[int] | None = None,
        channels: int = 3,
        n_cells: int = 0,
    ) -> "PublicationSession":
        """Reopen an existing generation to prove idempotence (never rewrite)."""
        generation = _validate_generation_token(generation)
        run_root = run_publication_root(output_dir, run_id)
        _assert_plain_dir(run_root, "run publication root")
        gen_dir = run_root / generation
        _assert_plain_dir(gen_dir, "generation dir")
        marker = gen_dir / STAGING_FILENAME
        nonce = "reattached"
        created = time.time_ns()
        if marker.is_file() and not _is_link(marker):
            payload, _ = _read_json_bytes(marker, "staging marker")
            if str(payload.get("run_id")) != str(run_id):
                raise PublicationCollisionError(
                    "staging marker belongs to another run identity"
                )
            nonce = str(payload.get("session_nonce"))
            created = int(payload.get("created_unix_ns", created))
        if plan is None:
            if shape_hw is None:
                raise PublicationError(
                    "reattach requires either a plan or the shape metadata"
                )
            # Read-only sizing only: reattach NEVER claims a production
            # admission and never allocates.
            plan = size_finishing_and_publication(
                output_dir=output_dir,
                shape_hw=shape_hw,
                channels=channels,
                finishing_config=finishing_config,
                n_cells=n_cells,
            )
        return cls(
            run_id=run_id,
            run_root=run_root,
            generation=generation,
            generation_dir=gen_dir,
            members={str(k): member_filename(v, f"member {k}") for k, v in members.items()},
            plan=dict(plan),
            identity_sha256=None if identity_sha256 is None else str(identity_sha256),
            tmp_dir=run_root / "tmp",
            session_nonce=nonce,
            created_unix_ns=int(created),
            output_dir=Path(output_dir),
        )

    # -- staging ---------------------------------------------------------
    def path_for(self, role: str) -> Path:
        try:
            filename = self.members[str(role)]
        except KeyError:
            raise PublicationError(f"publication has no member role {role!r}") from None
        target = self.generation_dir / filename
        # Confinement: the resolved parent must be exactly the generation dir.
        if target.parent.resolve() != self.generation_dir.resolve():
            raise PublicationPathError(f"member {filename!r} escapes the generation")
        if _is_link(target):
            raise PublicationPathError(f"member target is a link: {target}")
        return target

    def path_for_name(self, filename: str) -> Path:
        """Resolve a canonical output name to its staged generation member."""
        name = member_filename(filename)
        target = self.generation_dir / name
        if target.parent.resolve() != self.generation_dir.resolve():
            raise PublicationPathError(f"member {name!r} escapes the generation")
        return target

    def include_auxiliary(self, role: str, source: str | os.PathLike[str]) -> dict:
        """Copy a small auxiliary file (run log) into the generation, streamed."""
        src = Path(source)
        _plain_file(src, "auxiliary source")
        if self.plan.get("disk", {}).get("fits") is False:
            raise PublicationCapacityError(
                "cannot stage an auxiliary member: disk admission failed",
                plan=self.plan,
            )
        target = self.path_for(role)
        if target.exists() and _is_link(target):
            raise PublicationPathError(f"auxiliary target is a link: {target}")
        tmp = self.tmp_dir / f".aux-{os.getpid()}-{time.time_ns()}"
        h = hashlib.sha256()
        total = 0
        fd_in = os.open(str(src), os.O_RDONLY | int(getattr(os, "O_BINARY", 0)))
        fd_out = os.open(
            str(tmp), os.O_WRONLY | os.O_CREAT | os.O_EXCL | int(getattr(os, "O_BINARY", 0))
        )
        try:
            while True:
                block = os.read(fd_in, IO_CHUNK_BYTES)
                if not block:
                    break
                h.update(block)
                total += len(block)
                view = memoryview(block)
                while view:
                    written = os.write(fd_out, view)
                    if written <= 0:  # pragma: no cover - defensive
                        raise OSError("short write")
                    view = view[written:]
            zrs.fsync_file(fd_out)
        finally:
            os.close(fd_in)
            os.close(fd_out)
        try:
            os.replace(tmp, target)
        except BaseException:
            try:
                os.unlink(tmp)
            except OSError:
                pass
            raise
        zrs.fsync_dir(self.generation_dir)
        return {"role": role, "file": target.name, "sha256": h.hexdigest(), "nbytes": total}

    # -- verification + commit ------------------------------------------
    def _verify_manifest_contract(
        self, members: Mapping[str, dict], optional: Mapping[str, bool] | None = None
    ) -> None:
        optional = {str(k): bool(v) for k, v in (optional or {}).items()}
        manifest_path = self.generation_dir / self.members[ROLE_MANIFEST]
        payload, _ = _read_json_bytes(manifest_path, "zegrid manifest")
        outputs = payload.get("outputs")
        if not isinstance(outputs, dict):
            raise PublicationIntegrityError("manifest.outputs must be an object")
        expected = {
            ROLE_SCIENCE: ROLE_SCIENCE,
            ROLE_COVERAGE: ROLE_COVERAGE,
            ROLE_AESTHETIC: ROLE_AESTHETIC,
            ROLE_UINT16: ROLE_UINT16,
            ROLE_RUN_LOG: ROLE_RUN_LOG,
        }
        for role, key in expected.items():
            declared = outputs.get(key)
            if role in self.members:
                if optional.get(role) is False:
                    # The optional companion was not produced (e.g. uint16 after
                    # a fail-safe finishing failure). The manifest must agree.
                    if declared is not None:
                        raise PublicationIntegrityError(
                            f"manifest declares {key}={declared!r} but the "
                            "commit marks that optional member absent"
                        )
                    continue
                if declared != self.members[role]:
                    raise PublicationIntegrityError(
                        f"manifest.outputs.{key}={declared!r} does not name the "
                        f"staged {role} member {self.members[role]!r}"
                    )
            elif declared is not None and role != ROLE_RUN_LOG:
                # A declared science/aesthetic/coverage/uint16 companion that is
                # not staged is a missing companion, never a valid generation.
                # The run log is explicitly NON-authoritative (live diagnostic).
                raise PublicationIntegrityError(
                    f"manifest declares {key}={declared!r} but the generation "
                    "does not stage that member"
                )
        if payload.get("science_reference") != outputs.get("science"):
            raise PublicationIntegrityError(
                "manifest.science_reference does not name outputs.science"
            )
        contract = payload.get("science_output_contract")
        if not isinstance(contract, dict) or not isinstance(contract.get("raw"), dict):
            raise PublicationIntegrityError(
                "manifest.science_output_contract.raw is missing"
            )
        if contract["raw"].get("file") != outputs.get("science"):
            raise PublicationIntegrityError(
                "manifest science contract does not name outputs.science"
            )
        if ROLE_AESTHETIC in self.members:
            if not isinstance(contract.get("aesthetic"), dict) or (
                contract["aesthetic"].get("file") != outputs.get("aesthetic")
            ):
                raise PublicationIntegrityError(
                    "manifest aesthetic contract does not name the staged aesthetic"
                )

    def _verify_fits_layout(self, members: Mapping[str, dict]) -> None:
        try:
            from astropy.io import fits
        except Exception as exc:  # pragma: no cover - astropy is a hard dependency
            raise PublicationIntegrityError(
                f"FITS verification unavailable: {exc}"
            ) from exc
        h, w = (int(v) for v in self.plan["serialization"]["shape_hw"])
        ch = int(self.plan["serialization"]["channels"])
        checks = {
            ROLE_SCIENCE: (ch, h, w),
            ROLE_AESTHETIC: (ch, h, w),
            ROLE_UINT16: (ch, h, w),
            ROLE_COVERAGE: (h, w),
        }
        for role, expected in checks.items():
            row = members.get(role)
            if row is None:
                continue
            path = self.generation_dir / row["file"]
            try:
                header = fits.getheader(path, ext=0)
            except Exception as exc:
                raise PublicationIntegrityError(
                    f"staged {role} FITS is unreadable: {exc}"
                ) from exc
            naxis = int(header.get("NAXIS", 0))
            shape = tuple(int(header.get(f"NAXIS{i}", 0)) for i in range(1, naxis + 1))
            # FITS axis order is reversed versus numpy: (C, H, W) -> W, H, C.
            observed = tuple(reversed(shape))
            if observed != tuple(expected):
                raise PublicationIntegrityError(
                    f"staged {role} FITS shape {observed} != declared {expected}"
                )
            if int(header.get("NAXIS", 0)) != len(expected):
                raise PublicationIntegrityError(
                    f"staged {role} FITS dimensionality is inconsistent"
                )

    # -- F2: immutable publication records -----------------------------
    def _existing_publication_record(self) -> tuple[dict, bytes, str] | None:
        path = self.generation_dir / PUBLICATION_FILENAME
        if not path.exists():
            return None
        _plain_file(path, "publication.json")
        payload, raw = _read_json_bytes(path, "publication.json")
        return payload, raw, _sha256_bytes(raw)

    def _pointer(self) -> dict | None:
        return _read_pointer(self.run_root)

    def _assert_pointer_not_newer(self, pointer: Mapping[str, Any] | None) -> None:
        if pointer is None:
            return
        other = str(pointer.get("generation", ""))
        if other == self.generation:
            return
        try:
            other_n = int(other[len(GENERATION_PREFIX):])
            self_n = int(self.generation[len(GENERATION_PREFIX):])
        except ValueError:
            return
        if other_n > self_n:
            raise PublicationCollisionError(
                f"stale retry of {self.generation!r} would roll CURRENT "
                f"backward from {other!r}; refused without any write"
            )

    def _build_publication(
        self,
        *,
        members: Mapping[str, dict],
        required: Sequence[str],
        optional: Mapping[str, bool],
        metadata: Mapping[str, Any] | None,
        created_unix_ns: int,
        staging_created_unix_ns: int,
    ) -> dict:
        return {
            "schema": SCHEMA,
            "run_id": self.run_id,
            "generation": self.generation,
            "created_unix_ns": int(created_unix_ns),
            "staging_created_unix_ns": int(staging_created_unix_ns),
            "identity_sha256": self.identity_sha256,
            "shape_hw": [int(v) for v in self.plan["serialization"]["shape_hw"]],
            "channels": int(self.plan["serialization"]["channels"]),
            "members": {str(k): dict(v) for k, v in members.items()},
            "required_roles": [str(v) for v in required],
            "optional_roles": {str(k): bool(v) for k, v in optional.items()},
            "integrity": (
                "sha256 over every member's complete bytes and over this file; "
                "CURRENT.json is written last and names this file's digest"
            ),
            "roles": {
                ROLE_SCIENCE: "immutable pre-finishing assembled science (SCI)",
                ROLE_AESTHETIC: "delivered aesthetic float32 (optional)",
                ROLE_COVERAGE: "coverage/stack-depth plane",
                ROLE_UINT16: "derived uint16 render (optional, non-scientific)",
                ROLE_MANIFEST: "ZeGrid manifest (reader contract)",
                ROLE_RUN_LOG: "run log copy (auxiliary, non-authoritative)",
            },
            "note": (
                "ONE complete generation; readers resolve it through CURRENT.json "
                "and must not mix members from two generations"
            ),
            **({"metadata": dict(metadata)} if metadata else {}),
        }

    def _published_from_record(
        self,
        payload: Mapping[str, Any],
        raw: bytes,
        digest: str,
    ) -> PublishedGeneration:
        members = {
            str(role): {
                "file": member_filename(row.get("file"), f"member {role} file"),
                "sha256": str(row.get("sha256")),
                "nbytes": int(row.get("nbytes")),
            }
            for role, row in (payload.get("members") or {}).items()
        }
        return PublishedGeneration(
            run_id=self.run_id,
            generation=self.generation,
            generation_dir=self.generation_dir,
            pointer_path=self.run_root / POINTER_FILENAME,
            publication_path=self.generation_dir / PUBLICATION_FILENAME,
            publication_sha256=digest,
            members=members,
            required_roles=tuple(
                str(v) for v in (payload.get("required_roles") or ())
            ),
            optional_roles={
                str(k): bool(v)
                for k, v in (payload.get("optional_roles") or {}).items()
            },
            created_unix_ns=int(payload.get("created_unix_ns", 0)),
            identity_sha256=(
                None
                if payload.get("identity_sha256") is None
                else str(payload.get("identity_sha256"))
            ),
            verified=True,
        )

    def _flush_member(self, path: Path) -> None:
        """Flush ONE member's data to the OS durable layer (F4).

        Write-first, read-fallback, deliberately:

        * Win32: ``os.fsync`` is ``FlushFileBuffers``, which requires a handle
          carrying write access (GENERIC_WRITE). A read-only handle cannot be
          flushed, so the write attempt is the ONLY portable path to a durable
          member on Windows. The member is opened ``O_RDWR`` (never
          ``O_WRONLY``/``O_TRUNC``/``O_CREAT``): flushing must not change one
          byte, the size, or create a file.
        * POSIX: ``fsync`` also accepts a read-only descriptor, and a stricter
          filesystem/ACL can refuse the write open. The ``O_RDONLY`` attempt is
          the documented fallback for exactly that case.

        The fallback is a RETRY, never a skip: if both attempts fail the
        ``OSError`` is propagated, and ``commit`` turns it into
        ``PublicationError`` BEFORE ``publication.json`` and BEFORE the
        pointer, leaving the previous ``CURRENT.json`` authoritative.

        Real Windows/NTFS is NOT_RUN on this Linux host: the Win32 semantics
        above are documented platform behaviour, covered here defensively by
        simulation, not an observed NTFS capture. ``run_store.fsync_dir``
        already has its own documented Windows no-op and is out of scope.
        """
        binary = int(getattr(os, "O_BINARY", 0))
        write_error: OSError | None = None
        try:
            fd = os.open(str(path), os.O_RDWR | binary)
        except OSError as exc:
            write_error = exc
        else:
            try:
                os.fsync(fd)
                return
            except OSError as exc:
                write_error = exc
            finally:
                os.close(fd)
        try:
            fd = os.open(str(path), os.O_RDONLY | binary)
        except OSError as exc:
            raise OSError(
                "member flush needs a write-capable handle (Win32 "
                "FlushFileBuffers) and the read-only fallback could not be "
                f"opened either: {path}: {type(write_error).__name__}: "
                f"{write_error}; {type(exc).__name__}: {exc}"
            ) from exc
        try:
            os.fsync(fd)
        except OSError as exc:
            raise OSError(
                "member flush failed on both a write and a read-only handle: "
                f"{path}: {type(write_error).__name__}: {write_error}; "
                f"{type(exc).__name__}: {exc}"
            ) from exc
        finally:
            os.close(fd)

    def _fsync_ancestors(self) -> None:
        """Flush new generation entries and their ancestors BEFORE the pointer."""
        paths = [self.generation_dir, self.run_root]
        if self.output_dir is not None:
            paths.append(Path(self.output_dir) / PUBLICATION_DIRNAME)
            paths.append(Path(self.output_dir))
        for path in paths:
            zrs.fsync_dir(Path(path))

    def _pointer_payload(
        self,
        *,
        publication_sha: str,
        publication_bytes: int,
        members: Mapping[str, dict],
        required: Sequence[str],
    ) -> dict:
        return {
            "schema": POINTER_SCHEMA,
            "run_id": self.run_id,
            "generation": self.generation,
            "publication_sha256": publication_sha,
            "publication_bytes": int(publication_bytes),
            "member_count": int(len(members)),
            "required_roles": [str(v) for v in required],
            "identity_sha256": self.identity_sha256,
            "published_unix_ns": int(time.time_ns()),
        }

    def commit(
        self,
        *,
        required_roles: Sequence[str] = REQUIRED_ROLES,
        optional_roles: Mapping[str, bool] | None = None,
        metadata: Mapping[str, Any] | None = None,
        idempotent: bool = True,
    ) -> PublishedGeneration:
        """Verify every member, publish ``publication.json``, then the pointer.

        F2: a generation that already carries a publication record is
        IMMUTABLE. An exact retry validates it and returns it without touching
        members, publication.json or the pointer; any semantic difference
        (identity, roles, options, metadata, canvas, digests) is a collision.
        """
        required = tuple(str(role) for role in required_roles)
        unknown = set(required) - set(self.members)
        if unknown:
            raise PublicationError(f"required roles are not staged: {sorted(unknown)}")
        optional = {str(k): bool(v) for k, v in (optional_roles or {}).items()}
        unknown = set(optional) - set(self.members)
        if unknown:
            raise PublicationError(f"optional roles are not declared: {sorted(unknown)}")
        if set(required) & set(optional):
            raise PublicationError("a role cannot be both required and optional")
        for role in required:
            if self.plan.get("disk", {}).get("fits") is False:
                raise PublicationCapacityError(
                    "cannot publish on a refused disk admission", plan=self.plan
                )
            _plain_file(self.path_for(role), f"required member {role}")
        # The manifest must agree with the staged set BEFORE any digest is sold.
        if ROLE_MANIFEST in self.members:
            self._verify_manifest_contract({}, optional)
        members: dict[str, dict] = {}
        for role, filename in self.members.items():
            path = self.generation_dir / filename
            if not path.exists():
                if role in required:
                    raise PublicationIntegrityError(
                        f"required member {role} ({filename}) is missing"
                    )
                if optional.get(role) is True:
                    raise PublicationIntegrityError(
                        f"declared optional member {role} ({filename}) is missing; "
                        "a declared companion is never silently ignored"
                    )
                if role in optional:
                    continue
                if idempotent:
                    continue
                raise PublicationIntegrityError(
                    f"declared member {role} ({filename}) is missing"
                )
            _plain_file(path, f"member {role}")
            if path.stat().st_size <= 0:
                raise PublicationIntegrityError(f"member {role} is empty")
            digest, nbytes = _stream_sha256(path)
            members[role] = {"file": filename, "sha256": digest, "nbytes": int(nbytes)}
        self._verify_fits_layout(members)

        pointer = self._pointer()
        self._assert_pointer_not_newer(pointer)
        existing = self._existing_publication_record()
        if existing is not None:
            payload, raw, digest = existing
            candidate = self._build_publication(
                members=members,
                required=required,
                optional=optional,
                metadata=metadata,
                created_unix_ns=int(payload.get("created_unix_ns", 0)),
                staging_created_unix_ns=int(
                    payload.get(
                        "staging_created_unix_ns",
                        payload.get("created_unix_ns", self.created_unix_ns),
                    )
                ),
            )
            if _canonical_json_bytes(candidate) != raw:
                raise PublicationCollisionError(
                    f"generation {self.generation!r} already contains a "
                    "different immutable publication for this identity "
                    "(identity/roles/options/metadata/canvas/digests); "
                    "refused without any write"
                )
            if pointer is not None and str(pointer.get("run_id")) != self.run_id:
                raise PublicationIntegrityError(
                    "publication pointer belongs to another run identity"
                )
            if pointer is not None and str(pointer.get("generation")) == self.generation:
                if str(pointer.get("publication_sha256")) != digest:
                    raise PublicationIntegrityError(
                        "pointer/publication digest mismatch for the SAME "
                        "immutable generation; refused without any write"
                    )
                # Exact idempotent retry: NOTHING is written (no member, no
                # publication.json, no pointer).
                return self._published_from_record(payload, raw, digest)
            # The record is already durable but CURRENT is missing or points to
            # an older generation (crash before pointer). Recovering the pointer
            # writes ONLY the pointer; the publication record stays untouched.
            self._fsync_ancestors()
            pointer_payload = self._pointer_payload(
                publication_sha=digest,
                publication_bytes=len(raw),
                members=members,
                required=required,
            )
            _atomic_write(
                self.run_root / POINTER_FILENAME,
                _canonical_json_bytes(pointer_payload),
                tmp_dir=self.tmp_dir,
            )
            try:
                marker = self.generation_dir / STAGING_FILENAME
                if marker.is_file() and not _is_link(marker):
                    marker.unlink()
            except OSError:
                pass
            return self._published_from_record(payload, raw, digest)

        # A NEW publication requires a real production admission (F1). A
        # read-only sizing/reattach session may validate an existing record but
        # can never create one.
        if self.plan.get("admitted") is not True:
            raise PublicationCapacityError(
                "refusing to create a NEW publication without a production "
                "RAM/disk admission (sizing-only session)",
                plan=self.plan,
            )
        publication = self._build_publication(
            members=members,
            required=required,
            optional=optional,
            metadata=metadata,
            created_unix_ns=int(time.time_ns()),
            staging_created_unix_ns=int(self.created_unix_ns),
        )
        publication_bytes = _canonical_json_bytes(publication)
        publication_sha = _sha256_bytes(publication_bytes)
        # F4: flush EVERY member and the new directory entries BEFORE the
        # publication record and the final pointer. A pre-pointer failure is
        # explicit and leaves the previous CURRENT intact.
        try:
            for role in members:
                self._flush_member(self.generation_dir / members[role]["file"])
            zrs.fsync_dir(self.generation_dir)
            self._fsync_ancestors()
        except OSError as exc:
            raise PublicationError(
                "durable member/directory flush failed before publication: "
                f"{type(exc).__name__}: {exc}; no pointer was written"
            ) from exc
        pub_path = self.generation_dir / PUBLICATION_FILENAME
        _atomic_write(pub_path, publication_bytes, tmp_dir=self.tmp_dir)
        pointer = self._pointer()
        self._assert_pointer_not_newer(pointer)
        pointer_payload = self._pointer_payload(
            publication_sha=publication_sha,
            publication_bytes=len(publication_bytes),
            members=members,
            required=required,
        )
        pointer_path = self.run_root / POINTER_FILENAME
        _atomic_write(
            pointer_path,
            _canonical_json_bytes(pointer_payload),
            tmp_dir=self.tmp_dir,
        )
        # The pointer is the authority from this instant. Removing the staging
        # marker is cosmetic; any failure here must not deny the publication.
        try:
            marker = self.generation_dir / STAGING_FILENAME
            if marker.is_file() and not _is_link(marker):
                marker.unlink()
        except OSError:
            pass
        zrs.fsync_dir(self.generation_dir)
        return PublishedGeneration(
            run_id=self.run_id,
            generation=self.generation,
            generation_dir=self.generation_dir,
            pointer_path=pointer_path,
            publication_path=pub_path,
            publication_sha256=publication_sha,
            members=members,
            required_roles=required,
            optional_roles=optional,
            created_unix_ns=int(publication["created_unix_ns"]),
            identity_sha256=self.identity_sha256,
            verified=True,
        )


# ---------------------------------------------------------------------------
# VISIBLE product entry point (output root)
# ---------------------------------------------------------------------------

_VISIBLE_ROLE_ORDER = (
    ROLE_SCIENCE,
    ROLE_AESTHETIC,
    ROLE_UINT16,
    ROLE_COVERAGE,
    ROLE_MANIFEST,
    ROLE_RUN_LOG,
)
_VISIBLE_ROLE_LABEL = {
    ROLE_SCIENCE: "science",
    ROLE_AESTHETIC: "aesthetic",
    ROLE_UINT16: "uint16",
    ROLE_COVERAGE: "coverage",
    ROLE_MANIFEST: "manifest",
    ROLE_RUN_LOG: "run_log",
}


# ZM-ZEGRID-ROOT-DELIVERABLES-20261031: the finished products are DELIVERED at
# the OUTPUT ROOT under their Classic-compatible names, like Classic/SDS. The
# immutable generation directory stays the durable archive and the source of
# truth for resume/verification; the root copy is a convenience deliverable.
# ``run_log`` is excluded: it is already written at the root (auxiliary).
ROOT_DELIVERABLE_ROLES = (
    ROLE_SCIENCE,
    ROLE_AESTHETIC,
    ROLE_COVERAGE,
    ROLE_UINT16,
    ROLE_MANIFEST,
)


def _root_target_matches(target: Path, sha256: str, nbytes: int) -> bool:
    """True when the existing root file already IS the verified member bytes."""
    if _is_link(target) or not target.is_file():
        return False
    try:
        if target.stat().st_size != int(nbytes):
            return False
        return _stream_sha256(target)[0] == str(sha256)
    except OSError:
        return False


def _copy_member_to_root(
    source: Path,
    target: Path,
    *,
    expected_sha256: str,
    expected_nbytes: int,
) -> None:
    """Atomically copy ONE verified generation member onto the output root.

    The bytes are streamed into a temp file beside the staging directory,
    hashed WHILE copied and fsynced; only a byte-exact copy is renamed onto the
    target with ``os.replace``. A mismatch or any error removes the temp file
    and leaves a prior target byte-identical.
    """
    import tempfile

    if _is_link(target):
        raise PublicationPathError(f"refusing to replace a symlink: {target}")
    tmp = target.parent / "tmp"
    tmp.mkdir(parents=True, exist_ok=True)
    if _is_link(tmp):
        raise PublicationPathError(f"root staging directory is a link: {tmp}")
    fd, tmp_name = tempfile.mkstemp(prefix=f".{target.name}.", dir=str(tmp))
    digest = hashlib.sha256()
    total = 0
    try:
        with os.fdopen(fd, "wb") as out, open(source, "rb") as inp:
            while True:
                block = inp.read(IO_CHUNK_BYTES)
                if not block:
                    break
                digest.update(block)
                total += len(block)
                out.write(block)
            out.flush()
            os.fsync(out.fileno())
        if digest.hexdigest() != str(expected_sha256) or total != int(
            expected_nbytes
        ):
            raise PublicationIntegrityError(
                "root delivery refused: copied bytes do not match the verified "
                f"member for {target.name}"
            )
        os.replace(tmp_name, target)
        zrs.fsync_dir(target.parent)
    except BaseException:
        try:
            os.unlink(tmp_name)
        except OSError:
            pass
        _drop_empty_scratch(tmp)
        raise
    _drop_empty_scratch(tmp)


def publish_root_deliverables(
    output_dir: str | os.PathLike[str],
    generation: PublishedGeneration,
) -> dict:
    """Deliver the verified generation members to the OUTPUT ROOT.

    Called only AFTER ``CURRENT.json`` is durable and the generation verified.
    Each produced role is copied under its Classic-compatible name with the
    same atomic-write discipline as the visible markers: a temp file, an fsync,
    then ONE ``os.replace``. A file that already carries the verified bytes is
    left untouched (idempotent reconcile). A failure keeps the prior root file
    byte-identical and is REPORTED, never raised: the publication stays
    authoritative and a convenience copy must never fail the run.

    Returns ``{"root_files": {role: {path, sha256, nbytes}}, "failures": [...]}``.
    """
    root = Path(output_dir)
    _assert_plain_dir(root, "output root")
    gen_dir = Path(generation.generation_dir)
    _assert_plain_dir(gen_dir, "generation directory")
    root_resolved = root.resolve()
    delivered: dict[str, dict] = {}
    failures: list[str] = []
    for role in ROOT_DELIVERABLE_ROLES:
        row = generation.members.get(role)
        if row is None:
            continue
        try:
            filename = member_filename(row.get("file"), f"member {role} file")
            sha256 = str(row.get("sha256"))
            nbytes = int(row.get("nbytes"))
            source = gen_dir / filename
            _plain_file(source, f"generation member {role}")
            target = root / filename
            if not _root_target_matches(target, sha256, nbytes):
                _copy_member_to_root(
                    source,
                    target,
                    expected_sha256=sha256,
                    expected_nbytes=nbytes,
                )
        except Exception as exc:  # noqa: BLE001 - reported, never raised
            failures.append(f"{role}: {type(exc).__name__}: {exc}")
            continue
        delivered[str(role)] = {
            "path": str(root_resolved / filename),
            "sha256": sha256,
            "nbytes": nbytes,
        }
    return {"root_files": delivered, "failures": failures}


def _products_payload(
    generation: PublishedGeneration,
    root_files: Mapping[str, Mapping[str, Any]] | None = None,
) -> dict:
    gen_dir = Path(generation.generation_dir).resolve()
    products: dict[str, dict] = {}
    for role in _VISIBLE_ROLE_ORDER:
        row = generation.members.get(role)
        if row is None:
            continue
        products[str(role)] = {
            "file": str(row["file"]),
            "sha256": str(row["sha256"]),
            "nbytes": int(row["nbytes"]),
            "path": str(gen_dir / str(row["file"])),
        }
    delivered: dict[str, dict] = {}
    for role, row in (root_files or {}).items():
        if not isinstance(row, Mapping):
            continue
        delivered[str(role)] = {
            "path": str(row.get("path", "")),
            "sha256": str(row.get("sha256", "")),
            "nbytes": int(row.get("nbytes", 0)),
        }
    produced = int(getattr(generation, "created_unix_ns", 0) or 0)
    return {
        "schema": PRODUCTS_SCHEMA,
        "run_id": str(generation.run_id),
        "generation": str(generation.generation),
        "generation_dir": str(gen_dir),
        "publication_sha256": str(generation.publication_sha256),
        "produced_unix_ns": produced if produced > 0 else int(time.time_ns()),
        "products": products,
        "root_files": delivered,
        "note": (
            "finished products are DELIVERED at the output root (root_files, "
            "atomic copies of the verified members); the IMMUTABLE generation "
            "directory (generation_dir) remains the durable archive and the "
            "source of truth for verification and resume"
        ),
    }


def _products_text(payload: Mapping[str, Any]) -> str:
    lines = [
        "ZeMosaic ZeGrid — published products",
        f"run_id: {payload['run_id']}",
        f"generation: {payload['generation']} (immutable)",
        f"generation_dir: {payload['generation_dir']}",
        f"produced_unix_ns: {payload['produced_unix_ns']}",
        f"publication_sha256: {payload['publication_sha256']}",
        "",
    ]
    products = payload.get("products") or {}
    root_files = payload.get("root_files") or {}
    for role in _VISIBLE_ROLE_ORDER:
        row = products.get(role)
        if row is None:
            continue
        delivered = root_files.get(role)
        if isinstance(delivered, Mapping) and delivered.get("path"):
            lines.append(
                f"{_VISIBLE_ROLE_LABEL.get(role, role):10s}: "
                f"{delivered['path']} ({int(delivered['nbytes'])} B)"
            )
        else:
            lines.append(
                f"{_VISIBLE_ROLE_LABEL.get(role, role):10s}: {row['path']} "
                f"({int(row['nbytes'])} B) [generation copy, root delivery "
                "failed]"
            )
    lines += [
        "",
        "This file names the CURRENT published generation. The finished "
        "products above are",
        "delivered at the OUTPUT ROOT (atomic copies of the verified "
        "members). The immutable",
        "generation directory (generation_dir) remains the durable archive "
        "and the source of",
        "truth for verification and resume.",
        "",
    ]
    return "\n".join(lines)


def _replace_visible_pair(root: Path, files: Mapping[str, bytes]) -> None:
    """Stage BOTH visible files, then replace them; rollback on failure.

    A failure during staging leaves the previous files byte-identical (nothing
    is replaced). A failure during the replace phase restores the previous bytes
    of every file already replaced, so the pair is never torn.
    """
    import tempfile

    tmp = root / "tmp"
    tmp.mkdir(parents=True, exist_ok=True)
    previous: dict[str, bytes | None] = {}
    staged: list[tuple[Path, Path]] = []
    try:
        for name, data in files.items():
            target = root / name
            if _is_link(target):
                raise PublicationPathError(
                    f"refusing to replace a symlink: {target}"
                )
            previous[name] = target.read_bytes() if target.is_file() else None
            fd, tmp_name = tempfile.mkstemp(prefix=f".{name}.", dir=str(tmp))
            try:
                with os.fdopen(fd, "wb") as handle:
                    handle.write(data)
                    handle.flush()
                    os.fsync(handle.fileno())
            except BaseException:
                try:
                    os.unlink(tmp_name)
                except OSError:
                    pass
                raise
            staged.append((Path(tmp_name), target))
    except BaseException:
        for path, _target in staged:
            try:
                path.unlink()
            except OSError:
                pass
        _drop_empty_scratch(tmp)
        raise
    replaced: list[Path] = []
    try:
        for path, target in staged:
            os.replace(path, target)
            replaced.append(target)
    except BaseException:
        for target in replaced:
            old = previous.get(target.name)
            try:
                if old is None:
                    target.unlink(missing_ok=True)
                else:
                    fd, tmp_name = tempfile.mkstemp(
                        prefix=f".{target.name}.rb.", dir=str(tmp)
                    )
                    with os.fdopen(fd, "wb") as handle:
                        handle.write(old)
                        handle.flush()
                        os.fsync(handle.fileno())
                    os.replace(tmp_name, target)
            except OSError:
                pass
        for path, _target in staged:
            try:
                path.unlink()
            except OSError:
                pass
        _drop_empty_scratch(tmp)
        raise
    _drop_empty_scratch(tmp)
    zrs.fsync_dir(root)


def write_visible_products(
    output_dir: str | os.PathLike[str],
    generation: PublishedGeneration,
    *,
    root_files: Mapping[str, Mapping[str, Any]] | None = None,
) -> dict:
    """Publish the VISIBLE entry point for ONE durable generation.

    Called only AFTER ``CURRENT.json`` is durable and the generation verified.
    Writes ``zemosaic_products.json`` (machine-readable) and ``PRODUCTS.txt``
    (human-readable) at the OUTPUT ROOT. Both name the delivered ROOT paths
    (``root_files``, produced by :func:`publish_root_deliverables`) while still
    recording the immutable ``generation_dir`` for audit/resume. Idempotent for
    the same generation; a NEW generation replaces the pointers only after it
    is durable. A failure leaves the previous pointers byte-identical.
    """
    root = Path(output_dir)
    if not root.is_dir():
        raise PublicationPathError(f"output root is not a directory: {root}")
    gen_dir = Path(generation.generation_dir)
    if not gen_dir.is_dir() or _is_link(gen_dir):
        raise PublicationPathError(
            f"generation directory is missing or not a plain directory: {gen_dir}"
        )
    payload = _products_payload(generation, root_files=root_files)
    files = {
        PRODUCTS_JSON_FILENAME: _canonical_json_bytes(payload),
        PRODUCTS_TEXT_FILENAME: _products_text(payload).encode("utf-8"),
    }
    _replace_visible_pair(root, files)
    return payload


# ---------------------------------------------------------------------------
# Reader / resume reconciliation
# ---------------------------------------------------------------------------


def resolve_current(
    output_dir: str | os.PathLike[str],
    run_id: str,
    *,
    verify_members: bool = True,
    expected_identity_sha256: str | None = None,
) -> PublishedGeneration:
    """Resolve the single CURRENT generation and verify it, or refuse.

    No member is ever read from two generations. A missing pointer means no
    publication; a missing/corrupt member is an integrity refusal (never a
    silent fallback to an older generation).
    """
    run_root = run_publication_root(output_dir, run_id)
    pointer = _read_pointer(run_root) if run_root.is_dir() else None
    if pointer is None:
        raise PublicationNotFound(f"no published generation for run {run_id!r}")
    if str(pointer.get("run_id")) != str(run_id):
        raise PublicationIntegrityError(
            "publication pointer belongs to another run identity"
        )
    generation = _validate_generation_token(str(pointer.get("generation")))
    gen_dir = run_root / generation
    _assert_plain_dir(gen_dir, "published generation dir")
    pub_path = gen_dir / PUBLICATION_FILENAME
    _plain_file(pub_path, "publication.json")
    publication, raw = _read_json_bytes(pub_path, "publication.json")
    digest = _sha256_bytes(raw)
    if digest != str(pointer.get("publication_sha256")):
        raise PublicationIntegrityError(
            "publication.json digest does not match the pointer (publication "
            "is corrupt or was modified)"
        )
    if int(pointer.get("publication_bytes", len(raw))) != len(raw):
        raise PublicationIntegrityError("publication.json size mismatch")
    if publication.get("schema") != SCHEMA:
        raise PublicationIntegrityError(
            f"unsupported publication schema: {publication.get('schema')!r}"
        )
    if str(publication.get("run_id")) != str(run_id) or str(
        publication.get("generation")
    ) != generation:
        raise PublicationIntegrityError("publication identity/pointer mismatch")
    if expected_identity_sha256 is not None:
        observed = publication.get("identity_sha256")
        if observed is not None and str(observed) != str(expected_identity_sha256):
            raise zres.ResumeIncompatible(
                "resume refused: the published generation belongs to another "
                "run identity"
            )
    members = publication.get("members")
    if not isinstance(members, dict) or not members:
        raise PublicationIntegrityError("publication.members must be a non-empty object")
    required = tuple(str(v) for v in publication.get("required_roles") or ())
    for role in required:
        if role not in members:
            raise PublicationIntegrityError(
                f"required role {role!r} is absent from publication.members"
            )
    verified_members: dict[str, dict] = {}
    for role, row in members.items():
        if not isinstance(row, dict):
            raise PublicationIntegrityError(f"member {role!r} must be an object")
        filename = member_filename(row.get("file"), f"member {role} file")
        path = gen_dir / filename
        _plain_file(path, f"member {role}")
        nbytes = int(row.get("nbytes", -1))
        if path.stat().st_size != nbytes:
            raise PublicationIntegrityError(
                f"member {role} size {path.stat().st_size} != declared {nbytes}"
            )
        if verify_members:
            observed, observed_n = _stream_sha256(path)
            if observed != str(row.get("sha256")):
                raise PublicationIntegrityError(
                    f"member {role} sha256 mismatch (corrupt or replaced)"
                )
            if observed_n != nbytes:
                raise PublicationIntegrityError(f"member {role} byte count mismatch")
        verified_members[role] = {
            "file": filename,
            "sha256": str(row.get("sha256")),
            "nbytes": nbytes,
        }
    return PublishedGeneration(
        run_id=str(run_id),
        generation=generation,
        generation_dir=gen_dir,
        pointer_path=run_root / POINTER_FILENAME,
        publication_path=pub_path,
        publication_sha256=digest,
        members=verified_members,
        required_roles=required,
        optional_roles={
            str(k): bool(v) for k, v in (publication.get("optional_roles") or {}).items()
        },
        created_unix_ns=int(publication.get("created_unix_ns", 0)),
        identity_sha256=(
            None
            if publication.get("identity_sha256") is None
            else str(publication.get("identity_sha256"))
        ),
        verified=bool(verify_members),
    )


@dataclass(frozen=True)
class FinalizationVerdict:
    published: bool
    reason: str
    generation: PublishedGeneration | None
    state_finalized: bool

    def as_report(self) -> dict:
        return {
            "published": bool(self.published),
            "reason": self.reason,
            "state_finalized": bool(self.state_finalized),
            "generation": (
                None if self.generation is None else self.generation.as_report()
            ),
        }


ACK_ROW_FIELDS = ("file", "sha256", "nbytes", "generation")


def ack_matches_generation(outputs, generation) -> tuple[bool, list[str]]:
    """F5: is ``state.outputs`` a COMPLETE ack derived from this authority?

    It is not enough for the three required rows to carry a matching sha256: the
    ack must be exactly the generation-resolved contract produced by
    :meth:`PublishedGeneration.outputs_state` (generation/publication identity +
    file/sha256/nbytes/generation for EVERY published member, optional members
    included when present). Anything missing or different is a stale ack.
    """
    problems: list[str] = []
    if not isinstance(outputs, Mapping):
        return False, ["state.outputs is not an object"]
    if str(outputs.get("generation")) != generation.generation:
        problems.append("state.outputs.generation mismatch")
    if str(outputs.get("publication")) != PUBLICATION_FILENAME:
        problems.append("state.outputs.publication mismatch")
    if str(outputs.get("publication_sha256")) != generation.publication_sha256:
        problems.append("state.outputs.publication_sha256 mismatch")
    if outputs.get("verified") is not True:
        problems.append("state.outputs.verified is not true")
    for role, row in generation.members.items():
        state_row = outputs.get(role)
        if not isinstance(state_row, Mapping):
            problems.append(f"state.outputs.{role} is missing")
            continue
        expected = {
            "file": row["file"],
            "sha256": row["sha256"],
            "nbytes": int(row["nbytes"]),
            "generation": generation.generation,
        }
        for field in ACK_ROW_FIELDS:
            if state_row.get(field) != expected[field]:
                problems.append(f"state.outputs.{role}.{field} mismatch")
    return (not problems, problems)


def verify_finalized(
    output_dir: str | os.PathLike[str],
    run_root: str | os.PathLike[str],
    run_id: str,
    *,
    expected_identity_sha256: str | None = None,
) -> FinalizationVerdict:
    """Reconcile ``state.finalized`` with the DURABLE publication.

    * publication present + state acknowledged: verified finalized;
    * publication present + state not updated (lost ack): STILL published — the
      caller may repair the state, but no recomputation is required;
    * state finalized + publication missing/corrupt/incomplete: explicit
      integrity refusal (never a fake COMPLETE/finalized_noop).
    """
    try:
        state = zres.read_state(Path(run_root))
        state_finalized = bool(state.get("finalized"))
    except Exception as exc:
        state = None
        state_finalized = False
        state_error = f"{type(exc).__name__}: {exc}"
    else:
        state_error = None
    try:
        generation = resolve_current(
            output_dir,
            run_id,
            verify_members=True,
            expected_identity_sha256=expected_identity_sha256,
        )
    except PublicationNotFound:
        if state_finalized:
            raise PublicationIntegrityError(
                "run state claims finalized but no publication generation "
                "exists (refusing a fake COMPLETE)"
            ) from None
        return FinalizationVerdict(
            published=False,
            reason="no_publication",
            generation=None,
            state_finalized=False,
        )
    state_consistent = False
    if state is not None and state_finalized:
        state_consistent, _ack_problems = ack_matches_generation(
            state.get("outputs") or {}, generation
        )
    if state_finalized:
        return FinalizationVerdict(
            published=True,
            # The DURABLE generation is the authority. A stale/failed state
            # acknowledgement is reconciled from it, never allowed to fake a
            # COMPLETE (and never to force a recomputation).
            reason=(
                "verified_finalized"
                if state_consistent
                else "publication_durable_state_stale"
            ),
            generation=generation,
            state_finalized=True,
        )
    return FinalizationVerdict(
        published=True,
        reason="publication_durable_state_not_updated",
        generation=generation,
        state_finalized=False,
    )


__all__ = [
    "SCHEMA",
    "POINTER_SCHEMA",
    "STAGING_SCHEMA",
    "PUBLICATION_DIRNAME",
    "PUBLICATION_FILENAME",
    "POINTER_FILENAME",
    "PublicationError",
    "PublicationCapacityError",
    "PublicationPathError",
    "PublicationIntegrityError",
    "PublicationCollisionError",
    "PublicationNotFound",
    "PublishedGeneration",
    "FinalizationVerdict",
    "PublicationSession",
    "publication_root",
    "run_publication_root",
    "cleanup_orphan_generations",
    "finishing_peak_bytes",
    "serialization_peak_bytes",
    "publication_disk_bytes",
    "admit_finishing_and_publication",
    "resolve_current",
    "ack_matches_generation",
    "verify_finalized",
    "member_filename",
]
