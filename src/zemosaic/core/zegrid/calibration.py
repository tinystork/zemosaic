"""ZM-ZEGRID-R55-B — immutable, scoped calibration profiles.

A calibration profile is valid ONLY for the exact
``(phase, backend, host_class, path)`` scope and the declared ``valid_ranges``.
Out-of-range observations are never extrapolated silently: the caller either
marks the candidate ineligible or falls back to an explicitly supplied static
bound and records ``R55_UNCALIBRATED``.

No coefficient ships in this module. R54's OOC coefficients (47.2 B/cell or the
t16/t24 timings) are NOT product coefficients and cannot be loaded here:

* product-admissible profiles must use a ``product.*`` path;
* the OOC harness path is ``experimental.ooc`` and is refused by the product
  registry even when someone registers it.

Coefficients are non-negative integers (bytes or bytes/px/ch/frame). Floats are
refused so a serialized budget can never drift.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from types import MappingProxyType
from typing import Iterable, Mapping

from .resource_envelope import (
    BACKENDS,
    PHASES,
    EstimateEnvelope,
    InvalidEnvelope,
    make_estimate,
    require_int,
    SCHEMA as _ENVELOPE_SCHEMA,
)

__all__ = [
    "SCHEMA",
    "MODEL_LINEAR_COMPONENTS",
    "PRODUCT_PATH_PREFIX",
    "EXPERIMENTAL_PATH",
    "RANGE_KEYS",
    "COEFFICIENT_KEYS",
    "CalibrationError",
    "InvalidProfile",
    "CalibrationOutOfRange",
    "CalibrationNotAdmissible",
    "CalibrationProfile",
    "ProfileRegistry",
    "static_bound_estimate",
    "estimate_components",
]

SCHEMA = "zemosaic-r55-calibration-profile-v1"
MODEL_LINEAR_COMPONENTS = "linear_components"
PRODUCT_PATH_PREFIX = "product."
EXPERIMENTAL_PATH = "experimental.ooc"

RANGE_KEYS = ("n_frames", "patch_area_px", "tile_px")
COEFFICIENT_KEYS = frozenset(
    {
        "process_baseline_bytes",
        "tile_workspace_bytes_per_px_ch_frame",
        "tile_fixed_overhead_bytes",
        "aligned_input_bytes_per_px_ch_frame",
        "rejection_mask_bytes_per_px_ch_frame",
    }
)


class CalibrationError(ValueError):
    """Base class for calibration-contract violations."""


class InvalidProfile(CalibrationError):
    """A profile is structurally invalid or lies about its scope/coefficients."""


class CalibrationOutOfRange(CalibrationError):
    """An observation is outside the profile's valid range (no extrapolation)."""

    def __init__(self, message: str, *, field_name: str, value: int, bounds: tuple):
        super().__init__(message)
        self.field_name = field_name
        self.value = int(value)
        self.bounds = (int(bounds[0]), int(bounds[1]))


class CalibrationNotAdmissible(CalibrationError):
    """A matching profile exists but is not admissible on the product path."""


def _require_nonempty_str(value, name: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise InvalidProfile(f"{name} must be a non-empty string, got {value!r}")
    return value


def _require_margin(num, den) -> tuple[int, int]:
    n = require_int(num, "margin_num")
    d = require_int(den, "margin_den", minimum=1)
    if n > d:
        raise InvalidProfile(f"margin_num {n} > margin_den {d}")
    return n, d


@dataclass(frozen=True)
class CalibrationProfile:
    """Immutable, scoped, versioned calibration profile.

    ``coefficients`` and ``valid_ranges`` are normalised to sorted tuples of
    pairs and exposed through read-only mappings, so a frozen instance cannot be
    mutated through a nested dict.
    """

    calibration_id: str
    host_id: str
    os: str
    cpu_logical: int
    ram_total_bytes: int
    phase: str
    backend: str
    path: str
    model: str
    coefficients: tuple[tuple[str, int], ...]
    valid_ranges: tuple[tuple[str, tuple[int, int]], ...]
    measured: Mapping[str, object] = field(default_factory=dict)
    margin_num: int = 15
    margin_den: int = 100
    product_admissible: bool = True
    scope_note: str = ""
    schema: str = SCHEMA

    def __post_init__(self) -> None:
        # Normalise nested containers into immutable, canonically ordered views.
        coeffs = self.coefficients
        if isinstance(coeffs, Mapping):
            coeffs = tuple(coeffs.items())
        normalised = []
        for key, value in coeffs:
            if isinstance(value, bool) or not isinstance(value, int):
                raise InvalidProfile(
                    f"coefficient {key!r} must be an int (floats/bools are "
                    f"refused so a serialized budget cannot drift), got {value!r}"
                )
            normalised.append((str(key), int(value)))
        object.__setattr__(self, "coefficients", tuple(sorted(normalised)))
        ranges = self.valid_ranges
        if isinstance(ranges, Mapping):
            ranges = tuple(ranges.items())
        normalised_ranges = []
        for key, bounds in ranges:
            if (
                not isinstance(bounds, (tuple, list))
                or len(bounds) != 2
                or any(
                    isinstance(v, bool) or not isinstance(v, int) for v in bounds
                )
            ):
                raise InvalidProfile(
                    f"valid_ranges[{key!r}] must be two ints, got {bounds!r}"
                )
            normalised_ranges.append((str(key), (int(bounds[0]), int(bounds[1]))))
        object.__setattr__(
            self, "valid_ranges", tuple(sorted(normalised_ranges))
        )
        for name in ("margin_num", "margin_den"):
            value = getattr(self, name)
            if isinstance(value, bool) or not isinstance(value, int):
                raise InvalidProfile(f"{name} must be an int, got {value!r}")
        if not isinstance(self.measured, Mapping):
            raise InvalidProfile("measured must be a mapping")
        object.__setattr__(self, "measured", MappingProxyType(dict(self.measured)))

    # -- read-only views -------------------------------------------------
    @property
    def coefficient_map(self) -> Mapping[str, int]:
        return MappingProxyType(dict(self.coefficients))

    @property
    def range_map(self) -> Mapping[str, tuple[int, int]]:
        return MappingProxyType(dict(self.valid_ranges))

    def to_dict(self) -> dict:
        return {
            "schema": self.schema,
            "calibration_id": self.calibration_id,
            "host_id": self.host_id,
            "os": self.os,
            "cpu_logical": int(self.cpu_logical),
            "ram_total_bytes": int(self.ram_total_bytes),
            "phase": self.phase,
            "backend": self.backend,
            "path": self.path,
            "model": self.model,
            "coefficients": {k: int(v) for k, v in self.coefficients},
            "valid_ranges": {k: [int(v[0]), int(v[1])] for k, v in self.valid_ranges},
            "measured": dict(self.measured),
            "margin_num": int(self.margin_num),
            "margin_den": int(self.margin_den),
            "product_admissible": bool(self.product_admissible),
            "scope_note": self.scope_note,
        }

    def validate(self) -> None:
        if self.schema != SCHEMA:
            raise InvalidProfile(f"schema {self.schema!r} != {SCHEMA!r}")
        for name in ("calibration_id", "host_id", "os", "path", "scope_note"):
            value = getattr(self, name)
            if name == "scope_note" and value == "":
                continue
            _require_nonempty_str(value, name)
        if self.phase not in PHASES:
            raise InvalidProfile(f"unknown phase {self.phase!r} (closed vocabulary)")
        if self.backend not in BACKENDS:
            raise InvalidProfile(f"unknown backend {self.backend!r} (closed vocabulary)")
        if self.model != MODEL_LINEAR_COMPONENTS:
            raise InvalidProfile(
                f"unsupported model {self.model!r}; only {MODEL_LINEAR_COMPONENTS!r} "
                "is implemented"
            )
        require_int(self.cpu_logical, "cpu_logical", minimum=1)
        require_int(self.ram_total_bytes, "ram_total_bytes", minimum=1)
        _require_margin(self.margin_num, self.margin_den)

        # Scope rule: only a product.* profile may be product-admissible.
        if not isinstance(self.product_admissible, bool):
            raise InvalidProfile("product_admissible must be a bool")
        is_product_path = self.path.startswith(PRODUCT_PATH_PREFIX)
        if self.product_admissible and not is_product_path:
            raise InvalidProfile(
                f"profile path {self.path!r} is not product-admissible "
                f"(expected a {PRODUCT_PATH_PREFIX}* path)"
            )
        if not self.product_admissible and is_product_path:
            raise InvalidProfile(
                f"profile path {self.path!r} must be marked product_admissible=True"
            )
        if self.path == EXPERIMENTAL_PATH and self.product_admissible:
            raise InvalidProfile("experimental.ooc can never be product-admissible")

        if not self.coefficients:
            raise InvalidProfile("coefficients must not be empty")
        names = [name for name, _ in self.coefficients]
        if names != sorted(names) or len(set(names)) != len(names):
            raise InvalidProfile(f"coefficient names must be sorted/unique: {names}")
        unknown = set(names) - COEFFICIENT_KEYS
        if unknown:
            raise InvalidProfile(
                f"unknown coefficient(s) {sorted(unknown)}; closed vocabulary is "
                f"{sorted(COEFFICIENT_KEYS)}"
            )
        for name, value in self.coefficients:
            require_int(value, f"coefficients[{name}]")

        range_names = [name for name, _ in self.valid_ranges]
        if range_names != sorted(range_names) or len(set(range_names)) != len(range_names):
            raise InvalidProfile(f"valid_ranges names must be sorted/unique: {range_names}")
        missing = set(RANGE_KEYS) - set(range_names)
        if missing:
            raise InvalidProfile(f"valid_ranges is missing {sorted(missing)}")
        unknown = set(range_names) - set(RANGE_KEYS)
        if unknown:
            raise InvalidProfile(f"unknown valid_ranges key(s) {sorted(unknown)}")
        for name, (lo, hi) in self.valid_ranges:
            require_int(lo, f"valid_ranges[{name}].lo")
            require_int(hi, f"valid_ranges[{name}].hi")
            if hi < lo:
                raise InvalidProfile(
                    f"valid_ranges[{name}] has hi {hi} < lo {lo}"
                )
        if self.range_map["tile_px"][1] < 1:
            raise InvalidProfile("valid_ranges[tile_px] must allow tile_px >= 1")

        if self.measured:
            for name, value in self.measured.items():
                _require_nonempty_str(name, "measured key")
                if isinstance(value, bool) or not isinstance(value, (int, float, str)):
                    raise InvalidProfile(
                        f"measured[{name}] must be an int/float/str, got {value!r}"
                    )
                if isinstance(value, (int, float)) and float(value) < 0:
                    raise InvalidProfile(f"measured[{name}] must be non-negative")
            if "evidence_sha256" in self.measured:
                ev = self.measured["evidence_sha256"]
                if not isinstance(ev, str) or len(ev) != 64 or any(
                    c not in "0123456789abcdef" for c in ev
                ):
                    raise InvalidProfile(
                        "measured.evidence_sha256 must be 64 lowercase hex chars"
                    )

    # -- range guard -----------------------------------------------------
    def check_range(self, *, tile_px: int, n_frames: int, patch_area_px: int) -> None:
        """Refuse any observation outside ``valid_ranges`` (never extrapolate)."""
        values = {
            "tile_px": require_int(tile_px, "tile_px", minimum=1),
            "n_frames": require_int(n_frames, "n_frames", minimum=1),
            "patch_area_px": require_int(patch_area_px, "patch_area_px", minimum=1),
        }
        ranges = self.range_map
        for name in RANGE_KEYS:
            lo, hi = ranges[name]
            value = values[name]
            if value < lo or value > hi:
                raise CalibrationOutOfRange(
                    f"profile {self.calibration_id!r}: {name}={value} is outside its "
                    f"declared valid range [{lo}, {hi}]; refusing to extrapolate",
                    field_name=name,
                    value=value,
                    bounds=(lo, hi),
                )

    def in_range(self, *, tile_px: int, n_frames: int, patch_area_px: int) -> bool:
        try:
            self.check_range(
                tile_px=tile_px, n_frames=n_frames, patch_area_px=patch_area_px
            )
        except CalibrationOutOfRange:
            return False
        return True


# ---------------------------------------------------------------------------
# Estimate from a profile
# ---------------------------------------------------------------------------

def estimate_components(
    profile: CalibrationProfile,
    *,
    tile_px: int,
    n_frames: int,
    patch_area_px: int,
    channels: int = 3,
) -> dict[str, int]:
    """Return the linear component decomposition for ONE in-range observation.

    Raises :class:`CalibrationOutOfRange` outside the declared range. The result
    is a plain ``{name: integer bytes}`` mapping; no margin is applied here.
    """
    profile.validate()
    ch = require_int(channels, "channels", minimum=1)
    profile.check_range(
        tile_px=tile_px, n_frames=n_frames, patch_area_px=patch_area_px
    )
    c = profile.coefficient_map
    tile_px = int(tile_px)
    n_frames = int(n_frames)
    patch_area_px = int(patch_area_px)
    return {
        "aligned_input_bytes": (
            c["aligned_input_bytes_per_px_ch_frame"] * patch_area_px * ch * n_frames
        ),
        "process_baseline_bytes": c["process_baseline_bytes"],
        "rejection_mask_bytes": (
            c["rejection_mask_bytes_per_px_ch_frame"] * patch_area_px * ch
        ),
        "tile_fixed_overhead_bytes": c["tile_fixed_overhead_bytes"],
        "tile_workspace_bytes": (
            c["tile_workspace_bytes_per_px_ch_frame"] * tile_px * tile_px * ch
        ),
    }


def calibrated_estimate(
    profile: CalibrationProfile,
    *,
    tile_px: int,
    n_frames: int,
    patch_area_px: int,
    channels: int = 3,
) -> EstimateEnvelope:
    """Build an :class:`EstimateEnvelope` (method ``calibrated``) from a profile."""
    components = estimate_components(
        profile,
        tile_px=tile_px,
        n_frames=n_frames,
        patch_area_px=patch_area_px,
        channels=channels,
    )
    return make_estimate(
        components,
        method="calibrated",
        calibration_id=profile.calibration_id,
    )


def static_bound_estimate(
    components: Mapping[str, int], *, calibration_id: str | None = None
) -> EstimateEnvelope:
    """Declared conservative static bound (method ``static_bound``).

    The caller owns every byte here; this function adds no coefficient.
    """
    return make_estimate(
        components, method="static_bound", calibration_id=calibration_id
    )


# ---------------------------------------------------------------------------
# Registry
# ---------------------------------------------------------------------------

class ProfileRegistry:
    """Immutable lookup by ``(phase, backend, host_id, path)``.

    Duplicate scopes are refused at registration. A matching but non-admissible
    profile raises :class:`CalibrationNotAdmissible` instead of being silently
    ignored, so an OOC profile can never masquerade as a product calibration.
    """

    def __init__(self, profiles: Iterable[CalibrationProfile] = ()):
        table: dict[tuple[str, str, str, str], CalibrationProfile] = {}
        for profile in profiles:
            profile.validate()
            key = (profile.phase, profile.backend, profile.host_id, profile.path)
            if key in table:
                raise InvalidProfile(
                    f"duplicate calibration scope {key}: "
                    f"{table[key].calibration_id!r} and {profile.calibration_id!r}"
                )
            table[key] = profile
        self._table = MappingProxyType(table)

    def __len__(self) -> int:
        return len(self._table)

    @property
    def profiles(self) -> tuple[CalibrationProfile, ...]:
        return tuple(
            self._table[k]
            for k in sorted(self._table)
        )

    def lookup(
        self,
        *,
        phase: str,
        backend: str,
        host_class: str,
        path: str,
        require_admissible: bool = True,
    ) -> CalibrationProfile | None:
        """Exact-scope lookup; ``None`` when no profile exists for that scope."""
        if phase not in PHASES:
            raise InvalidProfile(f"unknown phase {phase!r}")
        if backend not in BACKENDS:
            raise InvalidProfile(f"unknown backend {backend!r}")
        _require_nonempty_str(host_class, "host_class")
        _require_nonempty_str(path, "path")
        profile = self._table.get((phase, backend, host_class, path))
        if profile is None:
            return None
        if require_admissible and not profile.product_admissible:
            raise CalibrationNotAdmissible(
                f"profile {profile.calibration_id!r} at path {path!r} is not "
                "admissible on the product path"
            )
        return profile


# ``SCHEMA as _ENVELOPE_SCHEMA`` is imported to assert the coupling at import
# time (a future rename must not silently break the two modules).
assert _ENVELOPE_SCHEMA == "zemosaic-r55-resource-envelope-v1"
