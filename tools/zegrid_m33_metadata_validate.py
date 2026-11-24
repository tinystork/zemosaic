"""M33 metadata validator — read-only, canonical, ZERO FITS hashing.

Purpose
-------
Before any M33 evidence bundle is reused (gauge import into a NEW run,
resume, or P1 qualification), the four durable metadata artifacts
(``gauge.json``, ``run.json``, ``geometry.json``, ``state.json``) must be
checked with the PRODUCT validators, not with an ad-hoc script. This tool runs
exactly those validators and writes nothing into the bundle:

* ``run_store.validate_run_identity``            — run.json identity contract;
* ``run_artifacts.read_gauge_artifact``          — strict JSON, self-digest,
                                                   gauge schema/vectors;
* ``run_artifacts.read_geometry_artifact``       — self-digest, schema,
                                                   run_id/identity binding;
* ``run_artifacts.precheck_gauge_import``        — the FAIL-FAST import gate:
                                                   artifact<->identity binding,
                                                   frame ids/order/role,
                                                   canvas, photometric
                                                   parameters (backend
                                                   EXCLUDED) and the gauge
                                                   fingerprint. It reads NO
                                                   FITS and hashes NO frame.

The full import (``run_artifacts.import_gauge_artifact``) additionally
requires the per-frame ``content_sha256``/``nbytes`` rows, i.e. one read pass
over the corpus. This tool never performs it and never starts a gauge
recomputation: a refused precheck is terminal until the identity is fixed.

Usage
-----
    python tools/zegrid_m33_metadata_validate.py --bundle-dir <dir> \
        [--out report.json] \
        [--expect gauge.json=<sha256> --expect run.json=<sha256> ...]

Exit codes: 0 = every check passed; 2 = refused (fail closed); 3 = usage/bug.

Safety: the bundle directory is opened READ-ONLY. Nothing is created, renamed,
repaired or normalised. Never run it against a live run's directory while that
run is executing (the durable store owns its files then).
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path

NAME_GAUGE = "gauge.json"
NAME_RUN = "run.json"
NAME_GEOMETRY = "geometry.json"
NAME_STATE = "state.json"

# ---------------------------------------------------------------------------
# checkout guard — ALWAYS validate with THIS checkout's product code
# ---------------------------------------------------------------------------
# The qualification venv is an editable install of another tree. Without the
# guard, ``import zemosaic`` resolved to that other checkout, i.e. the
# validator would judge the evidence with FOREIGN validators. The guard pins
# ``<repo>/src`` first and refuses (exit 4) if the imported module is outside
# this checkout.

CHECKOUT_REFUSAL_RC = 4


def _refuse_foreign_checkout(detail: str):
    sys.stderr.write(
        "zegrid_m33_metadata_validate: REFUSED — cannot pin the product code "
        f"of this checkout: {detail}\n"
        "Nothing was validated and no report was written. Remedy: run the tool "
        "from the worktree root (the guard pins ./src) or fix the "
        "virtualenv's editable install. A canonical validation must never run "
        "on foreign product code.\n"
    )
    raise SystemExit(CHECKOUT_REFUSAL_RC)


def _pin_checkout_source():
    """Pin ``sys.path`` to this checkout's ``src`` and prove it was used."""
    root = Path(__file__).resolve().parents[1]
    src = (root / "src").resolve()
    marker = src / "zemosaic" / "__init__.py"
    if not marker.is_file():
        _refuse_foreign_checkout(
            f"no local package at {marker} (this file is not inside a ZeMosaic "
            "checkout)"
        )
    removed = []
    for entry in list(sys.path):
        try:
            candidate = (Path(entry) if entry else Path.cwd()).resolve()
        except (OSError, RuntimeError):
            continue
        if candidate == src:
            continue
        if (candidate / "zemosaic" / "__init__.py").is_file():
            removed.append(str(entry))
            sys.path.remove(entry)
    for finder in list(sys.meta_path):
        owner = type(finder).__module__ or ""
        if "editable" in owner.lower() and "zemosaic" in owner.lower():
            sys.meta_path.remove(finder)
            removed.append(f"meta_path:{owner}")
    sys.path.insert(0, str(src))
    import zemosaic

    actual = Path(zemosaic.__file__).resolve()
    if src not in actual.parents:
        _refuse_foreign_checkout(
            f"the imported package is {actual}, outside {src}"
        )
    return {
        "repo_root": str(root),
        "src": str(src),
        "zemosaic_file": str(actual),
        "locked": True,
        "foreign_sys_path_entries_removed": removed,
    }


CODE_SOURCE = _pin_checkout_source()


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _check(checks: list, ident: str, func) -> None:
    """Run one product validator; never raise past this boundary."""
    try:
        detail = func()
    except Exception as exc:  # noqa: BLE001 - a refusal is data, not a crash
        checks.append({
            "id": ident,
            "ok": False,
            "error_type": type(exc).__name__,
            "detail": str(exc),
        })
        return
    checks.append({"id": ident, "ok": True, "detail": detail})


def validate(*, bundle_dir: Path, expected_hashes: dict | None = None) -> dict:
    from zemosaic.core.zegrid import run_artifacts as zart
    from zemosaic.core.zegrid import run_store as zrs

    root = Path(bundle_dir)
    report: dict = {
        "schema": "zemosaic-m33-metadata-validation-v1",
        "bundle_dir": str(root),
        "mode": "read_only_metadata_no_fits_hashing",
        "code_source": dict(CODE_SOURCE),
        "checks": [],
        "warnings": [],
    }
    checks = report["checks"]

    paths = {
        "gauge.json": root / NAME_GAUGE,
        "run.json": root / NAME_RUN,
        "geometry.json": root / NAME_GEOMETRY,
        "state.json": root / NAME_STATE,
    }
    missing = [name for name, path in paths.items() if not path.is_file()]
    if missing:
        checks.append({
            "id": "files.present",
            "ok": False,
            "detail": f"missing artifact(s) {missing} in {root}",
        })
        report["ok"] = False
        report["failed_checks"] = ["files.present"]
        report["note"] = (
            "a missing durable artifact is terminal: nothing is created, "
            "repaired or re-derived by this validator"
        )
        return report
    checks.append({"id": "files.present", "ok": True, "detail": sorted(paths)})

    report["file_sha256"] = {
        name: _sha256_file(path) for name, path in paths.items()
    }
    if expected_hashes:
        drift = {}
        for name, expected in expected_hashes.items():
            observed = report["file_sha256"].get(name)
            if observed is None:
                drift[name] = {"expected": expected, "observed": None}
            elif observed.lower() != str(expected).lower():
                drift[name] = {"expected": expected, "observed": observed}
        checks.append({
            "id": "files.expected_sha256",
            "ok": not drift,
            "detail": drift or "all expected file digests match",
        })

    identity = {}

    def _identity():
        payload = json.loads(paths["run.json"].read_text(encoding="utf-8"))
        identity.update(zrs.validate_run_identity(payload))
        contract = identity["science_contract"]
        return {
            "run_id": identity["run_id"],
            "schema_version": identity["schema_version"],
            "frames": len(contract["inputs"]["frames"]),
            "grid": contract.get("grid"),
            "halo": contract.get("halo"),
            "canvas": contract.get("canvas"),
            "backend": contract.get("backend"),
            "gauge_mode": (contract.get("gauge") or {}).get("mode"),
            "gauge_sha256": (contract.get("gauge") or {}).get("sha256"),
            "reference_frame_id": (
                contract.get("gauge") or {}
            ).get("reference_frame_id"),
        }

    _check(checks, "run.identity", _identity)
    if not identity:
        report["ok"] = False
        report["failed_checks"] = ["run.identity"]
        return report

    _check(
        checks,
        "gauge.artifact",
        lambda: {
            "run_id": zart.read_gauge_artifact(
                root, identity=identity
            ).get("run_id"),
            "gauge_sha256": zart.read_gauge_artifact(
                root, identity=identity
            ).get("gauge_sha256"),
        },
    )

    def _precheck():
        contract = identity["science_contract"]
        rows = contract["inputs"]["frames"]
        provenance = zart.precheck_gauge_import(
            paths["gauge.json"],
            origin_identity=identity,
            origin_run_json_path=str(paths["run.json"]),
            expected_frame_ids=[str(row["frame_id"]) for row in rows],
            expected_canvas=contract["canvas"],
            expected_science_params=dict(
                (contract.get("science") or {}).get("params") or {}
            ),
            expected_mode=str((contract.get("gauge") or {}).get("mode")),
        )
        report["gauge_import_precheck"] = dict(provenance)
        return {
            "section": "precheck_gauge_import",
            "fits_read": 0,
            "frame_hashes_computed": 0,
            "gauge_sha256": provenance["gauge_sha256"],
            "origin_run_id": provenance["origin_run_id"],
            "frame_count": provenance["frame_count"],
            "mode": provenance["mode"],
        }

    _check(checks, "gauge.import_precheck_no_fits", _precheck)

    _check(
        checks,
        "geometry.artifact",
        lambda: {
            "schema": zart.read_geometry_artifact(
                root, identity=identity
            ).get("schema"),
        },
    )

    def _state():
        payload = json.loads(paths["state.json"].read_text(encoding="utf-8"))
        gauge_sha = payload.get("gauge_sha256")
        contract_gauge = (identity["science_contract"].get("gauge") or {}).get(
            "sha256"
        )
        run_id = payload.get("run_id")
        mismatches = {}
        if run_id is not None and str(run_id) != str(identity["run_id"]):
            mismatches["run_id"] = {
                "state": run_id, "identity": identity["run_id"],
            }
        if gauge_sha is not None and contract_gauge is not None and str(
            gauge_sha
        ) != str(contract_gauge):
            mismatches["gauge_sha256"] = {
                "state": gauge_sha, "identity": contract_gauge,
            }
        if mismatches:
            raise ValueError(
                "state.json disagrees with the run identity: "
                + json.dumps(mismatches, sort_keys=True)
            )
        return {
            "run_id_matches": True,
            "gauge_sha256_matches": True,
            "mismatches": {},
            "finalized": payload.get("finalized"),
            "outputs": sorted(payload.get("outputs") or ()),
            "working_set": payload.get("working_set"),
            "backend": payload.get("backend"),
        }

    _check(checks, "state.cross_reference", _state)

    failed = [item for item in checks if not item["ok"]]
    report["ok"] = not failed
    report["failed_checks"] = [item["id"] for item in failed]
    report["note"] = (
        "a passing precheck authorises NO import by itself: the full import "
        "(run_artifacts.import_gauge_artifact) still verifies every frame's "
        "content_sha256/nbytes, and a refusal is terminal (never repaired, "
        "never followed by an automatic 30 h gauge recomputation)"
    )
    return report


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(
        description=(
            "Read-only canonical metadata validation of one ZeMosaic M33 "
            "evidence bundle (no FITS hashing, no gauge recomputation)."
        )
    )
    parser.add_argument("--bundle-dir", required=True)
    parser.add_argument("--out", default=None)
    parser.add_argument(
        "--expect",
        action="append",
        default=[],
        metavar="NAME=SHA256",
        help=(
            "expected digest for gauge.json|run.json|geometry.json|"
            "state.json (repeatable); drift is a refusal"
        ),
    )
    args = parser.parse_args(argv)

    expected = {}
    for item in args.expect:
        if "=" not in item:
            parser.error(f"--expect needs name=sha256, got {item!r}")
        name, value = item.split("=", 1)
        if name not in (NAME_GAUGE, NAME_RUN, NAME_GEOMETRY, NAME_STATE):
            parser.error(f"--expect unsupported artifact name {name!r}")
        expected[name] = value.strip()

    report = validate(
        bundle_dir=Path(args.bundle_dir), expected_hashes=expected or None
    )
    text = json.dumps(report, indent=2, sort_keys=True) + "\n"
    if args.out:
        target = Path(args.out)
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(text, encoding="utf-8")
    else:
        sys.stdout.write(text)
    return 0 if report.get("ok") else 2


if __name__ == "__main__":
    raise SystemExit(main())
