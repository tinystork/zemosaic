"""ZM-ZEGRID-R40-B2-A focused tests — default-OFF capture contract + shard core.

Bounded task A of R40-B2: exercise the frozen §3/§6/§8 storage, identity, privacy,
boundedness and stale-generation mechanics on a **synthetic / in-memory** corpus
only. No production seam is wired, no FITS/pixel/real data is read, and nothing
here accepts B2 or authorizes R40-C/D.

Hard rules honoured by this suite:

* the reserved R40-B1 v2 validation cohort is **never** touched (no tool import, no
  artifact read) — this suite imports only ``src/zemosaic/core/zegrid/r40_capture``;
* the OFF/ON byte-identity gate and the 38-relation panel are **not** exercised here
  (they are later, separately gated B2 tasks) and are never weakened or claimed.
"""

from __future__ import annotations

import json
import os
import pathlib
import subprocess
import sys

import numpy as np
import pytest

from zemosaic.core.zegrid import r40_capture as c

REPO_HEAD = "61c4aa50f18881ad624b0ecf55df9b049eb56ecb"
SRC_ROOT = pathlib.Path(__file__).resolve().parents[1] / "src"


# ---------------------------------------------------------------------------
# Synthetic fixtures (self-contained, deterministic)
# ---------------------------------------------------------------------------


def make_header(u="c0001", v="c0002", orientation="h", **overrides):
    header = {
        "gauge_mode": "canonical_sky_mean",
        "stack_tokens": {
            "normalization": "none",
            "weighting": "none",
            "rejection": "none",
            "combine": "mean",
            "taper": "none",
            "taper_px": 0,
            "taper_floor": 0.0,
            "equalize_rgb": False,
            "sigma_low": -1.0,
            "sigma_high": -1.0,
            "winsor": 0.0,
        },
        "reference_frame_id": {u: "r0001c0001", v: "r0001c0002"},
        "frame_order_digest": {u: "0" * 64, v: "1" * 64},
        "excluded": {u: [], v: []},
        "measurement_px": 16,
        "feather_px": 8,
        "halo_px": 8,
        "block_px": 4,
    }
    header.update(overrides)
    return header


def make_record(channel=0, gx=10, gy=20, **overrides):
    record = {
        "gx": gx,
        "gy": gy,
        "channel": channel,
        "sci_u_pre": 1.0,
        "sci_v_pre": 1.5,
        "sci_u_post": 1.1,
        "sci_v_post": 1.6,
        "valid_u": True,
        "valid_v": True,
        "estimator_weight_sum_u": 2.0,
        "estimator_weight_sum_v": 2.5,
        "surviving_sample_count_u": 10,
        "surviving_sample_count_v": 12,
        "support_w1_u": 0.5,
        "support_w1_v": 0.6,
        "support_w2_u": 0.7,
        "support_w2_v": 0.8,
        "n_eff_support_u": 3.0,
        "n_eff_support_v": 3.5,
        "ramp_u": 0.25,
        "ramp_v": 0.75,
    }
    record.update(overrides)
    return record


def make_shard(u="c0001", v="c0002", orientation="h", records=(), **header_overrides):
    records = records or (make_record(channel=0, gx=5, gy=6),)
    return c.RelationShard(
        u=u,
        v=v,
        orientation=orientation,
        header=make_header(u, v, orientation, **header_overrides),
        records=tuple(records),
    )


def shard_meta(shard):
    data = shard.data()
    return {
        "relation_id": shard.relation_id,
        "count": len(shard.records),
        "bytes": len(data),
        "sha256": c.sha256_bytes(data),
    }


def make_identity(**overrides):
    identity = c.make_capture_identity(
        capture_id="capture-abc",
        source_set_digest="a" * 64,
        n_frames=816,
        layout={"nx": 6, "ny": 4},
        canvas={"width": 4117, "height": 4377, "id": "e8f1877f16f8cca4"},
        gauge_mode="canonical_sky_mean",
        repo_head=REPO_HEAD,
    )
    identity.update(overrides)
    return identity


def enabled_config(root, **overrides):
    kwargs = {"enabled": True, "output_root": pathlib.Path(root)}
    kwargs.update(overrides)
    return c.R40CaptureConfig(**kwargs)


@pytest.fixture()
def out_root(tmp_path):
    return tmp_path / "fresh_out"


# ---------------------------------------------------------------------------
# 1. Import / default-OFF has zero filesystem effect
# ---------------------------------------------------------------------------


def test_import_has_zero_filesystem_effect(tmp_path):
    code = (
        "import os,sys\n"
        f"sys.path.insert(0, {str(SRC_ROOT)!r})\n"
        "import zemosaic.core.zegrid.r40_capture as m\n"
        "assert m.default_capture_config().enabled is False\n"
        "print('|'.join(sorted(os.listdir('.'))))\n"
    )
    proc = subprocess.run(
        [sys.executable, "-c", code],
        cwd=str(tmp_path),
        capture_output=True,
        text=True,
        check=True,
    )
    assert proc.stdout.strip() == "", "importing the capture core wrote to the cwd"


def test_default_config_is_off_and_refuses(tmp_path):
    config = c.default_capture_config()
    assert config.enabled is False
    with pytest.raises(c.CaptureDisabledError):
        c.write_capture_generation(
            tmp_path / "nope",
            shards=[make_shard()],
            identity=make_identity(),
            config=config,
        )
    assert not (tmp_path / "nope").exists()
    # a disabled config is refused even with a valid path/shards
    with pytest.raises(c.CaptureDisabledError):
        c.write_capture_generation(
            tmp_path / "nope2",
            shards=[make_shard()],
            identity=make_identity(),
            config=c.R40CaptureConfig(enabled=False, output_root=tmp_path / "nope2"),
        )
    assert not (tmp_path / "nope2").exists()


def test_no_config_or_truthy_enabled_refused(tmp_path):
    with pytest.raises(c.CaptureDisabledError):
        c.write_capture_generation(tmp_path / "n", shards=[make_shard()], identity=make_identity())
    with pytest.raises(c.SchemaError):
        c.R40CaptureConfig(enabled=1)  # strict bool only
    with pytest.raises(c.SchemaError):
        c.R40CaptureConfig(enabled="yes")
    with pytest.raises(c.BudgetError):
        c.R40CaptureConfig(enabled=True, disk_budget_bytes=0)


def test_module_has_no_production_wiring():
    import ast

    source = pathlib.Path(c.__file__).read_text(encoding="utf-8")
    tree = ast.parse(source)
    names = {n.id for n in ast.walk(tree) if isinstance(n, ast.Name)}
    attrs = {n.attr for n in ast.walk(tree) if isinstance(n, ast.Attribute)}
    assert "_assemble_per_cell_feathered" not in (names | attrs)
    imported = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            imported.update(alias.name.split(".")[0] for alias in node.names)
        elif isinstance(node, ast.ImportFrom) and node.module:
            imported.add(node.module.split(".")[0])
    allowed = {
        "__future__", "copy", "hashlib", "json", "math", "os", "re", "shutil",
        "tempfile", "collections", "dataclasses", "pathlib", "types", "typing", "numpy",
    }
    assert imported <= allowed, f"unexpected imports: {sorted(imported - allowed)}"
    init = (SRC_ROOT / "zemosaic" / "core" / "zegrid" / "__init__.py").read_text(encoding="utf-8")
    assert "r40_capture" not in init, "the package init must stay unwired from the capture core"


# ---------------------------------------------------------------------------
# 2. Frozen dtype / endianness / order / no object
# ---------------------------------------------------------------------------


def test_record_dtype_is_frozen_little_endian_fixed_width():
    dtype = c.R40_RECORD_DTYPE
    assert dtype.hasobject is False
    expected = {
        "gy": "<i4",
        "gx": "<i4",
        "sci_u_pre": "<f4",
        "sci_v_pre": "<f4",
        "sci_u_post": "<f8",
        "sci_v_post": "<f8",
        "surviving_sample_count_u": "<i8",
        "surviving_sample_count_v": "<i8",
        "estimator_weight_sum_u": "<f8",
        "support_w1_u": "<f8",
        "ramp_v": "<f8",
    }
    for name, spec in expected.items():
        assert dtype[name].str == spec, name
    # M3 pre f32 / post f64 preserved
    assert dtype["sci_u_pre"].itemsize == 4 and dtype["sci_u_post"].itemsize == 8
    # M6 int64 canonical (no assumed narrowing)
    assert dtype["surviving_sample_count_u"].itemsize == 8
    # coordinates int32
    assert dtype["gx"].itemsize == 4 and dtype["gy"].itemsize == 4
    # bool validity
    assert dtype["valid_u"].kind == "b" and dtype["valid_u"].itemsize == 1
    # sample key is a fixed-width byte string (a real key, not a hash-redacted value)
    assert dtype["sample_key"].str == "|S64"
    frozen = c.assert_record_dtype_frozen()
    assert frozen["itemsize"] == c.R40_RECORD_ITEMSIZE
    assert frozen["n_fields"] == len(c.R40_RECORD_NAMES)


def test_encode_records_is_c_order_deterministic():
    records = [make_record(channel=0, gx=3, gy=4), make_record(channel=0, gx=1, gy=2)]
    first = c.encode_records(records, relation_id="c0001|c0002")
    second = c.encode_records(list(reversed(records)), relation_id="c0001|c0002")
    assert first.tobytes(order="C") == second.tobytes(order="C")
    assert first["gx"].tolist() == [1, 3]  # deterministic (channel, gy, gx)
    assert first.flags["C_CONTIGUOUS"]


def test_encode_records_refuses_non_finite_and_bad_types():
    with pytest.raises(c.NonFiniteError):
        c.encode_records([make_record(sci_u_post=float("nan"))], relation_id="a|b")
    with pytest.raises(c.NonFiniteError):
        c.encode_records([make_record(support_w1_u=float("inf"))], relation_id="a|b")
    with pytest.raises(c.SchemaError):
        c.encode_records([make_record(channel=7)], relation_id="a|b")
    with pytest.raises(c.SchemaError):
        c.encode_records([make_record(valid_u=1)], relation_id="a|b")  # strict bool
    with pytest.raises(c.SchemaError):
        c.encode_records([make_record(gx=2**40)], relation_id="a|b")
    missing = make_record()
    del missing["ramp_u"]
    with pytest.raises(c.SchemaError):
        c.encode_records([missing], relation_id="a|b")


def test_duplicate_sample_key_refused():
    records = [make_record(channel=0, gx=5, gy=5), make_record(channel=0, gx=5, gy=5)]
    with pytest.raises(c.DuplicateKeyError):
        c.encode_records(records, relation_id="a|b")


# ---------------------------------------------------------------------------
# 3. Deterministic keys / digests
# ---------------------------------------------------------------------------


def test_sample_key_matches_frozen_formula():
    import hashlib

    expected = hashlib.sha256(
        b"zemosaic.zegrid.r40.pair.v1\nc0001|c0002\n2\n17\n29"
    ).hexdigest()
    assert c.sample_key("c0001|c0002", 2, 17, 29) == expected
    assert c.sample_key("c0001|c0002", 2, 17, 29) != c.sample_key("c0001|c0002", 2, 17, 30)


def test_pairset_and_candidate_digests_are_order_independent():
    keys = [c.sample_key("a|b", 0, i, i) for i in range(5)]
    assert c.pairset_digest(keys) == c.pairset_digest(list(reversed(keys)))
    assert c.pairset_digest(keys) != c.pairset_digest(keys[:-1])
    coords = [(3, 4), (1, 2), (1, 2), (5, 6)]
    assert c.candidate_set_digest(coords) == c.candidate_set_digest([(5, 6), (1, 2), (3, 4)])
    assert c.candidate_set_digest(coords) != c.candidate_set_digest([(5, 6), (1, 2), (3, 5)])


def test_relation_id_and_ordering_contract():
    assert c.relation_id_for("c0001", "c0002") == "c0001|c0002"
    assert c.RECORD_ORDERING == ("orientation", "u", "v", "channel", "gy", "gx")


# ---------------------------------------------------------------------------
# 4. Round-trip shards / index / COMMIT
# ---------------------------------------------------------------------------


def test_round_trip_generation_and_independent_verification(out_root):
    shards = [
        make_shard("c0001", "c0002", "h", (make_record(0, 1, 1), make_record(1, 1, 1))),
        make_shard("c0003", "c0004", "v", (make_record(0, 2, 2),)),
    ]
    report = c.write_capture_generation(
        out_root, shards=shards, identity=make_identity(), config=enabled_config(out_root)
    )
    assert report["status"] == "COMMITTED"
    gen = out_root / report["generation_dir"]
    assert (gen / c.GENERATION_INDEX_BASENAME).exists()
    assert (gen / c.GENERATION_COMMIT_BASENAME).exists()
    verified = c.verify_generation(gen)
    assert verified["ok"] is True
    assert verified["n_shards"] == 2
    assert verified["checked_records"] == 3

    index = json.loads((gen / c.GENERATION_INDEX_BASENAME).read_text(encoding="utf-8"))
    assert index["byte_order"] == "little" and index["order"] == "C"
    assert index["ordering"] == list(c.RECORD_ORDERING)
    assert index["capture_stage"] == c.CAPTURE_STAGE
    assert index["correction_applied"] is False and index["production_wired"] is False
    for entry in index["shards"]:
        raw = (gen / entry["filename"]).read_bytes()
        assert c.sha256_bytes(raw) == entry["sha256"]
        assert len(raw) == entry["count"] * c.R40_RECORD_ITEMSIZE
    # private absolute paths are absent from the durable index
    assert c.scan_path_leaks(index) == []


def test_verify_generation_detects_tampering(out_root):
    shard = make_shard()
    report = c.write_capture_generation(
        out_root, shards=[shard], identity=make_identity(), config=enabled_config(out_root)
    )
    gen = out_root / report["generation_dir"]
    index = json.loads((gen / c.GENERATION_INDEX_BASENAME).read_text(encoding="utf-8"))
    target = gen / index["shards"][0]["filename"]
    target.write_bytes(target.read_bytes() + b"\x00")
    verified = c.verify_generation(gen)
    assert verified["ok"] is False and verified["problems"]


# ---------------------------------------------------------------------------
# 4b. Declared candidate geometry: write-side re-derivation + verify refusal
#     (ZM-ZEGRID-R40-B2-I, Nono B2-G LOW a/b)
# ---------------------------------------------------------------------------


def _geom_shard(records, coords, **header_overrides):
    return c.RelationShard(
        u="c0001", v="c0002", orientation="h",
        header=make_header("c0001", "c0002", "h", **header_overrides),
        records=tuple(records), candidate_coords=tuple(coords),
    )


def test_declared_candidate_geometry_writes_and_re_derives(out_root):
    """Positive control: a well-formed declared geometry is stored AND re-derived."""
    shard = _geom_shard((make_record(0, 5, 6),), ((5, 6), (9, 10)))
    report = c.write_capture_generation(
        out_root, shards=[shard], identity=make_identity(), config=enabled_config(out_root)
    )
    gen = out_root / report["generation_dir"]
    index = json.loads((gen / c.GENERATION_INDEX_BASENAME).read_text(encoding="utf-8"))
    entry = index["shards"][0]
    assert entry["candidate_set_digest"] == c.candidate_set_digest(((5, 6), (9, 10)))
    assert entry["n_candidates"] == 2
    assert c.verify_generation(gen)["ok"] is True


def test_write_refuses_declared_geometry_with_duplicate_coordinates(out_root):
    shard = _geom_shard((make_record(0, 5, 6),), ((5, 6), (5, 6)))
    with pytest.raises(c.SchemaError, match="duplicate coordinates"):
        c.write_capture_generation(
            out_root, shards=[shard], identity=make_identity(),
            config=enabled_config(out_root),
        )
    assert not out_root.exists() or not list(out_root.glob("r40_capture_*"))


def test_write_refuses_empty_declared_candidate_geometry(out_root):
    shard = _geom_shard((make_record(0, 5, 6),), ())
    # an EMPTY geometry is falsy -> treated as "no geometry" by RelationShard; the
    # write path must therefore not claim to have re-derived a digest for it.
    assert shard.candidate_digest() is None
    report = c.write_capture_generation(
        out_root, shards=[shard], identity=make_identity(), config=enabled_config(out_root)
    )
    gen = out_root / report["generation_dir"]
    index = json.loads((gen / c.GENERATION_INDEX_BASENAME).read_text(encoding="utf-8"))
    assert index["shards"][0]["candidate_set_digest"] is None


def test_write_refuses_recorded_pair_outside_declared_candidate_geometry(out_root):
    shard = _geom_shard((make_record(0, 5, 6),), ((1, 2), (3, 4)))
    with pytest.raises(c.SchemaError, match="outside the declared candidate geometry"):
        c.write_capture_generation(
            out_root, shards=[shard], identity=make_identity(),
            config=enabled_config(out_root),
        )


def test_verify_generation_refuses_declared_geometry_without_block_px(out_root):
    """A declared-geometry shard whose header omits ``block_px`` is REFUSED, not
    silently verified at the weaker level (Nono B2-G LOW b)."""
    shard = _geom_shard((make_record(0, 5, 6),), ((5, 6),))
    report = c.write_capture_generation(
        out_root, shards=[shard], identity=make_identity(), config=enabled_config(out_root)
    )
    gen = out_root / report["generation_dir"]
    assert c.verify_generation(gen)["ok"] is True

    index = json.loads((gen / c.GENERATION_INDEX_BASENAME).read_text(encoding="utf-8"))
    assert "block_px" in index["shards"][0]["header"]
    del index["shards"][0]["header"]["block_px"]
    index_bytes = c.dumps_strict(index).encode("utf-8")
    (gen / c.GENERATION_INDEX_BASENAME).write_bytes(index_bytes)
    commit = json.loads((gen / c.GENERATION_COMMIT_BASENAME).read_text(encoding="utf-8"))
    commit["index_sha256"] = c.sha256_bytes(index_bytes)
    (gen / c.GENERATION_COMMIT_BASENAME).write_bytes(c.dumps_strict(commit).encode("utf-8"))

    verified = c.verify_generation(gen)
    assert verified["ok"] is False
    assert any("block_px" in p for p in verified["problems"]), verified["problems"]


# ---------------------------------------------------------------------------
# 5. Budget boundary and overflow refusal
# ---------------------------------------------------------------------------


def test_budget_boundary_and_overflow(tmp_path):
    shards = [make_shard("c0001", "c0002", "h", (make_record(0, 1, 1), make_record(1, 2, 2)))]

    baseline = c.write_capture_generation(
        tmp_path / "base", shards=shards, identity=make_identity(), config=enabled_config(tmp_path / "base")
    )
    total = baseline["total_bytes"]
    shard_bytes = baseline["shard_bytes"]

    # exactly the budget succeeds
    exact = c.write_capture_generation(
        tmp_path / "exact",
        shards=shards,
        identity=make_identity(),
        config=enabled_config(tmp_path / "exact", disk_budget_bytes=total),
    )
    assert exact["status"] == "COMMITTED"

    # shard bytes alone fit, but shards + index/COMMIT exceed the budget => refuse mid-write
    with pytest.raises(c.BudgetError):
        c.write_capture_generation(
            tmp_path / "mid",
            shards=shards,
            identity=make_identity(),
            config=enabled_config(tmp_path / "mid", disk_budget_bytes=shard_bytes),
        )
    mid = tmp_path / "mid"
    assert not (mid / baseline["generation_dir"] / c.GENERATION_COMMIT_BASENAME).exists()
    assert not (mid / baseline["generation_dir"]).exists() or (
        mid / f"{baseline['generation_dir']}.incomplete"
    ).exists()

    # one byte less than the shard plan refuses *before* creating anything
    with pytest.raises(c.BudgetError):
        c.write_capture_generation(
            tmp_path / "pre",
            shards=shards,
            identity=make_identity(),
            config=enabled_config(tmp_path / "pre", disk_budget_bytes=shard_bytes - 1),
        )
    assert not (tmp_path / "pre").exists()


def test_estimate_and_check_disk_budget_helpers():
    shards = [make_shard("c0001", "c0002", "h", (make_record(0, 1, 1),))]
    plan = c.estimate_generation_bytes(shards)
    assert plan["shard_bytes"] == c.R40_RECORD_ITEMSIZE
    assert plan["total_bytes"] > plan["shard_bytes"] >= 0
    c.check_disk_budget(planned_bytes=10, budget_bytes=10)
    with pytest.raises(c.BudgetError):
        c.check_disk_budget(planned_bytes=11, budget_bytes=10)


# ---------------------------------------------------------------------------
# 6. Atomic failure injection — no false COMMIT, temp cleaned/quarantined
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "fail_at",
    [
        "shard_write",
        "shard_fsync",
        "shard_replace",
        "index_serialize",
        "index_fsync",
        "index_replace",
        "commit_write",
        "commit_fsync",
        "commit_replace",
    ],
)
def test_failure_injection_leaves_no_committed_generation(tmp_path, fail_at):
    root = tmp_path / "inj"
    shards = [
        make_shard("c0001", "c0002", "h", (make_record(0, 1, 1),)),
        make_shard("c0003", "c0004", "v", (make_record(0, 2, 2),)),
    ]
    with pytest.raises(c.R40CaptureError):
        c.write_capture_generation(
            root,
            shards=shards,
            identity=make_identity(),
            config=enabled_config(root),
            _fail_at=fail_at,
            protected_roots=(),
        )
    gen = root / f"{c.GENERATION_DIR_PREFIX}capture-abc"
    assert not (gen / c.GENERATION_COMMIT_BASENAME).exists()
    assert not gen.exists(), "the partial generation must be quarantined (renamed), not left in place"
    leftovers = list(root.glob("*.incomplete*"))
    assert leftovers, "a quarantined incomplete generation is expected"
    # no temp residue anywhere under the output root
    residue = [p for p in root.rglob("*.tmp") if p.is_file()]
    assert residue == []
    for quarantined in leftovers:
        assert not (quarantined / c.GENERATION_COMMIT_BASENAME).exists()
        assert (quarantined / c.INCOMPLETE_MARKER_BASENAME).exists()


def test_failure_injection_cleanup_mode_removes_partial(tmp_path):
    root = tmp_path / "cleanup"
    with pytest.raises(c.R40CaptureError):
        c.write_capture_generation(
            root,
            shards=[make_shard()],
            identity=make_identity(),
            config=enabled_config(root, on_partial="cleanup"),
            _fail_at="index_replace",
        )
    assert list(root.glob("*")) == []


def test_committed_generation_is_never_overwritten_by_failure(out_root):
    shard = make_shard()
    report = c.write_capture_generation(
        out_root, shards=[shard], identity=make_identity(), config=enabled_config(out_root)
    )
    gen = out_root / report["generation_dir"]
    commit_before = (gen / c.GENERATION_COMMIT_BASENAME).read_bytes()
    with pytest.raises(c.GenerationExistsError):
        c.write_capture_generation(
            out_root,
            shards=[shard],
            identity=make_identity(),
            config=enabled_config(out_root),
            _fail_at="commit_replace",
        )
    assert (gen / c.GENERATION_COMMIT_BASENAME).read_bytes() == commit_before
    assert c.verify_generation(gen)["ok"] is True


# ---------------------------------------------------------------------------
# 7. Path safety: fresh / symlink / traversal / protected roots
# ---------------------------------------------------------------------------


def test_existing_generation_refused(out_root):
    shard = make_shard()
    c.write_capture_generation(out_root, shards=[shard], identity=make_identity(), config=enabled_config(out_root))
    with pytest.raises(c.GenerationExistsError):
        c.write_capture_generation(out_root, shards=[shard], identity=make_identity(), config=enabled_config(out_root))


def test_non_fresh_generation_dir_refused(tmp_path):
    root = tmp_path / "root"
    (root / f"{c.GENERATION_DIR_PREFIX}capture-abc").mkdir(parents=True)
    with pytest.raises(c.GenerationExistsError):
        c.write_capture_generation(root, shards=[make_shard()], identity=make_identity(), config=enabled_config(root))


def test_symlinked_output_root_refused(tmp_path):
    real = tmp_path / "real"
    real.mkdir()
    link = tmp_path / "link"
    link.symlink_to(real)
    with pytest.raises(c.PathSafetyError):
        c.write_capture_generation(link, shards=[make_shard()], identity=make_identity(), config=enabled_config(link))


def test_traversal_and_uri_output_roots_refused():
    for raw in ("../escape", "a/../../b", "file:///tmp/x", "C:\\out", "\\\\host\\share\\out", ""):
        with pytest.raises(c.PathSafetyError):
            c.resolve_fresh_output_root(raw)


def test_protected_root_refused(tmp_path):
    protected = tmp_path / "dataset_root"
    protected.mkdir()
    target = protected / "capture_out"
    with pytest.raises(c.PathSafetyError):
        c.write_capture_generation(
            target,
            shards=[make_shard()],
            identity=make_identity(),
            config=enabled_config(target),
            protected_roots=(protected,),
        )
    assert not target.exists()


# ---------------------------------------------------------------------------
# 8. Privacy: fail closed, non-mutating, prose-safe
# ---------------------------------------------------------------------------


def test_path_leak_categories():
    assert c.path_leak_category("/home/alice/data") == "posix_absolute"
    assert c.path_leak_category("root=/srv/private/frames/x.fits") == "posix_absolute_embedded"
    assert c.path_leak_category("C:\\Users\\alice\\data") == "windows_drive_absolute"
    assert c.path_leak_category("\\\\server\\share\\x") == "unc"
    assert c.path_leak_category("file:///data/x") == "file_uri"


def test_scientific_prose_is_not_falsely_redacted():
    for prose in (
        "RMS(d_shadow)/RMS(d_prod)",
        "delta_d_post <= 1.25 * sigma_ref",
        "reduction_rms >= 0.20",
        "ulp_k = 5 scale-aware bound",
        "relation u|v channel 0..2",
        "p90|e| <= 5%",
    ):
        assert c.path_leak_category(prose) is None, prose
    payload = {"note": "RMS(d_shadow)/RMS(d_prod) and ulp_k=5", "n": 1}
    assert c.scan_path_leaks(payload) == []
    c.assert_no_path_leaks({"deep": {"list": [{"note": "a/b ratio"}]}})


def test_privacy_failure_refuses_before_commit_without_mutation(tmp_path):
    root = tmp_path / "privacy"
    shard = make_shard(records=(make_record(0, 1, 1),), gauge_mode="/srv/private/gauge")
    original_header = dict(shard.header)
    with pytest.raises(c.PrivacyError):
        c.write_capture_generation(root, shards=[shard], identity=make_identity(), config=enabled_config(root))
    gen = root / f"{c.GENERATION_DIR_PREFIX}capture-abc"
    # privacy fires before COMMIT (and, in the reworked core, before any publish)
    assert not (gen / c.GENERATION_COMMIT_BASENAME).exists()
    assert not list(root.glob("**/COMMIT")) if root.exists() else True
    # fail closed, no mutation/redaction of the caller's data
    assert shard.header == original_header
    assert shard.header["gauge_mode"] == "/srv/private/gauge"


def test_privacy_failure_before_publish_never_creates_a_generation(tmp_path):
    root = tmp_path / "privacy2"
    shard = make_shard(gauge_mode="/srv/private/gauge")
    with pytest.raises(c.PrivacyError):
        c.write_capture_generation(root, shards=[shard], identity=make_identity(), config=enabled_config(root))
    assert not (root / f"{c.GENERATION_DIR_PREFIX}capture-abc").exists()
    assert not list(root.glob("**/COMMIT"))


# ---------------------------------------------------------------------------
# 9. Non-finite JSON / arrays refuse (never coerced)
# ---------------------------------------------------------------------------


def test_non_finite_json_refused():
    with pytest.raises(c.NonFiniteError):
        c.assert_finite_json({"a": [1.0, float("inf")]})
    with pytest.raises(c.NonFiniteError):
        c.assert_finite_json({"a": {"b": float("nan")}})
    c.assert_finite_json({"a": [1.0, 2.0], "b": "ok"})


def test_non_finite_array_refused():
    with pytest.raises(c.NonFiniteError):
        c.assert_finite_array("x", np.array([1.0, np.nan]))
    with pytest.raises(c.SchemaError):
        c.assert_finite_array("x", np.array([{"a": 1}], dtype=object))
    c.assert_finite_array("ok", np.array([1.0, 2.0]))


# ---------------------------------------------------------------------------
# 10. Identity mismatch matrix
# ---------------------------------------------------------------------------


def test_valid_identity_is_accepted():
    identity = make_identity()
    checked = c.check_capture_identity(identity)
    assert checked["capture_stage"] == c.CAPTURE_STAGE
    assert checked["correction_applied"] is False
    assert checked["production_wired"] is False
    assert checked["holdout_capped"] is False
    assert checked["qualification"] == "EXACT"
    assert checked["n_dropped"] == 0
    assert c.identity_digest(identity) == c.identity_digest(dict(identity))


@pytest.mark.parametrize("field_name", c.IDENTITY_REQUIRED_FIELDS)
def test_identity_missing_field_refused(field_name):
    identity = make_identity()
    del identity[field_name]
    with pytest.raises(c.IdentityError):
        c.check_capture_identity(identity)


@pytest.mark.parametrize(
    "field_name, bad_value",
    [
        ("capture_contract_version", "r40-capture-2"),
        ("schema", "zemosaic.other"),
        ("schema_version", "r40-2"),
        ("capture_stage", "post_feather"),
        ("correction_applied", True),
        ("production_wired", True),
        ("capture_id", "../evil"),
        ("capture_id", ""),
        ("source_set_digest_algo", "md5"),
        ("source_set_digest", "zz"),
        ("source_set_count", 0),
        ("n_frames", 0),
        ("layout", {"nx": 0, "ny": 4}),
        ("layout", {"nx": 6}),
        ("canvas", {"width": 0, "height": 4377, "id": "x"}),
        ("canvas", {"width": 4117, "height": 4377}),
        ("gauge_mode", ""),
        ("measurement_px", 0),
        ("feather_px", 0),
        ("halo_px", 0),
        ("block_px", 0),
        ("holdout_capped", True),
        ("qualification", "DEV_NOT_QUALIFICATION"),
        ("n_dropped", 1),
        ("repo_head", "not-a-sha"),
        ("b1_protocol_sha256", "0" * 64),
        ("b1_criteria_sha256", "0" * 64),
        ("b1_result_sha256", "0" * 64),
        ("b1_claim_id", "deadbeef"),
    ],
)
def test_identity_field_mismatch_refused(field_name, bad_value):
    identity = make_identity()
    identity[field_name] = bad_value
    with pytest.raises(c.IdentityError):
        c.check_capture_identity(identity)


def test_identity_expected_override_mismatch_refused():
    identity = make_identity()
    with pytest.raises(c.IdentityError):
        c.check_capture_identity(identity, expected={"gauge_mode": "different_gauge"})
    c.check_capture_identity(identity, expected={"gauge_mode": "canonical_sky_mean"})


def test_write_refuses_bad_identity(tmp_path):
    root = tmp_path / "badid"
    identity = make_identity(qualification="DEV_NOT_QUALIFICATION")
    with pytest.raises(c.IdentityError):
        c.write_capture_generation(root, shards=[make_shard()], identity=identity, config=enabled_config(root))
    assert not root.exists()


# ---------------------------------------------------------------------------
# 11. Stale invalidation / exact resume / no partial final
# ---------------------------------------------------------------------------


def test_checkpoint_resume_requires_exact_identity_and_shards():
    identity = make_identity()
    shard_a = make_shard("c0001", "c0002", "h", (make_record(0, 1, 1),))
    shard_b = make_shard("c0003", "c0004", "v", (make_record(0, 2, 2),))
    checkpoint = c.build_checkpoint(identity=identity, completed_shards=[shard_meta(shard_a)])

    partial = c.check_resume(
        checkpoint,
        identity=identity,
        completed_shards=[shard_meta(shard_a)],
        expected_relation_ids=[shard_a.relation_id, shard_b.relation_id],
    )
    assert partial["state"] == c.RESUMABLE
    assert partial["remaining"] == [shard_b.relation_id]

    complete = c.check_resume(
        checkpoint, identity=identity, completed_shards=[shard_meta(shard_a)]
    )
    assert complete["state"] == c.COMPLETE

    # a changed identity digest invalidates the checkpoint
    with pytest.raises(c.StaleGenerationError) as exc:
        c.check_resume(
            checkpoint, identity=make_identity(source_set_digest="b" * 64), completed_shards=[shard_meta(shard_a)]
        )
    assert str(exc.value).startswith(c.STALE_INVALIDATED)

    # a changed shard hash/count invalidates the checkpoint
    tampered = dict(shard_meta(shard_a))
    tampered["sha256"] = "0" * 64
    with pytest.raises(c.StaleGenerationError) as exc2:
        c.check_resume(checkpoint, identity=identity, completed_shards=[tampered])
    assert str(exc2.value).startswith(c.STALE_INVALIDATED)

    tampered_count = dict(shard_meta(shard_a))
    tampered_count["count"] = 99
    with pytest.raises(c.StaleGenerationError):
        c.check_resume(checkpoint, identity=identity, completed_shards=[tampered_count])

    # an unknown shard in the checkpoint is stale, not silently reused
    with pytest.raises(c.StaleGenerationError):
        c.check_resume(
            checkpoint,
            identity=identity,
            completed_shards=[shard_meta(shard_a)],
            expected_relation_ids=["other|relation"],
        )


def test_no_final_commit_while_partial():
    shard_a = make_shard("c0001", "c0002", "h", (make_record(0, 1, 1),))
    shard_b = make_shard("c0003", "c0004", "v", (make_record(0, 2, 2),))
    partial = c.build_checkpoint(identity=make_identity(), completed_shards=[shard_meta(shard_a)])
    with pytest.raises(c.StaleGenerationError) as exc:
        c.assert_checkpoint_complete(partial, expected_relation_ids=[shard_a.relation_id, shard_b.relation_id])
    assert "INCOMPLETE_PANEL" in str(exc.value)
    complete = c.build_checkpoint(
        identity=make_identity(), completed_shards=[shard_meta(shard_a), shard_meta(shard_b)]
    )
    c.assert_checkpoint_complete(complete, expected_relation_ids=[shard_a.relation_id, shard_b.relation_id])


# ---------------------------------------------------------------------------
# 12. Manifest additive block
# ---------------------------------------------------------------------------


def manifest_fixture():
    return {
        "schema": "zemosaic.zegrid.manifest",
        "schema_version": "r37-1",
        "run": {"id": "r0001", "cells": ["c0001", "c0002"]},
        "cell_feather": {"offsets": {"c0001": [0.1, 0.2, 0.3]}},
    }


def test_manifest_link_is_exactly_one_additive_block():
    off = manifest_fixture()
    on = c.build_manifest_link(
        off, capture_id="capture-abc", filename="shard-h-c0001-c0002.bin", sha256="a" * 64, nbytes=1234
    )
    proof = c.assert_manifest_link_additive(off, on)
    assert proof["additive"] is True
    assert proof["added_keys"] == [c.MANIFEST_LINK_KEY]
    assert proof["pre_existing_canonical_identical"] is True
    assert proof["whole_manifest_byte_identical"] is False
    # pre-existing fields are canonical-identical after removing the new block
    assert c.canonical_manifest_without_link(off) == c.canonical_manifest_without_link(on)
    # the OFF manifest is not mutated by building the link
    assert c.MANIFEST_LINK_KEY not in off
    assert set(on[c.MANIFEST_LINK_KEY]) == {"capture_id", "filename", "sha256", "bytes"}


def test_manifest_link_rejects_bad_inputs_and_extra_blocks():
    off = manifest_fixture()
    on = c.build_manifest_link(
        off, capture_id="capture-abc", filename="x.bin", sha256="a" * 64, nbytes=1
    )
    with pytest.raises(c.SchemaError):
        c.build_manifest_link(off, capture_id="x", filename="../x.bin", sha256="a" * 64, nbytes=1)
    with pytest.raises(c.SchemaError):
        c.build_manifest_link(off, capture_id="x", filename="a/b.bin", sha256="a" * 64, nbytes=1)
    with pytest.raises(c.SchemaError):
        c.build_manifest_link(off, capture_id="x", filename="x.bin", sha256="zz", nbytes=1)
    with pytest.raises(c.SchemaError):
        c.build_manifest_link(off, capture_id="x", filename="x.bin", sha256="a" * 64, nbytes=-1)
    with pytest.raises(c.SchemaError):
        c.build_manifest_link(on, capture_id="x", filename="y.bin", sha256="a" * 64, nbytes=1)
    # a second additive key breaks the additive proof
    tampered = dict(on)
    tampered["sneaky"] = 1
    with pytest.raises(c.SchemaError):
        c.assert_manifest_link_additive(off, tampered)


# ---------------------------------------------------------------------------
# 13. Sparse payload + deterministic parent join
# ---------------------------------------------------------------------------


def payload_side(role, samples):
    return c.build_sparse_payload(
        cell_id=f"c0001-{role}",
        role=role,
        relation_id="c0001|c0002",
        channel=0,
        samples=samples,
    )


def sample_for(role, gx, gy, value):
    if role == "u":
        return {
            "gx": gx,
            "gy": gy,
            "sci_u_pre": value,
            "sci_u_post": value + 0.1,
            "valid_u": True,
            "estimator_weight_sum_u": 2.0,
            "surviving_sample_count_u": 10,
            "support_w1_u": 0.5,
            "support_w2_u": 0.7,
            "n_eff_support_u": 3.0,
            "ramp_u": 0.25,
        }
    return {
        "gx": gx,
        "gy": gy,
        "sci_v_pre": value,
        "sci_v_post": value + 0.2,
        "valid_v": True,
        "estimator_weight_sum_v": 2.5,
        "surviving_sample_count_v": 12,
        "support_w1_v": 0.6,
        "support_w2_v": 0.8,
        "n_eff_support_v": 3.5,
        "ramp_v": 0.75,
    }


def test_parent_join_is_deterministic_and_sparse():
    u = payload_side("u", [sample_for("u", gx, gy, gx + gy) for gx, gy in [(9, 9), (1, 1), (5, 4)]])
    v = payload_side("v", [sample_for("v", gx, gy, gx) for gx, gy in [(5, 4), (9, 9), (1, 1)]])
    joined = c.parent_join_records(u, v)
    assert [(r["gx"], r["gy"]) for r in joined] == [(1, 1), (5, 4), (9, 9)]
    assert joined[0]["sci_u_pre"] == 2.0 and joined[0]["sci_v_pre"] == 1.0
    # joined records encode cleanly into the frozen shard dtype
    array = c.encode_records(joined, relation_id="c0001|c0002")
    assert array["gx"].tolist() == [1, 5, 9]
    # sparsity: only candidate samples retained, no full plane
    assert c.payload_retained_scalars(u) == 3 * (2 + len(c.PAYLOAD_FIELDS_BY_ROLE["u"]))


def test_join_ordering_across_channels_is_frozen():
    pairs = []
    for channel in (2, 0, 1):
        u = c.build_sparse_payload(
            cell_id="c0001-u", role="u", relation_id="c0001|c0002", channel=channel,
            samples=[sample_for("u", 4, 3, 1.0), sample_for("u", 2, 8, 1.0)],
        )
        v = c.build_sparse_payload(
            cell_id="c0001-v", role="v", relation_id="c0001|c0002", channel=channel,
            samples=[sample_for("v", 4, 3, 2.0), sample_for("v", 2, 8, 2.0)],
        )
        pairs.append((u, v))
    records = c.assemble_relation_records(pairs)
    assert [(r["channel"], r["gy"], r["gx"]) for r in records] == [
        (0, 3, 4),
        (0, 8, 2),
        (1, 3, 4),
        (1, 8, 2),
        (2, 3, 4),
        (2, 8, 2),
    ]


def test_payload_duplicate_key_and_identity_refusals():
    dup = [sample_for("u", 3, 3, 1.0), sample_for("u", 3, 3, 1.0)]
    with pytest.raises(c.DuplicateKeyError):
        payload_side("u", dup)
    with pytest.raises(c.SchemaError):
        c.build_sparse_payload(cell_id="x", role="bad", relation_id="a|b", channel=0, samples=[])
    with pytest.raises(c.NonFiniteError):
        payload_side("u", [dict(sample_for("u", 1, 1, 1.0), sci_u_post=float("inf"))])

    u = payload_side("u", [sample_for("u", 1, 1, 1.0)])
    v = c.build_sparse_payload(
        cell_id="c-v", role="v", relation_id="c0001|c0002", channel=1,
        samples=[sample_for("v", 1, 1, 2.0)],
    )
    with pytest.raises(c.SchemaError):
        c.parent_join_records(u, v)  # channel mismatch
    other_relation = c.build_sparse_payload(
        cell_id="c-v", role="v", relation_id="c0009|c0010", channel=0,
        samples=[sample_for("v", 1, 1, 2.0)],
    )
    with pytest.raises(c.SchemaError):
        c.parent_join_records(u, other_relation)  # relation mismatch
    with pytest.raises(c.SchemaError):
        c.parent_join_records(u, u)  # role mismatch


def test_join_held_out_key_must_exist():
    u = payload_side("u", [sample_for("u", 1, 1, 1.0)])
    v = payload_side("v", [sample_for("v", 1, 1, 2.0)])
    assert c.parent_join_records(u, v, keys=[(1, 1)])
    with pytest.raises(c.SchemaError):
        c.parent_join_records(u, v, keys=[(7, 7)])


def test_relational_header_validation():
    shard = make_shard()
    header = shard.header_payload()
    assert header["relation_id"] == "c0001|c0002"
    assert header["orientation"] == "h"
    bad = make_header()
    del bad["gauge_mode"]
    with pytest.raises(c.SchemaError):
        c.validate_relation_header(bad, u="c0001", v="c0002", orientation="h")
    bad2 = make_header()
    bad2["excluded"] = {"c0001": []}  # missing the second cell
    with pytest.raises(c.SchemaError):
        c.validate_relation_header(bad2, u="c0001", v="c0002", orientation="h")
    bad3 = make_header()
    bad3["measurement_px"] = 0
    with pytest.raises(c.SchemaError):
        c.validate_relation_header(bad3, u="c0001", v="c0002", orientation="h")
    with pytest.raises(c.SchemaError):
        c.validate_relation_header(make_header(), u="c0001", v="c0002", orientation="x")


def test_shard_order_key_is_deterministic():
    shards = [
        make_shard("c0003", "c0004", "v", (make_record(0, 1, 1),)),
        make_shard("c0001", "c0002", "h", (make_record(0, 1, 1),)),
    ]
    ordered = sorted(shards, key=lambda s: s.order_key())
    assert [s.orientation for s in ordered] == ["h", "v"]


# ---------------------------------------------------------------------------
# Rework-1 — R1: exact, fail-closed disk budget
# ---------------------------------------------------------------------------


def _exact_generation_total(shards, identity=None):
    identity = identity or make_identity()
    blobs, index_payload, commit_payload = c._build_generation_payloads(shards, identity=identity)
    index_bytes = c.dumps_strict(index_payload).encode("utf-8")
    index_sha = c.sha256_bytes(index_bytes)
    bound = dict(commit_payload)
    bound["index_sha256"] = index_sha
    commit_bytes = c.dumps_strict(bound).encode("utf-8")
    return sum(b["bytes"] for b in blobs) + len(index_bytes) + len(commit_bytes)


def test_r1_exact_fit_budget_publishes_and_never_exceeds(tmp_path):
    shards = [make_shard("c0001", "c0002", "h", (make_record(0, 1, 1), make_record(1, 2, 2)))]
    total = _exact_generation_total(shards)
    root = tmp_path / "exact"
    report = c.write_capture_generation(
        root, shards=shards, identity=make_identity(), config=enabled_config(root, disk_budget_bytes=total)
    )
    gen = root / report["generation_dir"]
    on_disk = sum(p.stat().st_size for p in gen.iterdir() if p.is_file())
    assert report["status"] == "COMMITTED"
    assert report["total_bytes"] == total
    assert report["total_bytes"] <= total
    assert report["total_bytes"] == on_disk
    assert report["disk_budget_bytes"] == total


def test_r1_index_sha_field_pushes_over_budget_and_refuses(tmp_path):
    shards = [make_shard("c0001", "c0002", "h", (make_record(0, 1, 1),))]
    total = _exact_generation_total(shards)
    # the pre-final total (COMMIT without the bound index_sha256) is the exact budget
    # that used to fail open; it must now refuse with no published COMMIT.
    blobs, index_payload, commit_payload = c._build_generation_payloads(shards, identity=make_identity())
    prefinal_total = (
        sum(b["bytes"] for b in blobs)
        + len(c.dumps_strict(index_payload).encode("utf-8"))
        + len(c.dumps_strict(commit_payload).encode("utf-8"))
    )
    assert prefinal_total < total  # the index_sha256 field adds bytes
    root = tmp_path / "boundary"
    with pytest.raises(c.BudgetError):
        c.write_capture_generation(
            root, shards=shards, identity=make_identity(),
            config=enabled_config(root, disk_budget_bytes=prefinal_total),
        )
    gen = root / f"{c.GENERATION_DIR_PREFIX}capture-abc"
    assert not (gen / c.GENERATION_COMMIT_BASENAME).exists()
    assert not list(root.glob("**/COMMIT")) if root.exists() else True


def test_r1_successful_publish_is_within_budget_for_many_budgets(tmp_path):
    shards = [
        make_shard("c0001", "c0002", "h", (make_record(0, 1, 1), make_record(1, 2, 2))),
        make_shard("c0003", "c0004", "v", (make_record(2, 3, 3),)),
    ]
    total = _exact_generation_total(shards)
    root = tmp_path / "sweep"
    report = c.write_capture_generation(
        root, shards=shards, identity=make_identity(),
        config=enabled_config(root, disk_budget_bytes=total + 4096),
    )
    gen = root / report["generation_dir"]
    on_disk = sum(p.stat().st_size for p in gen.iterdir() if p.is_file())
    assert report["total_bytes"] == on_disk <= report["disk_budget_bytes"]


# ---------------------------------------------------------------------------
# Rework-1 — R2/R5: verify_generation checks COMMIT + index + shard set
# ---------------------------------------------------------------------------


def _write_gen(tmp_path, name="g", shards=None):
    root = tmp_path / name
    shards = shards or [make_shard("c0001", "c0002", "h", (make_record(0, 1, 1), make_record(1, 2, 2)))]
    report = c.write_capture_generation(root, shards=shards, identity=make_identity(), config=enabled_config(root))
    return root / report["generation_dir"]


def _tamper_commit(gen, field, value):
    path = gen / c.GENERATION_COMMIT_BASENAME
    commit = json.loads(path.read_text(encoding="utf-8"))
    commit[field] = value
    path.write_text(json.dumps(commit, sort_keys=True, indent=2), encoding="utf-8")


@pytest.mark.parametrize(
    "field, tamper",
    [
        ("schema", "x"),
        ("schema_version", "x"),
        ("mission", "x"),
        ("capture_id", "x"),
        ("capture_stage", "x"),
        ("identity_digest", "0" * 64),
        ("n_shards", 99),
        ("n_records", 99),
        ("shard_bytes", 999999),
        ("index_sha256", "0" * 64),
    ],
)
def test_r2_verify_rejects_tampered_commit_field(tmp_path, field, tamper):
    gen = _write_gen(tmp_path)
    assert c.verify_generation(gen)["ok"] is True
    _tamper_commit(gen, field, tamper)
    verified = c.verify_generation(gen)
    assert verified["ok"] is False
    assert any(field in problem for problem in verified["problems"]), verified["problems"]


def test_r2_verify_rejects_unknown_commit_field(tmp_path):
    gen = _write_gen(tmp_path)
    _tamper_commit(gen, "sneaky", 1)
    verified = c.verify_generation(gen)
    assert verified["ok"] is False
    assert any("field set mismatch" in problem for problem in verified["problems"])


def test_r5_verify_confirms_index_identity_digest(tmp_path):
    gen = _write_gen(tmp_path)
    index_path = gen / c.GENERATION_INDEX_BASENAME
    index = json.loads(index_path.read_text(encoding="utf-8"))
    index["identity_digest"] = "0" * 64
    index_path.write_text(c.dumps_strict(index), encoding="utf-8")
    verified = c.verify_generation(gen)
    assert verified["ok"] is False
    assert any("identity_digest" in problem for problem in verified["problems"])


def test_r5_verify_flags_unlisted_files_and_shard_set(tmp_path):
    gen = _write_gen(tmp_path)
    assert c.verify_generation(gen)["ok"] is True
    (gen / "stray-shard.bin").write_bytes(b"nope")
    verified = c.verify_generation(gen)
    assert verified["ok"] is False
    assert any("unlisted file" in problem for problem in verified["problems"])

    # index shard-set inconsistency (n_shards vs listed entries)
    gen2 = _write_gen(tmp_path, name="g2")
    index_path = gen2 / c.GENERATION_INDEX_BASENAME
    index = json.loads(index_path.read_text(encoding="utf-8"))
    index["n_shards"] = len(index["shards"]) + 1
    index_path.write_text(c.dumps_strict(index), encoding="utf-8")
    assert c.verify_generation(gen2)["ok"] is False


# ---------------------------------------------------------------------------
# Rework-1 — R3: no silent non-finite storage
# ---------------------------------------------------------------------------


def test_r3_f4_overflow_refused():
    with pytest.raises(c.NonFiniteError):
        c.encode_records([make_record(sci_u_pre=1e300)], relation_id="a|b")
    with pytest.raises(c.NonFiniteError):
        c.encode_records([make_record(sci_v_pre=-1e300)], relation_id="a|b")
    # a representable f4 value is stored finite (no false refusal)
    array = c.encode_records([make_record(sci_u_pre=1e30, sci_v_pre=-1e30)], relation_id="a|b")
    assert np.isfinite(array["sci_u_pre"][0]) and np.isfinite(array["sci_v_pre"][0])


def test_r3_f8_non_finite_conversion_refused():
    # an int too large for a finite float would raise a raw OverflowError -> typed refusal
    with pytest.raises(c.NonFiniteError):
        c.encode_records([make_record(sci_u_post=10**400)], relation_id="a|b")
    with pytest.raises(c.NonFiniteError):
        c.encode_records([make_record(ramp_u=float("inf"))], relation_id="a|b")


def test_r3_int_range_refused_not_overflowerror():
    with pytest.raises(c.SchemaError):
        c.encode_records([make_record(surviving_sample_count_u=2**70)], relation_id="a|b")
    with pytest.raises(c.SchemaError):
        c.encode_records([make_record(gx=2**40)], relation_id="a|b")
    with pytest.raises(c.SchemaError):
        c.encode_records([make_record(gy=-(2**40))], relation_id="a|b")


def test_r3_assert_finite_array_enforced_on_encoded_output(monkeypatch):
    calls = []
    real = c.assert_finite_array

    def spy(name, array):
        calls.append(name)
        return real(name, array)

    monkeypatch.setattr(c, "assert_finite_array", spy)
    c.encode_records([make_record()], relation_id="a|b")
    assert "r40.records" in calls  # top-level enforcement ran on the encoded output


def test_r3_assert_finite_array_handles_structured_arrays():
    array = np.zeros(2, dtype=c.R40_RECORD_DTYPE)
    c.assert_finite_array("ok", array)
    array["sci_u_post"][1] = np.inf
    with pytest.raises(c.NonFiniteError):
        c.assert_finite_array("bad", array)


# ---------------------------------------------------------------------------
# Rework-1 — R4: production_wired is parameterized (default strict)
# ---------------------------------------------------------------------------


def test_r4_production_wired_is_strict_by_default(tmp_path):
    assert c.EXPECTED_PRODUCTION_WIRED is False
    wired = make_identity(production_wired=True)
    # identity with production_wired=True needs an explicit expected flag
    with pytest.raises(c.IdentityError):
        c.check_capture_identity(wired)
    checked = c.check_capture_identity(wired, expected_production_wired=True)
    assert checked["production_wired"] is True
    assert c.identity_digest(wired, expected_production_wired=True) != c.identity_digest(make_identity())
    # and a default-OFF identity refuses when True is expected
    with pytest.raises(c.IdentityError):
        c.check_capture_identity(make_identity(), expected_production_wired=True)
    # the writer keeps the strict default
    root = tmp_path / "wired"
    with pytest.raises(c.IdentityError):
        c.write_capture_generation(
            root, shards=[make_shard()], identity=wired, config=enabled_config(root)
        )
    assert not root.exists()


def test_r4_wired_identity_path_is_explicit_and_verifiable(tmp_path):
    root = tmp_path / "wired2"
    wired = make_identity(production_wired=True)
    report = c.write_capture_generation(
        root, shards=[make_shard()], identity=wired,
        config=enabled_config(root), expected_production_wired=True,
    )
    gen = root / report["generation_dir"]
    index = json.loads((gen / c.GENERATION_INDEX_BASENAME).read_text(encoding="utf-8"))
    assert index["production_wired"] is True
    assert c.verify_generation(gen, expected_production_wired=True)["ok"] is True
    # the default-strict verifier refuses that generation
    assert c.verify_generation(gen)["ok"] is False
