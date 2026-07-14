"""Regression test for fetch_ffhq.extract_subset's held-out identity exclusion.

This is the single most load-bearing property of the module: if a future edit
silently reintroduces eval identities into the training corpus, the Phase 4
gate becomes meaningless. Uses a small stub zip built in tmp_path so the test
runs in milliseconds and never touches the real 3.6 GB FFHQ-Makeup.zip.
"""
import zipfile
from pathlib import Path

from makeup_transfer.fetch_ffhq import HELDOUT, extract_subset


def _make_stub_zip(zip_path: Path, ids: list[str]) -> None:
    with zipfile.ZipFile(zip_path, "w") as z:
        for ident in ids:
            z.writestr(f"{ident}/bare.jpg", b"fake-bare-bytes")
            z.writestr(f"{ident}/makeup_01.jpg", b"fake-makeup-bytes")


def test_extract_subset_excludes_heldout_identities(tmp_path):
    heldout_sample = sorted(HELDOUT)[0]
    non_heldout = ["000001", "000002", "000003"]
    ids = [heldout_sample, *non_heldout]

    zip_path = tmp_path / "stub.zip"
    _make_stub_zip(zip_path, ids)
    out_dir = tmp_path / "out"

    chosen = extract_subset(zip_path, out_dir, n_identities=len(ids))

    assert heldout_sample not in chosen
    assert not (HELDOUT & set(chosen))

    written_names = [p.name for p in out_dir.glob("*.jpg")]
    assert written_names, "expected files to be written"
    for name in written_names:
        for held in HELDOUT:
            assert not name.startswith(held), f"held-out identity leaked into corpus: {name}"


def test_extract_subset_flattens_member_names(tmp_path):
    zip_path = tmp_path / "stub.zip"
    _make_stub_zip(zip_path, ["000001"])
    out_dir = tmp_path / "out"

    extract_subset(zip_path, out_dir, n_identities=1)

    assert (out_dir / "000001_makeup_01.jpg").exists()
    assert (out_dir / "000001_bare.jpg").exists()
