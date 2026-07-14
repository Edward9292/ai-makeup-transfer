"""Extract a training subset of FFHQ-Makeup identities from the full zip.

The HF `images/` folder holds only 20 identities; the full 18k live only inside
FFHQ-Makeup.zip. We pull a subset out of the zip and leave the rest alone.

The zip stores one directory per identity (`000000/bare.jpg`, `000000/makeup_01.jpg`), but the
rest of this project uses flat `<id>_bare.jpg` / `<id>_makeup_NN.jpg` names — as does the existing
`data/ffhq_makeup/` sample dir, which `priors.find_pairs` globs. So we flatten on the way out.

Run::

    PYTHONPATH=src python -m makeup_transfer.fetch_ffhq --n 400
"""
from __future__ import annotations

import re
import zipfile
from pathlib import Path

ZIP = Path("data/FFHQ-Makeup.zip")
OUT = Path("data/ffhq_makeup_train")

# Eval identities. Training on these would invalidate the gate.
HELDOUT = {"002386", "003125", "003417"}

# e.g. "000000/makeup_01.jpg" -> ("000000", "makeup_01")
_MEMBER_RE = re.compile(r"(?:^|/)(\d{6})/(bare|makeup_\d+)\.jpg$")


def extract_subset(zip_path: Path = ZIP, out_dir: Path = OUT, n_identities: int = 400) -> list[str]:
    """Write `n_identities` identities' photos into `out_dir`, flattened. Returns the IDs written."""
    out_dir.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(zip_path) as z:
        members: dict[str, list[tuple[str, str]]] = {}   # ident -> [(member_path, kind)]
        for name in z.namelist():
            m = _MEMBER_RE.search(name)
            if not m:
                continue
            ident, kind = m.group(1), m.group(2)
            if ident in HELDOUT:
                continue
            members.setdefault(ident, []).append((name, kind))

        chosen = sorted(members)[:n_identities]
        for ident in chosen:
            for member, kind in members[ident]:
                # 000000/makeup_01.jpg -> 000000_makeup_01.jpg
                (out_dir / f"{ident}_{kind}.jpg").write_bytes(z.read(member))
    return chosen


def _main() -> None:
    import argparse

    ap = argparse.ArgumentParser(description="Extract an FFHQ-Makeup training subset from the zip.")
    ap.add_argument("--zip", default=str(ZIP))
    ap.add_argument("--out", default=str(OUT))
    ap.add_argument("--n", type=int, default=400, help="number of identities")
    args = ap.parse_args()

    ids = extract_subset(Path(args.zip), Path(args.out), args.n)
    n_files = len(list(Path(args.out).glob("*.jpg")))
    print(f"Extracted {len(ids)} identities ({n_files} photos) to {args.out}")
    print(f"Held out (never trained on): {sorted(HELDOUT)}")


if __name__ == "__main__":
    _main()
