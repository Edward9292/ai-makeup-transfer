"""UV unwrap / re-projection via the MediaPipe canonical face model.

Maps between image space and a canonical UV texture space using the 468 FaceLandmarker
landmarks and the canonical face model's per-vertex UV coordinates + triangulation
(``weights/canonical_face_model.obj``, from the MediaPipe repo).

This is a 2D piecewise-affine (per-triangle barycentric) mapping, not a true 3DMM fit; it is
the same construction MediaPipe's own AR-makeup effects use. Good enough for Phase 0 and as a
fallback while the 3DDFA_V2/DECA route is stood up. Forward (``unwrap``) samples the photo into
UV space; inverse (``project``) renders a UV-space RGBA layer back onto a target photo.

Occlusion handling: per-triangle *visibility* = image-space area / UV-space area. Triangles on
the averted side of a turned face compress to slivers in the image, so their ratio collapses;
``mirror_fill`` uses the canonical UV chart's left/right symmetry to fill those from the
better-seen half.

Quick check::

    python -m makeup_transfer.uv_unwrap data/mt/makeup_vFG586.png --out preprocess_out
"""

from __future__ import annotations

from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path

import cv2
import numpy as np

_OBJ_PATH = Path(__file__).resolve().parents[2] / "weights" / "canonical_face_model.obj"

N_LANDMARKS = 468


# --- canonical model ----------------------------------------------------------------------


@dataclass(frozen=True)
class CanonicalMesh:
    """Canonical face mesh: per-landmark UV coords + triangulation."""

    uvs: np.ndarray        # (468, 2) float32, in [0, 1], y-down (image convention)
    triangles: np.ndarray  # (T, 3) int32 landmark indices


def _parse_obj(path: Path) -> CanonicalMesh:
    """Parse the canonical_face_model.obj: v/vt/f records, f as ``v/vt`` per corner."""
    uvs_raw: list[list[float]] = []
    faces: list[list[tuple[int, int]]] = []
    n_vertices = 0
    for line in path.read_text().splitlines():
        parts = line.split()
        if not parts:
            continue
        if parts[0] == "v":
            n_vertices += 1
        elif parts[0] == "vt":
            uvs_raw.append([float(parts[1]), float(parts[2])])
        elif parts[0] == "f":
            corners = []
            for tok in parts[1:4]:
                ids = tok.split("/")
                corners.append((int(ids[0]) - 1, int(ids[1]) - 1))
            faces.append(corners)

    if n_vertices != N_LANDMARKS:
        raise ValueError(f"Expected {N_LANDMARKS} vertices in {path}, got {n_vertices}")

    # Build per-vertex UV from the f records (vertex index -> vt index).
    uvs = np.zeros((N_LANDMARKS, 2), dtype=np.float32)
    seen = np.zeros(N_LANDMARKS, dtype=bool)
    triangles = np.zeros((len(faces), 3), dtype=np.int32)
    vts = np.asarray(uvs_raw, dtype=np.float32)
    for t, corners in enumerate(faces):
        for k, (vi, vti) in enumerate(corners):
            triangles[t, k] = vi
            if not seen[vi]:
                uvs[vi] = vts[vti]
                seen[vi] = True
    if not seen.all():
        raise ValueError("Some landmarks have no UV coordinate in the OBJ.")

    uvs[:, 1] = 1.0 - uvs[:, 1]  # OBJ vt is y-up; flip to image convention
    return CanonicalMesh(uvs=uvs, triangles=triangles)


@lru_cache(maxsize=1)
def canonical_mesh(obj_path: str | None = None) -> CanonicalMesh:
    path = Path(obj_path) if obj_path else _OBJ_PATH
    if not path.exists():
        raise FileNotFoundError(
            f"Canonical face model not found at {path}. Download it with:\n"
            "  curl -L -o weights/canonical_face_model.obj "
            "https://raw.githubusercontent.com/google-ai-edge/mediapipe/master/"
            "mediapipe/modules/face_geometry/data/canonical_face_model.obj"
        )
    return _parse_obj(path)


@lru_cache(maxsize=1)
def mirror_permutation() -> np.ndarray:
    """Landmark index permutation that mirrors the face left<->right in UV space.

    Computed numerically: nearest canonical UV to (1-u, v). The canonical chart is bilaterally
    symmetric about u=0.5, so this is (nearly) an exact involution.
    """
    uvs = canonical_mesh().uvs
    flipped = uvs.copy()
    flipped[:, 0] = 1.0 - flipped[:, 0]
    d = np.linalg.norm(uvs[None, :, :] - flipped[:, None, :], axis=2)  # (468, 468)
    perm = d.argmin(axis=1).astype(np.int32)
    if d[np.arange(N_LANDMARKS), perm].max() > 0.01:
        raise RuntimeError("Canonical UV chart is not left/right symmetric; cannot mirror.")
    return perm


# --- barycentric rasterization ------------------------------------------------------------


def _rasterize(points: np.ndarray, triangles: np.ndarray, h: int, w: int):
    """Rasterize a triangulated point set over an (h, w) grid.

    Returns ``(tri_id, bary)``: per-pixel triangle index (-1 outside) and barycentric weights
    (h, w, 3) w.r.t. that triangle's corners.
    """
    tri_id = np.full((h, w), -1, dtype=np.int32)
    buf = np.zeros((h, w), dtype=np.uint8)
    for t, tri in enumerate(triangles):
        pts = np.round(points[tri]).astype(np.int32)
        buf[:] = 0
        cv2.fillConvexPoly(buf, pts, 1)
        tri_id[buf > 0] = t

    ys, xs = np.nonzero(tri_id >= 0)
    p = np.stack([xs, ys], axis=1).astype(np.float64)
    corners = points[triangles[tri_id[ys, xs]]]  # (N, 3, 2)
    a, b, c = corners[:, 0], corners[:, 1], corners[:, 2]
    v0, v1, v2 = b - a, c - a, p - a
    d00 = (v0 * v0).sum(1)
    d01 = (v0 * v1).sum(1)
    d11 = (v1 * v1).sum(1)
    d20 = (v2 * v0).sum(1)
    d21 = (v2 * v1).sum(1)
    denom = d00 * d11 - d01 * d01
    denom[np.abs(denom) < 1e-12] = 1e-12
    wb = (d11 * d20 - d01 * d21) / denom
    wc = (d00 * d21 - d01 * d20) / denom
    bary = np.zeros((h, w, 3), dtype=np.float32)
    bary[ys, xs, 0] = 1.0 - wb - wc
    bary[ys, xs, 1] = wb
    bary[ys, xs, 2] = wc
    return tri_id, bary


@lru_cache(maxsize=4)
def _uv_raster(uv_size: int):
    """Cached rasterization of the canonical UV layout at a given texture size."""
    mesh = canonical_mesh()
    return _rasterize(mesh.uvs * (uv_size - 1), mesh.triangles, uv_size, uv_size)


def _triangle_areas(points: np.ndarray, triangles: np.ndarray) -> np.ndarray:
    a, b, c = points[triangles[:, 0]], points[triangles[:, 1]], points[triangles[:, 2]]
    return 0.5 * np.abs(
        (b[:, 0] - a[:, 0]) * (c[:, 1] - a[:, 1]) - (c[:, 0] - a[:, 0]) * (b[:, 1] - a[:, 1])
    )


# --- public API -----------------------------------------------------------------------------


def unwrap(image: np.ndarray, landmarks: np.ndarray, uv_size: int = 512):
    """Sample a face photo into canonical UV space.

    Returns ``(uv_texture, visibility)``: (uv_size, uv_size, 3) uint8 in the image's channel
    order, and a float32 (uv_size, uv_size) map in [0, 1] — low where the source triangle was
    compressed/occluded in the photo (and 0 outside the face chart).
    """
    mesh = canonical_mesh()
    tri_id, bary = _uv_raster(uv_size)
    inside = tri_id >= 0
    tris = mesh.triangles[np.where(inside, tri_id, 0)]  # (H, W, 3)

    map_x = (landmarks[tris, 0] * bary).sum(axis=2).astype(np.float32)
    map_y = (landmarks[tris, 1] * bary).sum(axis=2).astype(np.float32)
    uv_texture = cv2.remap(image, map_x, map_y, cv2.INTER_LINEAR, borderMode=cv2.BORDER_REPLICATE)
    uv_texture[~inside] = 0

    area_img = _triangle_areas(landmarks, mesh.triangles)
    area_uv = _triangle_areas(mesh.uvs * (uv_size - 1), mesh.triangles)
    ratio = area_img / np.maximum(area_uv, 1e-9)
    ratio /= max(np.percentile(ratio, 75), 1e-9)
    vis = np.where(inside, np.clip(ratio[tri_id], 0.0, 1.0), 0.0).astype(np.float32)
    return uv_texture, vis


def _mirror_weights(visibility: np.ndarray) -> np.ndarray:
    """Per-pixel weight on a pixel's *own* value vs its mirror, from relative visibility."""
    flip_vis = visibility[:, ::-1]
    total = visibility + flip_vis
    return np.where(total > 1e-6, visibility / np.maximum(total, 1e-6), 0.5)[..., None]


def _blend_mirror(uv_texture: np.ndarray, w: np.ndarray) -> np.ndarray:
    blended = w * uv_texture.astype(np.float32) + (1.0 - w) * uv_texture[:, ::-1].astype(np.float32)
    return blended.astype(uv_texture.dtype)


def mirror_fill(uv_texture: np.ndarray, visibility: np.ndarray):
    """Blend each UV pixel with its left/right mirror, weighted by relative visibility.

    Fills the averted half of a turned face from the better-seen half. Returns
    ``(filled_texture, filled_visibility)``.
    """
    mirror_permutation()  # raises if the chart is not symmetric
    w = _mirror_weights(visibility)
    return _blend_mirror(uv_texture, w), np.maximum(visibility, visibility[:, ::-1])


def mirror_fill_pair(
    uv_a: np.ndarray, uv_b: np.ndarray, vis_a: np.ndarray, vis_b: np.ndarray,
):
    """Mirror-fill two textures of the same face with **one shared** weight map.

    Filling each shot independently is wrong for a *pair*: a pixel that gets filled from the
    mirror in one shot but kept in the other compares two different bits of face, and the
    resulting mismatch shows up in the difference as makeup. Driving both fills from the pair's
    joint visibility (``min``) means every UV pixel is either kept in both shots or taken from
    the same mirrored location in both, so the comparison stays like-for-like.

    Returns ``(filled_a, filled_b, filled_visibility)``.
    """
    mirror_permutation()  # raises if the chart is not symmetric
    vis = np.minimum(vis_a, vis_b)
    w = _mirror_weights(vis)
    return _blend_mirror(uv_a, w), _blend_mirror(uv_b, w), np.maximum(vis, vis[:, ::-1])


def project(uv_layer: np.ndarray, landmarks: np.ndarray, height: int, width: int):
    """Render a UV-space layer (e.g. RGBA makeup) back onto image space for one face.

    Returns ``(layer_img, face_mask)``: the layer resampled to (height, width) with the same
    channel count, and a uint8 mask of the face chart's footprint in the image.
    """
    mesh = canonical_mesh()
    uv_size = uv_layer.shape[0]
    tri_id, bary = _rasterize(landmarks, mesh.triangles, height, width)
    inside = tri_id >= 0
    tris = mesh.triangles[np.where(inside, tri_id, 0)]

    uv_pts = mesh.uvs * (uv_size - 1)
    map_x = (uv_pts[tris[..., 0], 0] * bary[..., 0]
             + uv_pts[tris[..., 1], 0] * bary[..., 1]
             + uv_pts[tris[..., 2], 0] * bary[..., 2]).astype(np.float32)
    map_y = (uv_pts[tris[..., 0], 1] * bary[..., 0]
             + uv_pts[tris[..., 1], 1] * bary[..., 1]
             + uv_pts[tris[..., 2], 1] * bary[..., 2]).astype(np.float32)
    layer_img = cv2.remap(uv_layer, map_x, map_y, cv2.INTER_LINEAR)
    layer_img[~inside] = 0
    return layer_img, (inside.astype(np.uint8) * 255)


def region_masks_uv(uv_size: int = 512) -> dict[str, np.ndarray]:
    """The preprocess REGION_GROUPS polygons, drawn in canonical UV space (fixed for all faces)."""
    from .preprocess import REGION_GROUPS

    uvs = canonical_mesh().uvs * (uv_size - 1)
    masks: dict[str, np.ndarray] = {}
    for name, idxs in REGION_GROUPS.items():
        mask = np.zeros((uv_size, uv_size), dtype=np.uint8)
        hull = cv2.convexHull(uvs[idxs].astype(np.int32))
        cv2.fillConvexPoly(mask, hull, 255)
        masks[name] = mask
    return masks


def _main() -> None:
    import argparse

    from .preprocess import detect_face

    ap = argparse.ArgumentParser(description="Unwrap a face photo into canonical UV space.")
    ap.add_argument("image", help="path to a face image")
    ap.add_argument("--out", default="preprocess_out", help="output directory")
    ap.add_argument("--size", type=int, default=512, help="UV texture size")
    args = ap.parse_args()

    face = detect_face(args.image)
    uv_tex, vis = unwrap(face.image, face.landmarks, args.size)
    filled, _ = mirror_fill(uv_tex, vis)

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    stem = Path(args.image).stem
    cv2.imwrite(str(out / f"uv_{stem}.png"), uv_tex)
    cv2.imwrite(str(out / f"uv_{stem}_filled.png"), filled)
    cv2.imwrite(str(out / f"uv_{stem}_vis.png"), (vis * 255).astype(np.uint8))
    print(f"Wrote UV texture ({args.size}px), mirror-filled texture and visibility map to {out}/")


if __name__ == "__main__":
    _main()
