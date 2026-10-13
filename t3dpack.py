#!/usr/bin/env python3
"""
t3dpack: packs PBR textures into the layout required by Terrain3D (Godot).

Output per material:
  <name>_albedo_height.dds    RGB = albedo (multiplied by AO, if present), A = height
  <name>_normal_roughness.png RGB = OpenGL normal (Y+),                    A = roughness

Dependencies:
  - Python 3.9+, Pillow, numpy
  - ImageMagick 7 (`magick`) on PATH: used to write BC3/DXT5 DDS with mipmaps
    and as a fallback reader for formats Pillow can't open (e.g. EXR).

Non-obvious decisions:
  - AO is multiplied in 8-bit sRGB space, same as GIMP's "legacy" Multiply mode,
    which is the manual workflow this script replaces.
  - Height is read as float (accepts 16-bit PNG / EXR) and only quantized to 8 bits
    at the end, so no precision is lost before an optional normalization.
  - DirectX normals (Y-) are converted to OpenGL by flipping the G channel. Files
    without a GL/DX hint are treated as OpenGL (Poly Haven/ambientCG "GL" default);
    use --dx to force the flip.
  - Glossiness/smoothness is inverted to become roughness.
  - "ARM" maps (Poly Haven: R=AO, G=roughness, B=metal) are used as the AO and
    roughness source when the separate maps don't exist.
  - Normals are written as PNG by default (--normal-format): DXT5 compresses normal
    maps poorly and causes visible blocks in the relief. Formats are split per type
    because Terrain3D builds one array for albedos and another for normals.
  - Every texture of a material is resized to the albedo size (or --size). Terrain3D
    requires ALL textures in the array to share size and format, so --size is
    recommended for batches.
"""

from __future__ import annotations

import argparse
import logging
import math
import re
import shutil
import subprocess
import sys
import tempfile
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
from PIL import Image

# Warnings go through logging so the GUI can capture them and show them in its log.
log = logging.getLogger("t3dpack")

IMAGE_EXTS = {".png", ".jpg", ".jpeg", ".tga", ".tif", ".tiff", ".bmp", ".exr", ".webp", ".dds"}

# Keywords per map type. Order matters: the first matching type wins, so more
# specific terms (normal_dx, gloss) come before generic ones.
KEYWORDS: list[tuple[str, set[str]]] = [
    ("normal_dx", {"normaldx", "nordx", "nrmdx", "normal_dx", "nor_dx", "nrm_dx", "normal_directx"}),
    ("normal_gl", {"normalgl", "norgl", "nrmgl", "normal_gl", "nor_gl", "nrm_gl", "normal_opengl"}),
    ("normal", {"normal", "nor", "nrm", "norm", "normals"}),
    ("arm", {"arm"}),
    ("ao", {"ao", "ambientocclusion", "ambient_occlusion", "occlusion", "occ"}),
    ("gloss", {"gloss", "glossiness", "smoothness", "smooth"}),
    ("roughness", {"roughness", "rough", "rgh"}),
    ("height", {"height", "displacement", "disp", "displace", "heightmap", "bump"}),
    ("albedo", {"albedo", "basecolor", "base_color", "diffuse", "diff", "color", "colour", "col", "base"}),
]


@dataclass
class Material:
    name: str
    files: dict[str, Path] = field(default_factory=dict)


# ---------------------------------------------------------------- detection

def tokens_of(stem: str) -> set[str]:
    """Splits the name into tokens and also builds joined pairs ("nor"+"gl" -> "nor_gl", "norgl").

    Needed because every source names things differently: "rock_nor_gl_2k" (Poly Haven),
    "Ground054_2K_NormalGL" (ambientCG), "T_Rock_Normal" (Megascans).
    """
    # Split camelCase before lowercasing ("BaseColor" -> "Base Color").
    spaced = re.sub(r"([a-z])([A-Z])", r"\1 \2", stem)
    parts = [p for p in re.split(r"[^A-Za-z0-9]+", spaced.lower()) if p]
    toks = set(parts)
    for a, b in zip(parts, parts[1:]):
        toks.add(a + b)
        toks.add(f"{a}_{b}")
    return toks


def classify(path: Path) -> str | None:
    toks = tokens_of(path.stem)
    for kind, words in KEYWORDS:
        if toks & words:
            return kind
    return None


def detect(folder: Path) -> dict[str, Path]:
    """Classifies the maps in a folder. Doesn't require albedo (the GUI accepts normal-only materials)."""
    files: dict[str, Path] = {}
    for f in sorted(folder.iterdir()):
        if not f.is_file() or f.suffix.lower() not in IMAGE_EXTS:
            continue
        kind = classify(f)
        if kind is None:
            continue
        # With several resolutions/formats of the same map, prefer PNG/EXR (lossless)
        # over JPG, otherwise keep the first one found.
        prev = files.get(kind)
        if prev is None or (prev.suffix.lower() in {".jpg", ".jpeg"} and f.suffix.lower() not in {".jpg", ".jpeg"}):
            files[kind] = f
    return files


def scan_folder(folder: Path) -> Material | None:
    files = detect(folder)
    if "albedo" not in files:
        return None
    return Material(name=folder.name, files=files)


def find_materials(inputs: list[Path]) -> list[Material]:
    """Each folder with an albedo becomes a material; folders without one are scanned recursively."""
    found: list[Material] = []
    for root in inputs:
        if not root.is_dir():
            log.warning("skipping (not a folder): %s", root)
            continue
        mat = scan_folder(root)
        if mat:
            found.append(mat)
            continue
        for sub in sorted(p for p in root.iterdir() if p.is_dir()):
            found.extend(find_materials([sub]))
    return found


# ---------------------------------------------------------------- reading

def load_float(path: Path) -> np.ndarray:
    """Reads the image as float32 in [0,1], shape (H, W, C).

    Uses Pillow when possible; for EXR and similar it converts through ImageMagick
    to a temporary 16-bit PNG, since Pillow doesn't support those formats.
    """
    try:
        img = Image.open(path)
        img.load()
    except Exception:
        with tempfile.TemporaryDirectory() as td:
            tmp = Path(td) / "conv.png"
            subprocess.run(["magick", str(path), "-depth", "16", str(tmp)], check=True,
                           stdout=subprocess.DEVNULL, stderr=subprocess.PIPE)
            img = Image.open(tmp)
            img.load()

    # 16-bit/integer modes need their own scale; 8-bit modes become RGB(A).
    if img.mode in ("I;16", "I;16B", "I;16L", "I"):
        arr = np.asarray(img, dtype=np.float32)
        arr = arr / (65535.0 if arr.max() > 255 else 255.0)
        return arr[..., None]
    if img.mode == "F":
        return np.asarray(img, dtype=np.float32)[..., None]
    if img.mode not in ("L", "LA", "RGB", "RGBA"):
        img = img.convert("RGBA" if "A" in img.getbands() else "RGB")
    arr = np.asarray(img, dtype=np.float32) / 255.0
    # Mode "L" comes back 2D; the rest of the code always assumes (H, W, C).
    return arr[..., None] if arr.ndim == 2 else arr


def resize(arr: np.ndarray, size: tuple[int, int]) -> np.ndarray:
    """Resizes channel by channel in float mode ("F") to keep the height's 16 bits."""
    h, w = arr.shape[:2]
    if (w, h) == size:
        return arr
    chans = [np.asarray(Image.fromarray(arr[..., c]).resize(size, Image.Resampling.LANCZOS))
             for c in range(arr.shape[2])]
    return np.clip(np.stack(chans, axis=-1), 0.0, 1.0)


def gray(arr: np.ndarray) -> np.ndarray:
    """Scalar map: uses the R channel. RGB AO/height/roughness maps have R=G=B."""
    return arr[..., 0]


# ---------------------------------------------------------------- packing

def pack_albedo_height(mat: Material, size: tuple[int, int] | None, ao_strength: float,
                       normalize_height: bool) -> np.ndarray:
    albedo = load_float(mat.files["albedo"])
    if size is None:
        size = (albedo.shape[1], albedo.shape[0])
    albedo = resize(albedo, size)
    rgb = albedo[..., :3] if albedo.shape[2] >= 3 else np.repeat(albedo[..., :1], 3, axis=2)

    ao = None
    if "ao" in mat.files:
        ao = gray(resize(load_float(mat.files["ao"]), size))
    elif "arm" in mat.files:
        ao = resize(load_float(mat.files["arm"]), size)[..., 0]
    if ao is not None and ao_strength > 0:
        # lerp(1, ao, strength): strength 1 = plain GIMP Multiply at 100% opacity.
        factor = 1.0 - ao_strength * (1.0 - ao)
        rgb = rgb * factor[..., None]

    if "height" in mat.files:
        height = gray(resize(load_float(mat.files["height"]), size))
        if normalize_height:
            lo, hi = float(height.min()), float(height.max())
            if hi > lo:
                height = (height - lo) / (hi - lo)
    else:
        # No height: neutral alpha (mid gray) so height blending doesn't favor
        # this material over the others.
        log.warning("%s: no height, using 0.5", mat.name)
        height = np.full(rgb.shape[:2], 0.5, dtype=np.float32)

    return np.dstack([rgb, height])


def pack_normal_roughness(mat: Material, size: tuple[int, int] | None, force_dx: bool,
                          rough_channel: int = 0, invert_rough: bool = False) -> np.ndarray | None:
    """rough_channel picks the channel of the roughness file (1 = G, for ARM/ORM maps).
    invert_rough treats the "roughness" file as glossiness (used by the GUI).

    size None uses the normal's own size (GUI generating only normal_roughness).
    """
    kind = next((k for k in ("normal_gl", "normal_dx", "normal") if k in mat.files), None)
    if kind is None:
        return None
    normal = load_float(mat.files[kind])
    if size is None:
        size = (normal.shape[1], normal.shape[0])
    normal = resize(normal, size)[..., :3].copy()
    if kind == "normal_dx" or (kind == "normal" and force_dx):
        normal[..., 1] = 1.0 - normal[..., 1]  # DirectX (Y-) -> OpenGL (Y+)

    if "roughness" in mat.files:
        src = resize(load_float(mat.files["roughness"]), size)
        rough = src[..., min(rough_channel, src.shape[2] - 1)]
        if invert_rough:
            rough = 1.0 - rough
    elif "gloss" in mat.files:
        rough = 1.0 - gray(resize(load_float(mat.files["gloss"]), size))
    elif "arm" in mat.files:
        rough = resize(load_float(mat.files["arm"]), size)[..., 1]
    else:
        log.warning("%s: no roughness, using 0.5", mat.name)
        rough = np.full(normal.shape[:2], 0.5, dtype=np.float32)

    return np.dstack([normal, rough])


# ---------------------------------------------------------------- writing

def to_u8(arr: np.ndarray) -> np.ndarray:
    return (np.clip(arr, 0.0, 1.0) * 255.0 + 0.5).astype(np.uint8)


def write_image(arr: np.ndarray, out: Path, fmt: str) -> None:
    img = Image.fromarray(to_u8(arr), "RGBA")
    if fmt == "png":
        img.save(out.with_suffix(".png"))
        return

    # BC3/DXT5 DDS with a full mipmap chain: what Terrain3D expects.
    # Goes through a temporary PNG because ImageMagick's DXT encoder is more
    # mature than Pillow's and builds the mipmaps by itself.
    levels = int(math.log2(max(img.size))) + 1
    with tempfile.TemporaryDirectory() as td:
        tmp = Path(td) / "packed.png"
        img.save(tmp)
        subprocess.run([
            "magick", str(tmp),
            # Stops IM from "optimizing" to grayscale/no alpha when the image looks
            # uniform, which would swap DXT5 for DXT1.
            "-type", "TrueColorAlpha",
            "-define", "dds:compression=dxt5",
            "-define", "dds:cluster-fit=true",
            # Without this IM weights color by alpha, and low height (alpha ~0)
            # would ruin the albedo in those areas.
            "-define", "dds:weight-by-alpha=false",
            "-define", f"dds:mipmaps={levels - 1}",
            str(out.with_suffix(".dds")),
        ], check=True, stdout=subprocess.DEVNULL, stderr=subprocess.PIPE)


def is_pow2(n: int) -> bool:
    return n > 0 and (n & (n - 1)) == 0


# ---------------------------------------------------------------- main

def main() -> int:
    ap = argparse.ArgumentParser(
        description="Packs PBR textures for Terrain3D (albedo_height + normal_roughness).")
    ap.add_argument("inputs", nargs="+", type=Path,
                    help="material folder(s), or a folder containing several material folders")
    ap.add_argument("-o", "--out", type=Path, default=Path("terrain3d_out"), help="output folder")
    ap.add_argument("-s", "--size", type=int, help="force a square resolution (e.g. 1024, 2048)")
    ap.add_argument("-f", "--format", choices=("dds", "png"), default="dds",
                    help="albedo_height format (default dds)")
    ap.add_argument("--normal-format", choices=("dds", "png"), default="png",
                    help="normal_roughness format (default png)")
    ap.add_argument("--ao-strength", type=float, default=1.0, help="0 to 1, default 1 (full multiply)")
    ap.add_argument("--no-ao", action="store_true", help="don't apply AO to the albedo")
    ap.add_argument("--normalize-height", action="store_true", help="stretch height to 0..1")
    ap.add_argument("--dx", action="store_true", help="treat normals without a GL/DX suffix as DirectX")
    ap.add_argument("-n", "--dry-run", action="store_true", help="only show what was detected")
    args = ap.parse_args()
    logging.basicConfig(format="  [warning] %(message)s", level=logging.WARNING)

    if "dds" in (args.format, args.normal_format) and not shutil.which("magick") and not args.dry_run:
        print("error: ImageMagick 7 ('magick') not found; use png or install imagemagick",
              file=sys.stderr)
        return 1

    mats = find_materials(args.inputs)
    if not mats:
        print("no material found (needs at least one albedo/color/diffuse map)",
              file=sys.stderr)
        return 1

    args.out.mkdir(parents=True, exist_ok=True)
    size = (args.size, args.size) if args.size else None
    sizes_seen: set[tuple[int, int]] = set()

    for mat in mats:
        print(f"[{mat.name}]")
        for kind, path in mat.files.items():
            print(f"  {kind:10s} {path.name}")
        if args.dry_run:
            continue

        ah = pack_albedo_height(mat, size, 0.0 if args.no_ao else args.ao_strength,
                                args.normalize_height)
        real = (ah.shape[1], ah.shape[0])
        sizes_seen.add(real)
        if not all(is_pow2(v) for v in real):
            log.warning("%dx%d is not a power of 2", *real)
        write_image(ah, args.out / f"{mat.name}_albedo_height", args.format)

        nr = pack_normal_roughness(mat, real, args.dx)
        if nr is None:
            log.warning("%s: no normal map, normal_roughness not generated", mat.name)
        else:
            write_image(nr, args.out / f"{mat.name}_normal_roughness", args.normal_format)
        print(f"  -> {real[0]}x{real[1]} albedo={args.format} normal={args.normal_format}")

    if len(sizes_seen) > 1:
        print("\n[warning] materials have different sizes: Terrain3D requires the same size "
              "for every texture. Run again with --size.", file=sys.stderr)
    return 0


if __name__ == "__main__":
    sys.exit(main())
