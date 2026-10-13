#!/usr/bin/env python3
"""
t3dpack: empacota texturas PBR no formato exigido pelo Terrain3D (Godot).

Saida por material:
  <nome>_albedo_height.dds    RGB = albedo (multiplicado pelo AO, se houver), A = height
  <nome>_normal_roughness.png RGB = normal OpenGL (Y+),                       A = roughness

Dependencias:
  - Python 3.9+, Pillow, numpy
  - ImageMagick 7 (`magick`) no PATH: usado para escrever DDS BC3/DXT5 com mipmaps
    e como fallback para ler formatos que o Pillow nao le (ex.: EXR).

Decisoes nao obvias:
  - AO e multiplicado no espaco sRGB de 8 bits, igual ao modo Multiply "legacy" do GIMP,
    que e o fluxo manual que este script substitui.
  - Height e lido em float (aceita PNG 16 bits / EXR) e so quantizado para 8 bits no fim,
    para nao perder precisao antes de uma eventual normalizacao.
  - Normal DirectX (Y-) e convertido para OpenGL invertendo o canal G. Arquivos sem
    indicacao de GL/DX sao tratados como OpenGL (padrao do Poly Haven/ambientCG "GL");
    use --dx para forcar a inversao.
  - Glossiness/smoothness e invertido para virar roughness.
  - Mapas "ARM" (Poly Haven: R=AO, G=roughness, B=metal) sao usados como fonte de AO
    e roughness quando os mapas separados nao existem.
  - Normal sai em PNG por padrao (--normal-format): DXT5 comprime mal normal maps e
    gera blocos visiveis no relevo. Os formatos sao separados por tipo porque o
    Terrain3D monta um array para albedos e outro para normais.
  - Todas as texturas de um material sao redimensionadas para o tamanho do albedo
    (ou --size). O Terrain3D exige que TODAS as texturas do array tenham o mesmo
    tamanho e formato, por isso --size e recomendado em lote.
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

# Avisos passam pelo logging para que a GUI possa capturar e exibir no log dela.
log = logging.getLogger("t3dpack")

IMAGE_EXTS = {".png", ".jpg", ".jpeg", ".tga", ".tif", ".tiff", ".bmp", ".exr", ".webp", ".dds"}

# Palavras-chave por tipo de mapa. A ordem importa: o primeiro tipo que casar ganha,
# entao termos mais especificos (normal_dx, gloss) vem antes dos genericos.
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


# ---------------------------------------------------------------- deteccao

def tokens_of(stem: str) -> set[str]:
    """Quebra o nome em tokens e tambem gera pares unidos ("nor"+"gl" -> "nor_gl", "norgl").

    Necessario porque cada fonte nomeia de um jeito: "rock_nor_gl_2k" (Poly Haven),
    "Ground054_2K_NormalGL" (ambientCG), "T_Rock_Normal" (Megascans).
    """
    # Separa camelCase antes de baixar a caixa ("BaseColor" -> "Base Color").
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
    """Classifica os mapas de uma pasta. Nao exige albedo (a GUI aceita material so com normal)."""
    files: dict[str, Path] = {}
    for f in sorted(folder.iterdir()):
        if not f.is_file() or f.suffix.lower() not in IMAGE_EXTS:
            continue
        kind = classify(f)
        if kind is None:
            continue
        # Se houver varias resolucoes/formatos do mesmo mapa, prefere PNG/EXR (sem perdas)
        # a JPG, e mantem o primeiro encontrado nos demais casos.
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
    """Cada pasta com um albedo vira um material; pastas sem albedo sao varridas recursivamente."""
    found: list[Material] = []
    for root in inputs:
        if not root.is_dir():
            log.warning("ignorando (nao e pasta): %s", root)
            continue
        mat = scan_folder(root)
        if mat:
            found.append(mat)
            continue
        for sub in sorted(p for p in root.iterdir() if p.is_dir()):
            found.extend(find_materials([sub]))
    return found


# ---------------------------------------------------------------- leitura

def load_float(path: Path) -> np.ndarray:
    """Le a imagem como float32 em [0,1], shape (H, W, C).

    Usa o Pillow quando possivel; para EXR e afins converte via ImageMagick para
    PNG 16 bits temporario, ja que o Pillow nao suporta esses formatos.
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

    # Modos de 16 bits/inteiros precisam de escala propria; os de 8 bits viram RGB(A).
    if img.mode in ("I;16", "I;16B", "I;16L", "I"):
        arr = np.asarray(img, dtype=np.float32)
        arr = arr / (65535.0 if arr.max() > 255 else 255.0)
        return arr[..., None]
    if img.mode == "F":
        return np.asarray(img, dtype=np.float32)[..., None]
    if img.mode not in ("L", "LA", "RGB", "RGBA"):
        img = img.convert("RGBA" if "A" in img.getbands() else "RGB")
    arr = np.asarray(img, dtype=np.float32) / 255.0
    # Modo "L" volta 2D; o resto do codigo assume sempre (H, W, C).
    return arr[..., None] if arr.ndim == 2 else arr


def resize(arr: np.ndarray, size: tuple[int, int]) -> np.ndarray:
    """Redimensiona canal a canal em modo float ("F") para nao perder os 16 bits do height."""
    h, w = arr.shape[:2]
    if (w, h) == size:
        return arr
    chans = [np.asarray(Image.fromarray(arr[..., c]).resize(size, Image.Resampling.LANCZOS))
             for c in range(arr.shape[2])]
    return np.clip(np.stack(chans, axis=-1), 0.0, 1.0)


def gray(arr: np.ndarray) -> np.ndarray:
    """Mapa escalar: usa o canal R. Mapas de AO/height/roughness RGB tem R=G=B."""
    return arr[..., 0]


# ---------------------------------------------------------------- empacotamento

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
        # lerp(1, ao, strength): strength 1 = Multiply puro do GIMP com opacidade 100%.
        factor = 1.0 - ao_strength * (1.0 - ao)
        rgb = rgb * factor[..., None]

    if "height" in mat.files:
        height = gray(resize(load_float(mat.files["height"]), size))
        if normalize_height:
            lo, hi = float(height.min()), float(height.max())
            if hi > lo:
                height = (height - lo) / (hi - lo)
    else:
        # Sem height, alpha neutro (meio cinza) para o blend por altura nao favorecer
        # este material sobre os outros.
        log.warning("%s: sem height, usando 0.5", mat.name)
        height = np.full(rgb.shape[:2], 0.5, dtype=np.float32)

    return np.dstack([rgb, height])


def pack_normal_roughness(mat: Material, size: tuple[int, int] | None, force_dx: bool,
                          rough_channel: int = 0, invert_rough: bool = False) -> np.ndarray | None:
    """rough_channel escolhe o canal do arquivo de roughness (1 = G, para mapas ARM/ORM).
    invert_rough trata o arquivo "roughness" como glossiness (usado pela GUI).

    size None usa o tamanho da propria normal (caso da GUI gerando so normal_roughness).
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
        log.warning("%s: sem roughness, usando 0.5", mat.name)
        rough = np.full(normal.shape[:2], 0.5, dtype=np.float32)

    return np.dstack([normal, rough])


# ---------------------------------------------------------------- escrita

def to_u8(arr: np.ndarray) -> np.ndarray:
    return (np.clip(arr, 0.0, 1.0) * 255.0 + 0.5).astype(np.uint8)


def write_image(arr: np.ndarray, out: Path, fmt: str) -> None:
    img = Image.fromarray(to_u8(arr), "RGBA")
    if fmt == "png":
        img.save(out.with_suffix(".png"))
        return

    # DDS BC3/DXT5 com cadeia completa de mipmaps: e o que o Terrain3D espera.
    # Passamos por um PNG temporario porque o encoder DXT do ImageMagick e mais
    # maduro que o do Pillow e gera mipmaps sozinho.
    levels = int(math.log2(max(img.size))) + 1
    with tempfile.TemporaryDirectory() as td:
        tmp = Path(td) / "packed.png"
        img.save(tmp)
        subprocess.run([
            "magick", str(tmp),
            # Impede o IM de "otimizar" para cinza/sem alpha quando a imagem parecer
            # uniforme, o que trocaria o formato DXT5 por DXT1.
            "-type", "TrueColorAlpha",
            "-define", "dds:compression=dxt5",
            "-define", "dds:cluster-fit=true",
            # Sem isso o IM pondera a cor pelo alpha, e height baixo (alpha ~0)
            # estragaria o albedo dessas regioes.
            "-define", "dds:weight-by-alpha=false",
            "-define", f"dds:mipmaps={levels - 1}",
            str(out.with_suffix(".dds")),
        ], check=True, stdout=subprocess.DEVNULL, stderr=subprocess.PIPE)


def is_pow2(n: int) -> bool:
    return n > 0 and (n & (n - 1)) == 0


# ---------------------------------------------------------------- main

def main() -> int:
    ap = argparse.ArgumentParser(
        description="Empacota texturas PBR para o Terrain3D (albedo_height + normal_roughness).")
    ap.add_argument("inputs", nargs="+", type=Path,
                    help="pasta(s) de material, ou pasta contendo varias pastas de material")
    ap.add_argument("-o", "--out", type=Path, default=Path("terrain3d_out"), help="pasta de saida")
    ap.add_argument("-s", "--size", type=int, help="forca resolucao quadrada (ex.: 1024, 2048)")
    ap.add_argument("-f", "--format", choices=("dds", "png"), default="dds",
                    help="formato do albedo_height (padrao dds)")
    ap.add_argument("--normal-format", choices=("dds", "png"), default="png",
                    help="formato do normal_roughness (padrao png)")
    ap.add_argument("--ao-strength", type=float, default=1.0, help="0 a 1, padrao 1 (multiply total)")
    ap.add_argument("--no-ao", action="store_true", help="nao aplica AO no albedo")
    ap.add_argument("--normalize-height", action="store_true", help="estica o height para 0..1")
    ap.add_argument("--dx", action="store_true", help="trata normais sem sufixo GL/DX como DirectX")
    ap.add_argument("-n", "--dry-run", action="store_true", help="so mostra o que foi detectado")
    args = ap.parse_args()
    logging.basicConfig(format="  [aviso] %(message)s", level=logging.WARNING)

    if "dds" in (args.format, args.normal_format) and not shutil.which("magick") and not args.dry_run:
        print("erro: ImageMagick 7 ('magick') nao encontrado; use png ou instale o imagemagick",
              file=sys.stderr)
        return 1

    mats = find_materials(args.inputs)
    if not mats:
        print("nenhum material encontrado (precisa de pelo menos um mapa de albedo/color/diffuse)",
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
            log.warning("%dx%d nao e potencia de 2", *real)
        write_image(ah, args.out / f"{mat.name}_albedo_height", args.format)

        nr = pack_normal_roughness(mat, real, args.dx)
        if nr is None:
            log.warning("%s: sem normal map, normal_roughness nao gerado", mat.name)
        else:
            write_image(nr, args.out / f"{mat.name}_normal_roughness", args.normal_format)
        print(f"  -> {real[0]}x{real[1]} albedo={args.format} normal={args.normal_format}")

    if len(sizes_seen) > 1:
        print("\n[aviso] materiais com tamanhos diferentes: o Terrain3D exige o mesmo tamanho "
              "em todas as texturas. Rode de novo com --size.", file=sys.stderr)
    return 0


if __name__ == "__main__":
    sys.exit(main())
