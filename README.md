# Godot Terrain3D Texture Packer

Empacota texturas PBR no formato que o [Terrain3D](https://github.com/TokisanGames/Terrain3D) (Godot) espera, sem precisar do GIMP:

| Arquivo gerado | RGB | Alpha |
|---|---|---|
| `<nome>_albedo_height.dds` | albedo (x AO, opcional) | height |
| `<nome>_normal_roughness.png` | normal OpenGL | roughness |

## Requisitos

- Python 3.9+ com `pillow` e `numpy`
- `PySide6` (só para a GUI)
- ImageMagick 7 (`magick`) para gerar DDS

## GUI

```bash
./t3dpack_gui.py
```

1. Arraste a pasta do material para a janela (ou use **Escolher pasta...**). Os mapas são detectados pelo nome e a saída vira `<pasta>/output`.
2. Ajuste o que precisar: AO e intensidade, normal DirectX, canal do roughness, tamanho, formatos.
3. Clique em **Gerar**.

## CLI

```bash
# um material
./t3dpack.py ~/texturas/Ground054 -o ~/projeto/textures/terrain

# vários materiais (uma pasta com subpastas), todos em 2048
./t3dpack.py ~/texturas -o out -s 2048

# só mostrar o que foi detectado
./t3dpack.py ~/texturas -n
```

Veja todas as opções com `./t3dpack.py --help`.

> O Terrain3D exige que todas as texturas do terreno tenham o mesmo tamanho. Em lote, use `-s`.
