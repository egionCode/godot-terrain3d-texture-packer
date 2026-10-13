# Godot Terrain3D Texture Packer

Packs PBR textures into the layout [Terrain3D](https://github.com/TokisanGames/Terrain3D) (Godot) expects, no GIMP needed:

| Output file | RGB | Alpha |
|---|---|---|
| `<name>_albedo_height.dds` | albedo (x AO, optional) | height |
| `<name>_normal_roughness.png` | OpenGL normal | roughness |

## Requirements

- Python 3.9+ with `pillow` and `numpy`
- `PySide6` (GUI only)
- ImageMagick 7 (`magick`) to write DDS

## GUI

```bash
./t3dpack_gui.py
```

1. Drop the material folder on the window (or use **Choose folder...**). Maps are detected by filename and the output is set to `<folder>/output`.
2. Adjust what you need: AO and strength, DirectX normal, roughness channel, size, formats.
3. Click **Generate**.

## CLI

```bash
# one material
./t3dpack.py ~/textures/Ground054 -o ~/project/textures/terrain

# several materials (a folder with subfolders), all at 2048
./t3dpack.py ~/textures -o out -s 2048

# only show what was detected
./t3dpack.py ~/textures -n
```

See all options with `./t3dpack.py --help`.

> Terrain3D requires every terrain texture to have the same size. For batches, use `-s`.

## License

[GPL-3.0-or-later](LICENSE). Free and open source forever: you can use, modify and share it (including to make commercial games), but any distributed version or fork must stay open source under the same license.
