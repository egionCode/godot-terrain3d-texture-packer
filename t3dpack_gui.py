#!/usr/bin/env python3
# SPDX-License-Identifier: GPL-3.0-or-later
# Copyright (C) 2026 Victor
"""
t3dpack_gui: graphical interface for t3dpack (Terrain3D texture packer).

Dependencies:
  - PySide6 (Qt 6)
  - t3dpack.py in the same folder (all packing logic lives there; this file is UI only)

Non-obvious decisions:
  - PySide6 instead of Tkinter: Tk isn't always installed on Arch/CachyOS, and Qt has
    native drag and drop and works well on Wayland.
  - Packing runs in a QThread because 4K files and ImageMagick's DXT5 compression
    take a few seconds and would freeze the window.
  - In the GUI the normal is always passed as "normal" and the DirectX checkbox decides
    the flip; roughness is always passed as "roughness" with the chosen channel and
    inversion. That way the user can fix anything the auto-detection gets wrong.
  - The output folder follows the material (<material folder>/output), so it isn't
    saved between sessions. Since detect() only looks at files at the folder's top
    level, results inside output/ are never mistaken for input maps.
  - ARM/ORM maps: AO uses the R channel (the same one gray() reads), and roughness
    uses the G channel selected in the combo box.
"""

from __future__ import annotations

import logging
import shutil
import sys
from pathlib import Path

import numpy as np
from PIL import Image
from PySide6.QtCore import QSettings, QSize, Qt, QThread, QObject, Signal
from PySide6.QtGui import QImage, QImageReader, QPixmap
from PySide6.QtWidgets import (
    QApplication, QCheckBox, QComboBox, QFileDialog, QFormLayout, QGridLayout, QGroupBox,
    QHBoxLayout, QLabel, QLineEdit, QMainWindow, QMessageBox, QPlainTextEdit, QProgressBar,
    QPushButton, QSlider, QVBoxLayout, QWidget,
)

sys.path.insert(0, str(Path(__file__).resolve().parent))
import t3dpack  # noqa: E402

FILE_FILTER = "Images (*.png *.jpg *.jpeg *.tga *.tif *.tiff *.bmp *.exr *.webp *.dds);;All (*)"
SIZES = ["Original", "256", "512", "1024", "2048", "4096"]
CHANNELS = ["R", "G", "B", "A"]
THUMB = 48
PREVIEW = 160
# Subfolder created inside the material folder for the generated files.
OUT_SUBDIR = "output"


# ---------------------------------------------------------------- widgets

class MapSlot(QWidget):
    """Row with thumbnail, path, browse and clear buttons. Accepts a dropped file."""

    changed = Signal()

    def __init__(self, title: str, required: bool = False):
        super().__init__()
        self.setAcceptDrops(True)
        self.thumb = QLabel()
        self.thumb.setFixedSize(THUMB, THUMB)
        self.thumb.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.thumb.setStyleSheet("border: 1px solid palette(mid);")
        self.edit = QLineEdit()
        self.edit.setPlaceholderText(f"{title} ({'required' if required else 'optional'}): drop or browse")
        self.edit.editingFinished.connect(self._refresh)
        browse = QPushButton("...")
        browse.setFixedWidth(32)
        browse.clicked.connect(self._browse)
        clear = QPushButton("x")
        clear.setFixedWidth(28)
        clear.setToolTip("Clear")
        clear.clicked.connect(lambda: self.set_path(None))

        lay = QHBoxLayout(self)
        lay.setContentsMargins(0, 0, 0, 0)
        lay.addWidget(self.thumb)
        lay.addWidget(self.edit, 1)
        lay.addWidget(browse)
        lay.addWidget(clear)

    def path(self) -> Path | None:
        text = self.edit.text().strip()
        return Path(text) if text else None

    def set_path(self, path: Path | None) -> None:
        self.edit.setText(str(path) if path else "")
        self._refresh()

    def _browse(self) -> None:
        start = str(self.path().parent) if self.path() else ""
        f, _ = QFileDialog.getOpenFileName(self, "Choose texture", start, FILE_FILTER)
        if f:
            self.set_path(Path(f))

    def _refresh(self) -> None:
        p = self.path()
        self.thumb.clear()
        if p and p.is_file():
            # QImageReader with scaledSize decodes straight to a small size, without
            # loading the whole 4K image just for the thumbnail. EXR isn't supported: shows "?".
            reader = QImageReader(str(p))
            reader.setScaledSize(QSize(THUMB, THUMB))
            img = reader.read()
            if img.isNull():
                self.thumb.setText("?")
            else:
                self.thumb.setPixmap(QPixmap.fromImage(img))
        elif p:
            self.thumb.setText("!")
        self.changed.emit()

    def dragEnterEvent(self, e):
        if e.mimeData().hasUrls():
            e.acceptProposedAction()

    def dropEvent(self, e):
        urls = e.mimeData().urls()
        if urls:
            self.set_path(Path(urls[0].toLocalFile()))
            e.acceptProposedAction()


class PreviewBox(QWidget):
    def __init__(self, title: str):
        super().__init__()
        self.img = QLabel("-")
        self.img.setFixedSize(PREVIEW, PREVIEW)
        self.img.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.img.setStyleSheet("border: 1px solid palette(mid);")
        lay = QVBoxLayout(self)
        lay.setContentsMargins(0, 0, 0, 0)
        lay.addWidget(self.img)
        lay.addWidget(QLabel(title), alignment=Qt.AlignmentFlag.AlignHCenter)

    def set_image(self, img: QImage | None) -> None:
        if img is None:
            self.img.clear()
            self.img.setText("-")
        else:
            self.img.setPixmap(QPixmap.fromImage(img))


# ---------------------------------------------------------------- worker

class LogBridge(QObject):
    """Forwards t3dpack warnings (emitted on the worker thread) to the window log.

    Composition (QObject + inner Handler) instead of multiple inheritance: mixing
    QObject with Python classes that have their own __init__ is fragile in PySide6.
    The Qt signal crosses threads safely (queued connection).
    """

    message = Signal(str)

    def __init__(self):
        super().__init__()
        bridge = self

        class _Handler(logging.Handler):
            def emit(self, record):
                bridge.message.emit(f"[warning] {record.getMessage()}")

        self.handler = _Handler(logging.INFO)


def to_qimages(arr: np.ndarray) -> tuple[QImage, QImage]:
    """Builds the RGB and alpha thumbnails. QImage can be created off the GUI thread
    (QPixmap can't), and .copy() detaches it from the numpy buffer that will be freed."""
    u8 = t3dpack.to_u8(arr)
    # RGB and alpha are downscaled separately: Pillow's thumbnail() on RGBA premultiplies
    # by alpha, and where height/roughness ~0 the preview color turned into noise.
    rgb_img = Image.fromarray(np.ascontiguousarray(u8[..., :3]), "RGB")
    a_img = Image.fromarray(np.ascontiguousarray(u8[..., 3]), "L")
    rgb_img.thumbnail((PREVIEW, PREVIEW))
    a_img.thumbnail((PREVIEW, PREVIEW))
    rgb = np.ascontiguousarray(np.asarray(rgb_img))
    a = np.ascontiguousarray(np.asarray(a_img))
    h, w = a.shape
    qrgb = QImage(rgb.data, w, h, 3 * w, QImage.Format.Format_RGB888).copy()
    qa = QImage(a.data, w, h, w, QImage.Format.Format_Grayscale8).copy()
    return qrgb, qa


class PackWorker(QThread):
    log = Signal(str)
    preview = Signal(str, QImage, QImage)  # "albedo" | "normal", rgb, alpha
    done = Signal(bool)

    def __init__(self, job: dict):
        super().__init__()
        self.job = job

    def run(self) -> None:
        j = self.job
        try:
            out: Path = j["out"]
            out.mkdir(parents=True, exist_ok=True)
            size = j["size"]
            real = size

            if j["do_albedo"]:
                files = {"albedo": j["albedo"]}
                if j["ao"]:
                    files["ao"] = j["ao"]
                if j["height"]:
                    files["height"] = j["height"]
                mat = t3dpack.Material(j["name"], files)
                ao_strength = j["ao_strength"] if j["use_ao"] else 0.0
                self.log.emit("packing albedo_height...")
                ah = t3dpack.pack_albedo_height(mat, size, ao_strength, j["normalize_height"])
                # normal_roughness follows the albedo size, like the CLI, so both
                # textures of the material match.
                real = (ah.shape[1], ah.shape[0])
                dst = out / f"{j['name']}_albedo_height"
                t3dpack.write_image(ah, dst, j["format"])
                self.preview.emit("albedo", *to_qimages(ah))
                self.log.emit(f"ok: {dst}.{j['format']} ({real[0]}x{real[1]})")

            if j["do_normal"]:
                files = {"normal": j["normal"]}
                if j["rough"]:
                    files["roughness"] = j["rough"]
                mat = t3dpack.Material(j["name"], files)
                self.log.emit("packing normal_roughness...")
                nr = t3dpack.pack_normal_roughness(mat, real, j["dx"], j["rough_channel"],
                                                   j["invert_rough"])
                dst = out / f"{j['name']}_normal_roughness"
                t3dpack.write_image(nr, dst, j["normal_format"])
                self.preview.emit("normal", *to_qimages(nr))
                self.log.emit(f"ok: {dst}.{j['normal_format']} ({nr.shape[1]}x{nr.shape[0]})")

            self.done.emit(True)
        except Exception as e:  # show any failure in the log instead of silently killing the thread
            stderr = getattr(e, "stderr", None)
            detail = f": {stderr.decode(errors='replace').strip()}" if stderr else ""
            self.log.emit(f"[error] {type(e).__name__}: {e}{detail}")
            self.done.emit(False)


# ---------------------------------------------------------------- window

class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Terrain3D Texture Packer")
        self.setAcceptDrops(True)
        self.settings = QSettings("egion", "t3dpack")
        self.worker: PackWorker | None = None

        root = QWidget()
        self.setCentralWidget(root)
        main = QVBoxLayout(root)

        # --- material
        src = QGroupBox("Material")
        f = QFormLayout(src)
        row = QHBoxLayout()
        self.folder = QLineEdit()
        self.folder.setPlaceholderText("drop a folder on the window to auto-detect")
        pick = QPushButton("Choose folder...")
        pick.clicked.connect(self._pick_folder)
        detect = QPushButton("Auto-detect")
        detect.clicked.connect(lambda: self._autodetect(Path(self.folder.text().strip())))
        row.addWidget(self.folder, 1)
        row.addWidget(pick)
        row.addWidget(detect)
        f.addRow("Folder:", row)
        self.name = QLineEdit()
        self.name.setPlaceholderText("output file prefix")
        f.addRow("Name:", self.name)
        main.addWidget(src)

        # --- albedo + height
        self.g_albedo = QGroupBox("Albedo + Height  ->  <name>_albedo_height")
        self.g_albedo.setCheckable(True)
        f = QFormLayout(self.g_albedo)
        self.s_albedo = MapSlot("Albedo", required=True)
        # Albedo picked by hand without a material folder: suggest the output from it.
        self.s_albedo.changed.connect(
            lambda: self.s_albedo.path() and self._suggest_out(self.s_albedo.path().parent, force=False))
        self.s_ao = MapSlot("AO")
        self.s_height = MapSlot("Height")
        f.addRow("Albedo:", self.s_albedo)
        f.addRow("AO:", self.s_ao)
        ao_row = QHBoxLayout()
        self.use_ao = QCheckBox("Multiply AO into albedo")
        self.use_ao.setChecked(True)
        self.ao_slider = QSlider(Qt.Orientation.Horizontal)
        self.ao_slider.setRange(0, 100)
        self.ao_slider.setValue(100)
        self.ao_label = QLabel("100%")
        self.ao_label.setFixedWidth(40)
        self.ao_slider.valueChanged.connect(lambda v: self.ao_label.setText(f"{v}%"))
        ao_row.addWidget(self.use_ao)
        ao_row.addWidget(QLabel("Strength:"))
        ao_row.addWidget(self.ao_slider, 1)
        ao_row.addWidget(self.ao_label)
        f.addRow("", ao_row)
        f.addRow("Height:", self.s_height)
        self.normalize = QCheckBox("Normalize height (stretch to 0..1)")
        f.addRow("", self.normalize)
        main.addWidget(self.g_albedo)

        # --- normal + roughness
        self.g_normal = QGroupBox("Normal + Roughness  ->  <name>_normal_roughness")
        self.g_normal.setCheckable(True)
        f = QFormLayout(self.g_normal)
        self.s_normal = MapSlot("Normal", required=True)
        f.addRow("Normal:", self.s_normal)
        self.dx = QCheckBox("DirectX normal (flip green channel to OpenGL)")
        f.addRow("", self.dx)
        self.s_rough = MapSlot("Roughness")
        f.addRow("Roughness:", self.s_rough)
        r_row = QHBoxLayout()
        self.rough_ch = QComboBox()
        self.rough_ch.addItems(CHANNELS)
        self.rough_ch.setToolTip("Use G for ARM/ORM maps")
        self.gloss = QCheckBox("Is glossiness/smoothness (invert)")
        r_row.addWidget(QLabel("Channel:"))
        r_row.addWidget(self.rough_ch)
        r_row.addSpacing(16)
        r_row.addWidget(self.gloss)
        r_row.addStretch(1)
        f.addRow("", r_row)
        main.addWidget(self.g_normal)

        # --- output
        g_out = QGroupBox("Output")
        f = QFormLayout(g_out)
        row = QHBoxLayout()
        self.out = QLineEdit()
        self.out.setPlaceholderText(f"filled in when a material is chosen (<folder>/{OUT_SUBDIR})")
        out_pick = QPushButton("...")
        out_pick.setFixedWidth(32)
        out_pick.clicked.connect(self._pick_out)
        row.addWidget(self.out, 1)
        row.addWidget(out_pick)
        f.addRow("Folder:", row)
        row = QHBoxLayout()
        self.size = QComboBox()
        self.size.addItems(SIZES)
        self.size.setToolTip("Terrain3D requires the same size for every terrain texture")
        self.fmt = QComboBox()
        self.fmt.addItems(["dds", "png"])
        self.fmt.setToolTip("dds = BC3/DXT5 with mipmaps (recommended). png = Godot compresses on import")
        # Separate normal format: DXT5 degrades normal maps, so png is the default here.
        self.nfmt = QComboBox()
        self.nfmt.addItems(["png", "dds"])
        self.nfmt.setToolTip("png recommended: DXT5 causes visible blocks in the relief")
        row.addWidget(QLabel("Size:"))
        row.addWidget(self.size)
        row.addSpacing(16)
        row.addWidget(QLabel("Albedo:"))
        row.addWidget(self.fmt)
        row.addSpacing(16)
        row.addWidget(QLabel("Normal:"))
        row.addWidget(self.nfmt)
        row.addStretch(1)
        f.addRow("", row)
        main.addWidget(g_out)

        # --- generate
        row = QHBoxLayout()
        self.go = QPushButton("Generate")
        self.go.setMinimumHeight(36)
        self.go.clicked.connect(self._generate)
        self.busy = QProgressBar()
        self.busy.setRange(0, 1)
        self.busy.setTextVisible(False)
        row.addWidget(self.go, 1)
        row.addWidget(self.busy, 2)
        main.addLayout(row)

        # --- preview
        prev = QGroupBox("Result")
        grid = QGridLayout(prev)
        self.p_albedo = PreviewBox("albedo (RGB)")
        self.p_height = PreviewBox("height (A)")
        self.p_normal = PreviewBox("normal (RGB)")
        self.p_rough = PreviewBox("roughness (A)")
        for i, w in enumerate((self.p_albedo, self.p_height, self.p_normal, self.p_rough)):
            grid.addWidget(w, 0, i)
        main.addWidget(prev)

        self.log = QPlainTextEdit()
        self.log.setReadOnly(True)
        self.log.setMaximumHeight(110)
        main.addWidget(self.log)

        self.bridge = LogBridge()
        self.bridge.message.connect(self.log.appendPlainText)
        t3dpack.log.addHandler(self.bridge.handler)
        t3dpack.log.propagate = False

        self._load_settings()

    # ---- persistence: remembers size, formats and AO strength between sessions

    def _load_settings(self) -> None:
        self.size.setCurrentText(self.settings.value("size", "Original"))
        self.fmt.setCurrentText(self.settings.value("format", "dds"))
        self.nfmt.setCurrentText(self.settings.value("normal_format", "png"))
        self.ao_slider.setValue(int(self.settings.value("ao_strength", 100)))

    def closeEvent(self, e):
        self.settings.setValue("size", self.size.currentText())
        self.settings.setValue("format", self.fmt.currentText())
        self.settings.setValue("normal_format", self.nfmt.currentText())
        self.settings.setValue("ao_strength", self.ao_slider.value())
        super().closeEvent(e)

    # ---- input

    def _pick_folder(self) -> None:
        d = QFileDialog.getExistingDirectory(self, "Material folder", self.folder.text())
        if d:
            self._autodetect(Path(d))

    def _pick_out(self) -> None:
        d = QFileDialog.getExistingDirectory(self, "Output folder", self.out.text())
        if d:
            self.out.setText(d)

    def _apply(self, files: dict[str, Path]) -> None:
        """Fills the slots from the map types detected by t3dpack."""
        if "albedo" in files:
            self.s_albedo.set_path(files["albedo"])
        ao = files.get("ao") or files.get("arm")
        if ao:
            self.s_ao.set_path(ao)
        if "height" in files:
            self.s_height.set_path(files["height"])
        normal_kind = next((k for k in ("normal_gl", "normal_dx", "normal") if k in files), None)
        if normal_kind:
            self.s_normal.set_path(files[normal_kind])
            self.dx.setChecked(normal_kind == "normal_dx")
        if "roughness" in files:
            self.s_rough.set_path(files["roughness"])
            self.rough_ch.setCurrentText("R")
            self.gloss.setChecked(False)
        elif "gloss" in files:
            self.s_rough.set_path(files["gloss"])
            self.rough_ch.setCurrentText("R")
            self.gloss.setChecked(True)
        elif "arm" in files:
            self.s_rough.set_path(files["arm"])
            self.rough_ch.setCurrentText("G")
            self.gloss.setChecked(False)

    def _suggest_out(self, folder: Path, force: bool = True) -> None:
        """Sets the output to <folder>/output. The folder is only created on generate
        (mkdir in the worker), so browsing materials doesn't leave empty folders behind."""
        if force or not self.out.text().strip():
            self.out.setText(str(folder / OUT_SUBDIR))

    def _autodetect(self, folder: Path) -> None:
        if not folder.is_dir():
            self.log.appendPlainText(f"[error] invalid folder: {folder}")
            return
        for s in (self.s_albedo, self.s_ao, self.s_height, self.s_normal, self.s_rough):
            s.set_path(None)
        self.folder.setText(str(folder))
        self.name.setText(folder.name)
        self._suggest_out(folder)
        files = t3dpack.detect(folder)
        self._apply(files)
        found = ", ".join(f"{k}={v.name}" for k, v in files.items()) or "nothing"
        self.log.appendPlainText(f"detected in {folder.name}: {found}")

    def dragEnterEvent(self, e):
        if e.mimeData().hasUrls():
            e.acceptProposedAction()

    def dropEvent(self, e):
        """Folder dropped on the window: auto-detect. Loose files: classify each by name."""
        paths = [Path(u.toLocalFile()) for u in e.mimeData().urls()]
        dirs = [p for p in paths if p.is_dir()]
        if dirs:
            self._autodetect(dirs[0])
            return
        files = {}
        for p in paths:
            kind = t3dpack.classify(p)
            if kind:
                files[kind] = p
            else:
                self.log.appendPlainText(f"[warning] type not recognized from name: {p.name}")
        self._apply(files)
        if files:
            self._suggest_out(paths[0].parent)
            if not self.name.text():
                self.name.setText(paths[0].parent.name)

    # ---- generation

    def _generate(self) -> None:
        errors = []
        do_albedo = self.g_albedo.isChecked()
        do_normal = self.g_normal.isChecked()
        if not (do_albedo or do_normal):
            errors.append("enable at least one group")
        if do_albedo and not (self.s_albedo.path() and self.s_albedo.path().is_file()):
            errors.append("albedo not found")
        if do_normal and not (self.s_normal.path() and self.s_normal.path().is_file()):
            errors.append("normal not found")
        for slot, label in ((self.s_ao, "AO"), (self.s_height, "height"), (self.s_rough, "roughness")):
            if slot.path() and not slot.path().is_file():
                errors.append(f"{label} file doesn't exist")
        name = self.name.text().strip()
        if not name:
            errors.append("set the name")
        if not self.out.text().strip():
            errors.append("set the output folder")
        if "dds" in (self.fmt.currentText(), self.nfmt.currentText()) and not shutil.which("magick"):
            errors.append("ImageMagick ('magick') not found; use png")
        if errors:
            QMessageBox.warning(self, "Missing", "\n".join(errors))
            return

        size_txt = self.size.currentText()
        job = {
            "name": name,
            "out": Path(self.out.text().strip()).expanduser(),
            "size": None if size_txt == "Original" else (int(size_txt), int(size_txt)),
            "format": self.fmt.currentText(),
            "normal_format": self.nfmt.currentText(),
            "do_albedo": do_albedo,
            "albedo": self.s_albedo.path(),
            "ao": self.s_ao.path(),
            "use_ao": self.use_ao.isChecked(),
            "ao_strength": self.ao_slider.value() / 100.0,
            "height": self.s_height.path(),
            "normalize_height": self.normalize.isChecked(),
            "do_normal": do_normal,
            "normal": self.s_normal.path(),
            "dx": self.dx.isChecked(),
            "rough": self.s_rough.path(),
            "rough_channel": CHANNELS.index(self.rough_ch.currentText()),
            "invert_rough": self.gloss.isChecked(),
        }

        for p in (self.p_albedo, self.p_height, self.p_normal, self.p_rough):
            p.set_image(None)
        self.go.setEnabled(False)
        self.busy.setRange(0, 0)  # indeterminate mode: ImageMagick doesn't report progress
        self.log.appendPlainText(f"--- {name}")

        self.worker = PackWorker(job)
        self.worker.log.connect(self.log.appendPlainText)
        self.worker.preview.connect(self._show_preview)
        self.worker.done.connect(self._finished)
        self.worker.start()

    def _show_preview(self, which: str, rgb: QImage, alpha: QImage) -> None:
        if which == "albedo":
            self.p_albedo.set_image(rgb)
            self.p_height.set_image(alpha)
        else:
            self.p_normal.set_image(rgb)
            self.p_rough.set_image(alpha)

    def _finished(self, ok: bool) -> None:
        self.go.setEnabled(True)
        self.busy.setRange(0, 1)
        self.busy.setValue(1 if ok else 0)
        self.log.appendPlainText("done" if ok else "failed")


def main() -> int:
    app = QApplication(sys.argv)
    app.setApplicationName("t3dpack")
    w = MainWindow()
    w.resize(760, 820)
    w.show()
    return app.exec()


if __name__ == "__main__":
    sys.exit(main())
