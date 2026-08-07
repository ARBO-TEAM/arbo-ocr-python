"""Downloads the arboOCR release binary matching this package's pinned
version for the host OS, and extracts it to `arbo_ocr/bin/<platform>/`.
Never raises on failure when called via the `arbo-ocr-install` console
script's main() — arboOCR still works if the caller points Engine at a
manually-downloaded binary via the `bin_path` option. (The lower-level
functions here — download_and_extract, ensure_installed — do raise; only
`main()` catches and reports.)
"""

from __future__ import annotations

import platform as _platform
from pathlib import Path
from typing import Optional

# shutil/tarfile/tempfile/urllib.request/zipfile are imported lazily inside
# download_and_extract()/main() below, not here: engine.py imports this
# module just for detect_platform()/default_bin_path() on every recognize()
# call, and those heavy stdlib modules (urllib.request alone pulls in ssl,
# http.client, etc.) add ~50ms of avoidable interpreter startup to every
# subprocess spawn if imported eagerly - measured via bare `python -c
# "import arbo_ocr"` timing during benchmarking. Only the explicit
# `arbo-ocr-install` path actually needs them.

REPO = "wafik/ArboOCR"
PINNED_VERSION = "v0.2.0"

_PACKAGE_DIR = Path(__file__).resolve().parent


def detect_platform() -> Optional[str]:
    system = _platform.system()
    if system == "Windows":
        return "windows-x64"
    if system == "Linux":
        return "linux-x64"
    return None


def default_bin_path() -> Path:
    platform_name = detect_platform() or "linux-x64"
    bin_name = "arboocr_demo.exe" if platform_name == "windows-x64" else "arboocr_demo"
    return _PACKAGE_DIR / "bin" / platform_name / bin_name


def ensure_installed(platform_name: Optional[str] = None) -> Path:
    """Downloads+extracts the binary for `platform_name` (auto-detected if
    omitted) if not already present, and returns its path. Raises
    RuntimeError if the platform is unsupported or the download/extract
    fails."""
    platform_name = platform_name or detect_platform()
    if platform_name is None:
        raise RuntimeError(
            "Unsupported OS for auto-download. Download a release manually "
            f"from https://github.com/{REPO}/releases and pass bin_path to Engine."
        )

    target_dir = _PACKAGE_DIR / "bin" / platform_name
    bin_name = "arboocr_demo.exe" if platform_name == "windows-x64" else "arboocr_demo"
    bin_path = target_dir / bin_name

    if bin_path.is_file():
        return bin_path

    asset = (
        "arboocr-windows-x64.zip" if platform_name == "windows-x64" else "arboocr-linux-x64.tar.gz"
    )
    url = f"https://github.com/{REPO}/releases/download/{PINNED_VERSION}/{asset}"

    download_and_extract(url, target_dir, asset)

    if platform_name == "linux-x64":
        bin_path.chmod(0o755)

    return bin_path


def download_and_extract(url: str, target_dir: Path, asset_name: str) -> None:
    import shutil
    import tarfile
    import tempfile
    import urllib.request
    import zipfile

    target_dir.mkdir(parents=True, exist_ok=True)

    suffix = ".zip" if asset_name.endswith(".zip") else ".tar.gz"
    with tempfile.NamedTemporaryFile(suffix=suffix, delete=False) as tmp:
        tmp_path = Path(tmp.name)

    try:
        with urllib.request.urlopen(url, timeout=120) as response, open(tmp_path, "wb") as f:
            shutil.copyfileobj(response, f)

        if suffix == ".zip":
            with zipfile.ZipFile(tmp_path) as zf:
                zf.extractall(target_dir)
        else:
            with tarfile.open(tmp_path, "r:gz") as tf:
                tf.extractall(target_dir)  # noqa: S202 — trusted first-party release asset

        _flatten_single_subdir(target_dir)
    finally:
        tmp_path.unlink(missing_ok=True)


def _flatten_single_subdir(target_dir: Path) -> None:
    """The release archives contain one top-level folder (e.g.
    arboocr-windows-x64/...). Move its contents up into target_dir so callers
    get bin/<platform>/arboocr_demo directly, not
    bin/<platform>/arboocr-windows-x64/arboocr_demo."""
    entries = list(target_dir.iterdir())
    if len(entries) != 1 or not entries[0].is_dir():
        return
    subdir = entries[0]
    for item in subdir.iterdir():
        item.rename(target_dir / item.name)
    subdir.rmdir()


def main() -> None:
    """Entry point for the `arbo-ocr-install` console script."""
    import sys

    try:
        path = ensure_installed()
        print(f"[arbo-ocr-python] Installed arboocr_demo at {path}")
    except Exception as e:  # noqa: BLE001 — top-level CLI entry point, must not crash
        print(
            f"[arbo-ocr-python] Could not auto-download arboOCR binary: {e}\n"
            f"Download manually from https://github.com/{REPO}/releases and "
            "pass bin_path to Engine.",
            file=sys.stderr,
        )
        sys.exit(1)


if __name__ == "__main__":
    main()
