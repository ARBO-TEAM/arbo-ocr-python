"""Downloads the arboOCR release binary matching this package's pinned
version for the host OS, and extracts it to `arbo_ocr/bin/<platform>/`.
Re-downloads whenever the pin changes — the installed tag is recorded in a
marker file alongside the binary, see _needs_install().
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
# v0.3.0 is the first release with model auto-download, so the
# `no_download` / `models_url` Engine options and download_models() below
# all work against the binary this package installs. It also ships
# onnxruntime_providers_shared, missing from earlier archives, without which
# --cuda / --tensorrt could not load from a release build.
PINNED_VERSION = "v0.3.0"

# Name of the file written inside bin/<platform>/ recording which release tag
# the binary sitting next to it was extracted from. Dot-prefixed so it never
# collides with an archive member.
VERSION_MARKER = ".arboocr-version"

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
    omitted) unless PINNED_VERSION is already installed, and returns its
    path. Raises RuntimeError if the platform is unsupported or the
    download/extract fails.

    Only ever touches `arbo_ocr/bin/<platform>/`. A binary a caller supplies
    itself (Engine's / download_models' `bin_path`) never comes through here
    and is used exactly as given — see _needs_install()."""
    platform_name = platform_name or detect_platform()
    if platform_name is None:
        raise RuntimeError(
            "Unsupported OS for auto-download. Download a release manually "
            f"from https://github.com/{REPO}/releases and pass bin_path to Engine."
        )

    target_dir = _PACKAGE_DIR / "bin" / platform_name
    bin_name = "arboocr_demo.exe" if platform_name == "windows-x64" else "arboocr_demo"
    bin_path = target_dir / bin_name

    if not _needs_install(target_dir, bin_path):
        return bin_path

    asset = (
        "arboocr-windows-x64.zip" if platform_name == "windows-x64" else "arboocr-linux-x64.tar.gz"
    )
    url = f"https://github.com/{REPO}/releases/download/{PINNED_VERSION}/{asset}"

    download_and_extract(url, target_dir, asset)

    if platform_name == "linux-x64":
        bin_path.chmod(0o755)

    # Only once the new binary is actually on disk — a marker written any
    # earlier (or on a failed download) would claim a version that isn't
    # there, and the next run would trust it and skip the download.
    _write_version_marker(target_dir)

    return bin_path


def _needs_install(target_dir: Path, bin_path: Path) -> bool:
    """Whether bin/<platform>/ has to be (re)populated for PINNED_VERSION.

    DO NOT reduce this back to `if bin_path.is_file(): return bin_path` —
    that was the bug. ensure_installed() used to short-circuit on "some
    binary is there", which made bumping PINNED_VERSION a silent no-op for
    everyone who already had one: `arbo-ocr-install` reported success, the
    previous release's executable stayed exactly where it was, and Engine
    carried on driving a CLI contract it no longer matched. Bumping the pin
    to v0.3.0 that way left users on a v0.2.0 binary with no
    --no-download / --models-url / --download-models and no
    onnxruntime_providers_shared, i.e. with every feature this package
    advertises silently absent. pip does not clean the directory up either:
    the binary is fetched at runtime, so it is not in the wheel's RECORD and
    survives an upgrade untouched (as does a source checkout, and an
    editable install).

    arbo-ocr-go and arbo-ocr-rust fixed the identical bug by putting the
    version in the cache path (…/arbo-ocr-go/v0.3.0/windows-x64). That fits
    them because they cache *outside* the package, where a fresh directory
    per version costs nothing. This package's bin/ lives *inside* the
    installed package, and default_bin_path() — hence Engine's default, the
    README, and any script anyone wrote against it — resolves to
    arbo_ocr/bin/<platform>/; adding a version segment would move that path
    on every release. So this follows arbo-ocr-php instead: same directory,
    plus a marker file recording what was installed into it.

    A marker that is missing, unreadable or empty counts as a mismatch and
    triggers a fresh download: an install made before the marker existed is
    of unknown provenance, and assuming such an install is current is
    precisely the failure this replaced. Re-downloading a binary that turns
    out to have been fine costs one archive; skipping one that wasn't costs
    silent wrong behaviour.
    """
    if not bin_path.is_file():
        return True

    return _installed_version(target_dir) != PINNED_VERSION


def _installed_version(target_dir: Path) -> Optional[str]:
    """The release tag recorded in bin/<platform>/.arboocr-version, or None
    when there is no readable, non-empty marker — which is how an install
    predating the marker reports "unknown version". None never equals
    PINNED_VERSION, so it always reads as a mismatch upstream."""
    try:
        recorded = (target_dir / VERSION_MARKER).read_text(encoding="utf-8").strip()
    except (OSError, UnicodeDecodeError):
        return None

    return recorded or None


def _write_version_marker(target_dir: Path) -> None:
    (target_dir / VERSION_MARKER).write_text(PINNED_VERSION + "\n", encoding="utf-8")


def download_models(
    *,
    ocr_version: Optional[str] = None,
    model_type: Optional[str] = None,
    models_url: Optional[str] = None,
    bin_path: Optional[str] = None,
) -> None:
    """Prefetches the OCR models for `ocr_version`/`model_type` into arboOCR's
    own tag-scoped model cache, by shelling out to `arboocr_demo
    --download-models` — which downloads, SHA-256-verifies, and exits without
    running OCR. Handy in a Docker build or CI step so the first
    `Engine.recognize()` call does no network I/O.

    Deliberately thin: arboOCR itself owns the cache layout, the URLs, and the
    hash verification, so this is a subprocess call and an exit-code check,
    not a download manager. Unset arguments emit no flag at all, leaving the
    binary's own defaults in charge.

    Requires an `arboocr_demo` from arboOCR v0.3.0 or newer, which is what
    PINNED_VERSION installs. An older binary passed via `bin_path` predates
    the flag and will exit 1 with a usage error.

    Raises RuntimeError on any non-zero exit, with the binary's stderr
    attached (it is silent on stderr unless it got --log-level, so that text
    may well be empty — trust the code, not the emptiness).
    """
    import subprocess

    if bin_path is None:
        command = [str(default_bin_path())]
    elif isinstance(bin_path, (list, tuple)):
        command = list(bin_path)
    else:
        command = [bin_path]

    argv = [*command, "--download-models"]
    for flag, value in (
        ("--ocr-version", ocr_version),
        ("--model-type", model_type),
        ("--models-url", models_url),
    ):
        if value is not None:
            argv += [flag, str(value)]

    result = subprocess.run(argv, capture_output=True, text=True, encoding="utf-8")

    if result.returncode != 0:
        raise RuntimeError(
            f"arboocr_demo --download-models exited with code {result.returncode}. "
            "Note that arboOCR releases older than "
            f"{PINNED_VERSION} do not support --download-models and report "
            f"that as exit code 1.\n{result.stderr}"
        )


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

        _flatten_archive_root(target_dir, asset_name)
    finally:
        tmp_path.unlink(missing_ok=True)


def _flatten_archive_root(target_dir: Path, asset_name: str) -> None:
    """The linux tar.gz wraps everything in one top-level folder
    (arboocr-linux-x64/...); the windows zip is already flat. Where that
    folder is present, move its contents up into target_dir so callers get
    bin/<platform>/arboocr_demo directly, not
    bin/<platform>/arboocr-linux-x64/arboocr_demo.

    The folder is located by name, derived from the asset filename, rather
    than by the old "did the extract leave exactly one entry here?" test.
    That test silently only worked on a clean directory, so it stopped
    firing the moment re-downloads became possible (see _needs_install):
    unpacking over an existing install leaves the previous version's files
    sitting beside the new folder, the entry count is no longer 1, and the
    new binary would stay stranded one level down while the stale one keeps
    the path Engine actually looks at — with a freshly written version
    marker now asserting a release that is not on disk.
    """
    import shutil

    for suffix in (".zip", ".tar.gz"):
        if asset_name.endswith(suffix):
            subdir = target_dir / asset_name[: -len(suffix)]
            break
    else:
        return

    if not subdir.is_dir():
        return

    for item in subdir.iterdir():
        dest = target_dir / item.name
        # On a re-download dest is the previous version's file. replace(),
        # not rename(): rename() refuses to clobber an existing file on
        # Windows. Directories can't be clobbered either way, so clear those
        # out first — by this point the replacement is already fully
        # extracted, so there is nothing left to lose.
        if dest.is_dir() and not dest.is_symlink():
            shutil.rmtree(dest)
        item.replace(dest)

    subdir.rmdir()


def main() -> None:
    """Entry point for the `arbo-ocr-install` console script."""
    import sys

    try:
        path = ensure_installed()
        print(f"[arbo-ocr-python] Installed arboocr_demo ({PINNED_VERSION}) at {path}")
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
