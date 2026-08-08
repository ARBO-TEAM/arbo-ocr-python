"""Integration test for arbo_ocr.installer.download_and_extract against a
real local tar.gz, with no network access involved (urllib's urlopen
supports file:// URLs natively). Python's tarfile.open() doesn't share the
PHP PharData extension quirk the original arbo-ocr-php test worked around,
but the test is still a valuable real check that download_and_extract (and
_flatten_single_subdir) correctly unwraps the release archive's single
top-level directory.
"""

from __future__ import annotations

import io
import subprocess
import tarfile

import pytest

from arbo_ocr import installer


def _add_bytes(tf: tarfile.TarFile, arcname: str, data: bytes) -> None:
    info = tarfile.TarInfo(name=arcname)
    info.size = len(data)
    tf.addfile(info, io.BytesIO(data))


def test_download_and_extract_handles_tar_gz_without_network_access(tmp_path):
    gz_path = tmp_path / "arboocr-linux-x64.tar.gz"
    binary_contents = b"fake-binary-contents"
    license_contents = b"fake-license"

    with tarfile.open(gz_path, "w:gz") as tf:
        _add_bytes(tf, "arboocr-linux-x64/arboocr_demo", binary_contents)
        _add_bytes(tf, "arboocr-linux-x64/LICENSE", license_contents)

    target_dir = tmp_path / "target"

    # Path.as_uri() produces a correctly-formed file:// URL on both POSIX
    # and Windows (e.g. file:///D:/... with the drive letter) — plain
    # f"file://{gz_path}" string interpolation breaks on Windows because
    # backslash path separators aren't valid in a URI.
    installer.download_and_extract(
        gz_path.as_uri(), target_dir, "arboocr-linux-x64.tar.gz"
    )

    bin_path = target_dir / "arboocr_demo"
    assert bin_path.is_file()
    assert bin_path.read_bytes() == binary_contents

    # Also confirm the sibling file was flattened up alongside it, not left
    # nested inside the now-removed arboocr-linux-x64/ wrapper folder.
    license_path = target_dir / "LICENSE"
    assert license_path.is_file()
    assert license_path.read_bytes() == license_contents


class _FakeCompletedProcess:
    def __init__(self, returncode: int = 0) -> None:
        self.returncode = returncode
        self.stdout = ""
        self.stderr = "simulated stderr"


def _capture_argv(monkeypatch, returncode: int = 0) -> dict:
    """Intercepts the subprocess.run() call download_models() makes, so the
    emitted argv can be asserted on without a real arboocr_demo — the same
    thing test_engine.py checks via Engine._flags_from_options()."""
    captured: dict = {}

    def fake_run(argv, **kwargs):
        captured["argv"] = argv
        return _FakeCompletedProcess(returncode)

    # download_models() imports subprocess lazily inside the function body,
    # so patching the module attribute here still takes effect at call time.
    monkeypatch.setattr(subprocess, "run", fake_run)
    return captured


def test_download_models_emits_download_models_flag_and_set_options(monkeypatch):
    captured = _capture_argv(monkeypatch)

    installer.download_models(
        ocr_version="PP-OCRv6",
        model_type="tiny",
        models_url="https://mirror.internal/arboocr/models/",
        bin_path="/fake/arboocr_demo",
    )

    assert captured["argv"] == [
        "/fake/arboocr_demo",
        "--download-models",
        "--ocr-version",
        "PP-OCRv6",
        "--model-type",
        "tiny",
        "--models-url",
        "https://mirror.internal/arboocr/models/",
    ]


def test_download_models_omits_unset_options(monkeypatch):
    # Unset arguments must not turn into empty-valued flags; arboOCR's own
    # defaults stay in charge.
    captured = _capture_argv(monkeypatch)

    installer.download_models(bin_path="/fake/arboocr_demo")

    assert captured["argv"] == ["/fake/arboocr_demo", "--download-models"]


def test_download_models_raises_on_nonzero_exit(monkeypatch):
    # Exit code 1 is exactly what an arboOCR release predating model
    # auto-download reports for the unknown --download-models flag, so the
    # error message has to point at that possibility.
    _capture_argv(monkeypatch, returncode=1)

    with pytest.raises(RuntimeError, match="--download-models"):
        installer.download_models(bin_path="/fake/arboocr_demo")
