"""Tests for arbo_ocr.installer.

download_and_extract runs against a real local tar.gz, with no network
access involved (urllib's urlopen supports file:// URLs natively). Python's
tarfile.open() doesn't share the PHP PharData extension quirk the original
arbo-ocr-php test worked around, but the test is still a valuable real check
that download_and_extract (and _flatten_archive_root) correctly unwraps the
release archive's single top-level directory.

ensure_installed's version guard is exercised with the download stubbed out
entirely — the assertion is about *whether* it fetches, so no test here ever
reaches GitHub.
"""

from __future__ import annotations

import io
import subprocess
import tarfile

import pytest

from arbo_ocr import installer
from arbo_ocr.engine import Engine


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


# The marker filename is spelled out here rather than read off
# installer.VERSION_MARKER on purpose: it is a cross-package contract
# (arbo-ocr-php writes the same name), and hard-coding it keeps these tests
# asserting observable behaviour instead of mirroring the implementation.
_MARKER = ".arboocr-version"


def test_download_and_extract_flattens_over_an_existing_install(tmp_path):
    """Re-download case: the archive's top-level folder has to be unwrapped
    even though target_dir already holds the previous version's files. The
    old "exactly one entry in target_dir" heuristic only ever held for a
    clean directory, so it left the new binary stranded one level down and
    the stale one still sitting at the path Engine reads."""
    gz_path = tmp_path / "arboocr-linux-x64.tar.gz"
    with tarfile.open(gz_path, "w:gz") as tf:
        _add_bytes(tf, "arboocr-linux-x64/arboocr_demo", b"new-binary")
        _add_bytes(tf, "arboocr-linux-x64/onnxruntime_providers_shared.so", b"new-lib")

    target_dir = tmp_path / "target"
    target_dir.mkdir()
    (target_dir / "arboocr_demo").write_bytes(b"stale-binary")
    (target_dir / "LICENSE").write_bytes(b"stale-license")

    installer.download_and_extract(
        gz_path.as_uri(), target_dir, "arboocr-linux-x64.tar.gz"
    )

    assert (target_dir / "arboocr_demo").read_bytes() == b"new-binary"
    assert (target_dir / "onnxruntime_providers_shared.so").read_bytes() == b"new-lib"
    # Nothing may be left nested inside the archive's wrapper folder.
    assert not (target_dir / "arboocr-linux-x64").exists()


def _stub_install(monkeypatch, tmp_path, *, binary_contents=b"new-binary"):
    """Points the installer at tmp_path instead of the real package
    directory and replaces the network fetch with a local file write,
    recording the URLs it was asked for. Nothing here touches GitHub."""
    calls: list[str] = []

    def fake_download_and_extract(url, target_dir, asset_name):
        calls.append(url)
        target_dir.mkdir(parents=True, exist_ok=True)
        (target_dir / "arboocr_demo").write_bytes(binary_contents)

    monkeypatch.setattr(installer, "_PACKAGE_DIR", tmp_path)
    monkeypatch.setattr(installer, "download_and_extract", fake_download_and_extract)
    return calls


def _preinstall(tmp_path, *, marker: str | None, contents: bytes = b"stale-binary"):
    """Simulates an already-populated arbo_ocr/bin/linux-x64/, optionally
    carrying a version marker."""
    target_dir = tmp_path / "bin" / "linux-x64"
    target_dir.mkdir(parents=True)
    (target_dir / "arboocr_demo").write_bytes(contents)
    if marker is not None:
        (target_dir / _MARKER).write_text(marker, encoding="utf-8")
    return target_dir


def test_ensure_installed_redownloads_when_marker_is_absent(monkeypatch, tmp_path):
    """The exact case that shipped broken: a binary left behind by an older
    release of this package, with no marker at all. Its provenance is
    unknown, so it must not be trusted."""
    calls = _stub_install(monkeypatch, tmp_path)
    target_dir = _preinstall(tmp_path, marker=None)

    bin_path = installer.ensure_installed("linux-x64")

    assert calls == [
        "https://github.com/wafik/ArboOCR/releases/download/"
        f"{installer.PINNED_VERSION}/arboocr-linux-x64.tar.gz"
    ]
    assert bin_path.read_bytes() == b"new-binary"
    assert (target_dir / _MARKER).read_text(
        encoding="utf-8"
    ).strip() == installer.PINNED_VERSION


@pytest.mark.parametrize("marker", ["v0.2.0", "", "   \n"])
def test_ensure_installed_redownloads_when_marker_does_not_match(
    monkeypatch, tmp_path, marker
):
    """A recorded version other than the pin means the pin was bumped since
    this binary was installed; an empty/whitespace marker is unreadable and
    counts the same way."""
    calls = _stub_install(monkeypatch, tmp_path)
    _preinstall(tmp_path, marker=marker)

    bin_path = installer.ensure_installed("linux-x64")

    assert len(calls) == 1
    assert bin_path.read_bytes() == b"new-binary"


def test_ensure_installed_skips_download_when_marker_matches(monkeypatch, tmp_path):
    """The other half of the guard: a matching marker must still short-
    circuit, or every run re-downloads ~40 MB."""
    calls = _stub_install(monkeypatch, tmp_path)
    _preinstall(tmp_path, marker=installer.PINNED_VERSION + "\n")

    bin_path = installer.ensure_installed("linux-x64")

    assert calls == []
    assert bin_path.read_bytes() == b"stale-binary"  # i.e. left untouched


def test_ensure_installed_does_not_write_marker_when_download_fails(
    monkeypatch, tmp_path
):
    """A marker written despite a failed download would claim a version that
    is not on disk, and the next run would trust it and skip the retry."""
    target_dir = _preinstall(tmp_path, marker="v0.2.0")

    def exploding_download(url, target_dir, asset_name):
        raise RuntimeError("simulated network failure")

    monkeypatch.setattr(installer, "_PACKAGE_DIR", tmp_path)
    monkeypatch.setattr(installer, "download_and_extract", exploding_download)

    with pytest.raises(RuntimeError, match="simulated network failure"):
        installer.ensure_installed("linux-x64")

    assert (target_dir / _MARKER).read_text(
        encoding="utf-8"
    ).strip() == "v0.2.0"


def test_explicit_bin_path_is_never_second_guessed(monkeypatch, tmp_path):
    """A binary the caller supplies is theirs: no marker check, no download,
    no overwrite — even though it carries no version marker at all."""
    monkeypatch.setattr(installer, "_PACKAGE_DIR", tmp_path)
    monkeypatch.setattr(
        installer,
        "download_and_extract",
        lambda *a, **kw: pytest.fail("explicit bin_path must never trigger a download"),
    )
    captured = _capture_argv(monkeypatch)

    custom = tmp_path / "my-own" / "arboocr_demo"
    custom.parent.mkdir()
    custom.write_bytes(b"users-own-binary")

    Engine(bin_path=str(custom))
    installer.download_models(bin_path=str(custom))

    assert captured["argv"] == [str(custom), "--download-models"]
    assert custom.read_bytes() == b"users-own-binary"
    assert not (custom.parent / _MARKER).exists()


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
