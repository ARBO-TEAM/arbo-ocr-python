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
import tarfile

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
