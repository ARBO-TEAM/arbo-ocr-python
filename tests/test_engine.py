"""Tests for arbo_ocr.engine.Engine, ported from the arbo-ocr-php test
suite. Instead of a real arboocr_demo binary, these tests point Engine at
tests/fixtures/fake_arboocr.py — run via the current Python interpreter as
an array-form bin_path (["python", "fake_arboocr.py", ...]). Engine already
supports array-form bin_path (see Engine._bin_command in engine.py), and
using it here also means the len(self._bin_command) == 1 "does this file
exist" check in Engine.recognize() never fires for these tests — the fake
binary is a script invoked via the interpreter, not a standalone file at a
single path.
"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

from arbo_ocr.engine import Engine
from arbo_ocr.exceptions import OcrError

FIXTURE_PATH = Path(__file__).parent / "fixtures" / "fake_arboocr.py"


def fake_bin(extra_args=()) -> list[str]:
    """Array-form bin_path pointing at the fake arboocr_demo stand-in,
    invoked with the current interpreter."""
    return [sys.executable, str(FIXTURE_PATH), *extra_args]


def test_recognize_parses_successful_json_output():
    engine = Engine(bin_path=fake_bin())

    result = engine.recognize("/some/page.jpg")

    assert result.backend == "cpu"
    assert result.image == "page.jpg"
    assert result.elapsed_ms == 12.5
    assert len(result.lines) == 1
    assert result.lines[0].text == "hello"
    assert result.lines[0].score == 0.9
    assert result.lines[0].polygon[0]["x"] == 1.0


def test_bool_flags_use_single_token_form():
    # Regression test: cxxopts binds a bool flag's value only via "=" — a
    # bare "--angle" followed by a separate "true"/"false" token leaves the
    # flag implicitly true and the value ignored (confirmed against the
    # real arboocr_demo binary). _flags_from_options() must always emit bool
    # options as a single "--flag=value" token.
    engine = Engine(
        bin_path=fake_bin(),
        use_angle_cls=False,
        use_cuda=True,
        use_tensorrt=False,
        use_fp16=False,
        use_clahe=True,
    )

    flags = engine._flags_from_options()

    for expected in ("--angle=false", "--cuda=true", "--tensorrt=false",
                      "--fp16=false", "--clahe=true"):
        assert expected in flags

    for bare in ("--angle", "--cuda", "--tensorrt", "--fp16", "--clahe"):
        assert bare not in flags


def test_recognize_throws_on_nonzero_exit():
    engine = Engine(bin_path=fake_bin(["--fail"]))

    with pytest.raises(OcrError, match="exited with code"):
        engine.recognize("/some/page.jpg")


def test_recognize_handles_large_stderr_without_hanging():
    # Regression test (renamed from the PHP version's deadlock-focused name):
    # PHP's proc_open could deadlock if a child filled the stderr pipe buffer
    # before its stdout was drained. Python's subprocess.run(capture_output=True)
    # reads stdout/stderr concurrently internally (threads on Windows,
    # select() on POSIX) via Popen.communicate(), so it can't deadlock that
    # way — this test just confirms large stderr output doesn't otherwise
    # break parsing. (No pytest-timeout plugin is used here; if this ever
    # did hang, it would need one to fail fast instead of blocking the run.)
    engine = Engine(bin_path=fake_bin(["--noisy-stderr"]))

    result = engine.recognize("/some/page.jpg")

    assert result.backend == "cpu"


def test_recognize_throws_on_unparseable_output():
    engine = Engine(bin_path=fake_bin(["--garbage"]))

    with pytest.raises(OcrError):
        engine.recognize("/some/page.jpg")


def test_recognize_throws_when_binary_missing():
    engine = Engine(bin_path="/no/such/binary")

    with pytest.raises(OcrError, match="not found"):
        engine.recognize("/some/page.jpg")


def test_flags_from_options_map_to_cli_flags():
    # Indirect check: models_dir/use_angle_cls/etc. must reach argv without
    # erroring subprocess and must not break the fake binary's parsing (it
    # only reads --image, so any well-formed extra flags are fine).
    engine = Engine(
        bin_path=fake_bin(),
        models_dir="models",
        use_angle_cls=True,
        use_cuda=False,
    )

    result = engine.recognize("/some/page.jpg")

    assert result.backend == "cpu"
