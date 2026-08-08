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


def test_recognize_handles_non_ascii_utf8_output():
    # Regression test: subprocess.run(text=True) with no explicit encoding
    # decodes with the locale default (cp1252 on Windows), not UTF-8. Real
    # recognized text from arboocr_demo (always UTF-8) can contain a byte
    # cp1252 has no mapping for, raising UnicodeDecodeError inside
    # subprocess's internal reader thread - silently swallowed there, only
    # visible here as stdout being None despite returncode 0. Engine must
    # pass encoding="utf-8" explicitly.
    engine = Engine(bin_path=fake_bin(["--nonascii"]))

    result = engine.recognize("/some/page.jpg")

    assert result.lines[0].text == "Ёlka — café"


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


def test_v020_value_flags_map_to_cli_flags():
    # arboOCR v0.2.0 accuracy flags + --log-level. Each is a separate
    # "--flag" / "<value>" token pair, matching the pre-existing string-flag
    # form (cxxopts accepts both "--flag value" and "--flag=value" for
    # value-taking options; only bools are "=" -only).
    engine = Engine(
        bin_path=fake_bin(),
        min_confidence=0.42,
        rec_batch_num=12,
        det_limit_side_len=1280,
        log_level="warn",
    )

    flags = engine._flags_from_options()

    for flag, value in (("--min-confidence", "0.42"), ("--rec-batch-num", "12"),
                        ("--det-limit-side-len", "1280"), ("--log-level", "warn")):
        assert flag in flags
        assert flags[flags.index(flag) + 1] == value


def test_word_boxes_flag_uses_single_token_form_and_parses_words():
    # --word-boxes is a cxxopts bool, so it must go through _BOOL_FLAGS'
    # "--flag=value" path like --angle/--cuda, not the "--flag value" path.
    engine = Engine(bin_path=fake_bin(), word_boxes=True)

    assert "--word-boxes=true" in engine._flags_from_options()
    assert "--word-boxes" not in engine._flags_from_options()

    result = engine.recognize("/some/page.jpg")

    assert [w["text"] for w in result.lines[0].words] == ["hel", "lo"]
    assert result.lines[0].words[0]["polygon"][0]["x"] == 1.0


def test_words_defaults_to_empty_when_not_requested():
    # arboOCR omits the "words" key entirely without --word-boxes; that is a
    # normal successful result, not a parse failure.
    engine = Engine(bin_path=fake_bin())

    result = engine.recognize("/some/page.jpg")

    assert result.lines[0].words == []


def test_model_download_options_emit_nothing_when_omitted():
    # THE regression test for everyone currently on installer.PINNED_VERSION.
    # That binary predates arboOCR's model auto-download and rejects an
    # unknown flag with exit code 1 (usage error), so adding the
    # `no_download` / `models_url` options must not change a single token of
    # the command line for callers who don't ask for them. Asserting on the
    # whole flag list, not just membership, is the point here — a default of
    # e.g. no_download=False in Engine.__init__ would still emit
    # "--no-download=false" and break every v0.2.0 user.
    engine = Engine(bin_path=fake_bin(), models_dir="models", model_type="tiny")

    flags = engine._flags_from_options()

    assert flags == ["--models-dir", "models", "--model-type", "tiny"]

    # And the same holds for a bare Engine with no options at all.
    assert Engine(bin_path=fake_bin())._flags_from_options() == []


def test_no_download_uses_single_token_bool_form():
    # --no-download is a cxxopts bool, so it goes through _BOOL_FLAGS'
    # "--flag=value" path like --angle/--cuda, not the "--flag value" path.
    engine = Engine(bin_path=fake_bin(), no_download=True)

    flags = engine._flags_from_options()

    assert "--no-download=true" in flags
    assert "--no-download" not in flags

    # Explicitly asking for the opposite must still emit the flag, matching
    # how every other bool option behaves.
    assert "--no-download=false" in Engine(
        bin_path=fake_bin(), no_download=False
    )._flags_from_options()


def test_models_url_maps_to_cli_flag():
    # A value-taking option, so the "--flag" / "<value>" token-pair form.
    url = "https://mirror.internal/arboocr/models/"
    engine = Engine(bin_path=fake_bin(), models_url=url)

    flags = engine._flags_from_options()

    assert "--models-url" in flags
    assert flags[flags.index("--models-url") + 1] == url


def test_model_download_options_reach_the_binary_without_erroring():
    # Mirrors test_flags_from_options_map_to_cli_flags: the fake binary only
    # reads --image, so this just confirms the new flags are well-formed
    # enough to travel through subprocess and back.
    engine = Engine(
        bin_path=fake_bin(),
        no_download=True,
        models_url="https://mirror.internal/arboocr/models/",
    )

    result = engine.recognize("/some/page.jpg")

    assert result.backend == "cpu"
