"""Runs the prebuilt arboocr_demo binary via subprocess and parses its
--json output. Requires no C++ build — only the binary the installer
downloaded (or one you point at manually via the `bin_path` option).
"""

from __future__ import annotations

import json
import os
import subprocess
import tempfile
from pathlib import Path
from typing import Any

from .exceptions import OcrError
from .installer import default_bin_path
from .models import PageResult

_STRING_FLAGS = {
    "models_dir": "models-dir",
    "ocr_version": "ocr-version",
    "model_type": "model-type",
    "det_model_path": "det-model",
    "cls_model_path": "cls-model",
    "rec_model_path": "rec-model",
    "dict_path": "dict",
    # arboOCR v0.2.0. Without it the binary is silent on stderr; one of
    # "debug" | "info" | "warn" | "error" (arboocr_demo rejects anything
    # else with exit code 1).
    "log_level": "log-level",
    # arboOCR v0.3.0 (model auto-download) — directory URL to fetch missing
    # models from, e.g. an internal mirror. Older binaries have no such flag
    # and exit 1 (usage error) if it is passed, so see _BOOL_FLAGS'
    # "no_download" note for why omitting the option must stay a no-op.
    "models_url": "models-url",
}

# Same "--flag <value>" emission as _STRING_FLAGS (str() handles the
# conversion); kept separate only to document that these carry numbers.
# All three arrived in arboOCR v0.2.0's accuracy-flag set.
_NUMERIC_FLAGS = {
    "min_confidence": "min-confidence",  # float — drop lines below this score, 0 disables
    "rec_batch_num": "rec-batch-num",  # int — crops per recognition inference call
    "det_limit_side_len": "det-limit-side-len",  # int — longest image side for detection resize
}

_BOOL_FLAGS = {
    "use_angle_cls": "angle",
    "use_cuda": "cuda",
    "use_tensorrt": "tensorrt",
    "use_fp16": "fp16",
    "use_clahe": "clahe",
    # arboOCR v0.2.0 — adds a "words" array (a polygon per word, per
    # character for CJK) to every line in the JSON. Omitted entirely when
    # not requested, so LineResult.words is [] by default.
    "word_boxes": "word-boxes",
    # arboOCR v0.3.0 (model auto-download); refuse to fetch missing models
    # and fail instead. Like every other option here it is emitted only when
    # the caller actually passes it (the `opt_key in self._options` check
    # below) — that is load-bearing, not incidental: anyone pointing
    # bin_path at a binary older than installer.PINNED_VERSION has one that
    # predates the flag and exits 1 with a usage error the moment it appears
    # in argv.
    "no_download": "no-download",
}


class Engine:
    def __init__(self, **options: Any) -> None:
        bin_path = options.pop("bin_path", None)
        if bin_path is None:
            bin_path = str(default_bin_path())
        self._bin_command: list[str] = (
            list(bin_path) if isinstance(bin_path, (list, tuple)) else [bin_path]
        )
        self._options = options

    def recognize(self, image_path: str) -> PageResult:
        if len(self._bin_command) == 1 and not Path(self._bin_command[0]).is_file():
            raise OcrError(
                f"arboocr_demo binary not found at {self._bin_command[0]}. "
                "Run 'arbo-ocr-install' or pass bin_path explicitly."
            )

        argv = [*self._bin_command, "--image", image_path, "--json", *self._flags_from_options()]

        # Python's subprocess.run(capture_output=True) uses Popen.communicate()
        # internally, which reads stdout and stderr concurrently (via threads on
        # Windows, select() on POSIX) — unlike PHP's proc_open, there's no risk
        # of deadlocking on a filled pipe buffer here, so no stderr-to-tempfile
        # workaround is needed. encoding must be explicit: text=True alone
        # decodes with the locale default (cp1252 on Windows), and arboocr_demo
        # always emits UTF-8 - some recognized text hits a byte cp1252 has no
        # mapping for, raising UnicodeDecodeError inside subprocess's internal
        # background reader thread, silently swallowed there, surfacing here
        # only as stdout/stderr being None despite returncode 0.
        result = subprocess.run(argv, capture_output=True, text=True, encoding="utf-8")

        # Deliberately "!= 0", not "== 1": arboOCR v0.2.0 added exit code 2
        # for model-load / recognition failure, distinct from 1 (bad usage).
        # The exact code is preserved on OcrError.exit_code for callers that
        # want to tell the two apart. Note also that v0.2.0's arboocr_demo is
        # silent on stderr unless --log-level is passed, so stderr may be
        # empty even on a genuine failure — never treat stderr emptiness as
        # success, only returncode.
        if result.returncode != 0:
            raise OcrError(
                f"arboocr_demo exited with code {result.returncode}",
                exit_code=result.returncode,
                stderr=result.stderr,
            )

        return PageResult.from_json(result.stdout.strip())

    def recognize_batch(self, image_paths: list[str]) -> list[PageResult]:
        """OCR many images with **one** arboocr_demo process
        (`--images-from <list> --json`), returning one PageResult per input in
        input order.

        recognize() starts a fresh process per image, and that process start
        plus model load dominates a short page; this pays it once for the whole
        list instead.

        Results are matched to inputs **by position** because the binary
        reports only a basename. That is sound only while the counts agree, so
        a mismatch raises rather than returning a shifted list.

        A batch exits 1 when *any* image came back empty. That is an ordinary
        outcome, not a failure, and is tolerated as long as the JSON array is
        still on stdout — a usage error exits 1 too but leaves stdout empty,
        and that one raises.
        """
        if not image_paths:
            return []

        if len(self._bin_command) == 1 and not Path(self._bin_command[0]).is_file():
            raise OcrError(
                f"arboocr_demo binary not found at {self._bin_command[0]}. "
                "Run 'arbo-ocr-install' or pass bin_path explicitly."
            )

        # The list file is newline-delimited, and the binary skips blank lines
        # and '#' lines as comments. A path in either shape would be dropped
        # silently and shift every later result onto the wrong input, so it is
        # rejected up front rather than mis-attributed later.
        for i, path in enumerate(image_paths):
            if path == "":
                raise OcrError(f"recognize_batch: image_paths[{i}] is empty")
            if "\n" in path or "\r" in path:
                raise OcrError(
                    f"recognize_batch: image_paths[{i}] contains a newline, "
                    f"which the image list format cannot represent: {path!r}"
                )
            if path.lstrip(" \t").startswith("#"):
                raise OcrError(
                    f"recognize_batch: image_paths[{i}] starts with '#', which "
                    f"arboocr_demo reads as a comment and would skip: {path!r}"
                )

        # Same explicit encoding and same communicate()-based capture as
        # recognize(), for the same reasons — see the notes there.
        with tempfile.NamedTemporaryFile(
            "w", suffix=".txt", prefix="arbo-ocr-list-", delete=False, encoding="utf-8"
        ) as list_file:
            list_file.write("\n".join(image_paths) + "\n")
            list_path = list_file.name
        try:
            argv = [
                *self._bin_command,
                "--images-from", list_path, "--json",
                *self._flags_from_options(),
            ]
            result = subprocess.run(argv, capture_output=True, text=True, encoding="utf-8")
        finally:
            os.unlink(list_path)

        out = result.stdout.strip()
        if result.returncode != 0 and not (
            result.returncode == 1 and out.startswith("[")
        ):
            raise OcrError(
                f"arboocr_demo exited with code {result.returncode}",
                exit_code=result.returncode,
                stderr=result.stderr,
            )

        try:
            pages = json.loads(out)
        except json.JSONDecodeError as e:
            raise OcrError(
                f"arboocr_demo --images-from produced unparseable output: {out[:500]!r}"
            ) from e

        if not isinstance(pages, list):
            raise OcrError(
                f"arboocr_demo --images-from produced unparseable output: {out[:500]!r}"
            )

        # Count first: every later check is positional, so a short or long
        # array has to fail here rather than shift text onto the wrong file.
        if len(pages) != len(image_paths):
            raise OcrError(
                f"arboocr_demo returned {len(pages)} results for "
                f"{len(image_paths)} images; cannot match results to inputs by position"
            )

        for i, page in enumerate(pages):
            if not isinstance(page, dict) or not isinstance(page.get("lines"), list):
                raise OcrError(
                    f"arboocr_demo --images-from element {i} has no 'lines' array: "
                    f"{json.dumps(page)[:500]}"
                )

        return [PageResult.from_dict(page) for page in pages]

    def _flags_from_options(self) -> list[str]:
        argv: list[str] = []
        for mapping in (_STRING_FLAGS, _NUMERIC_FLAGS):
            for opt_key, cli_flag in mapping.items():
                if opt_key in self._options:
                    argv += [f"--{cli_flag}", str(self._options[opt_key])]
        for opt_key, cli_flag in _BOOL_FLAGS.items():
            if opt_key in self._options:
                # cxxopts only binds a bool flag's value via "=" — a bare
                # "--flag" followed by a separate "true"/"false" token leaves
                # the flag implicitly true and the value ignored (confirmed
                # against the real arboocr_demo binary).
                value = "true" if self._options[opt_key] else "false"
                argv.append(f"--{cli_flag}={value}")
        return argv
