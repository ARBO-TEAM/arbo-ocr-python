"""Runs the prebuilt arboocr_demo binary via subprocess and parses its
--json output. Requires no C++ build — only the binary the installer
downloaded (or one you point at manually via the `bin_path` option).
"""

from __future__ import annotations

import subprocess
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
    # Needs the arboOCR release that adds model auto-download — NOT the
    # currently pinned installer.PINNED_VERSION, which has no such flag and
    # exits 1 (usage error) if it is passed. Directory URL to fetch missing
    # models from, e.g. an internal mirror. See _BOOL_FLAGS' "no_download"
    # note for why omitting the option must stay a no-op.
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
    # Needs the arboOCR release that adds model auto-download; refuse to
    # fetch missing models and fail instead. Like every other option here it
    # is emitted only when the caller actually passes it (the `opt_key in
    # self._options` check below) — that is load-bearing, not incidental:
    # anyone still on installer.PINNED_VERSION has a binary that predates
    # the flag and exits 1 with a usage error the moment it appears in argv.
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
