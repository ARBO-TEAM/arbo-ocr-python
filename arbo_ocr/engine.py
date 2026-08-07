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
}

_BOOL_FLAGS = {
    "use_angle_cls": "angle",
    "use_cuda": "cuda",
    "use_tensorrt": "tensorrt",
    "use_fp16": "fp16",
    "use_clahe": "clahe",
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

        if result.returncode != 0:
            raise OcrError(
                f"arboocr_demo exited with code {result.returncode}",
                exit_code=result.returncode,
                stderr=result.stderr,
            )

        return PageResult.from_json(result.stdout.strip())

    def _flags_from_options(self) -> list[str]:
        argv: list[str] = []
        for opt_key, cli_flag in _STRING_FLAGS.items():
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
