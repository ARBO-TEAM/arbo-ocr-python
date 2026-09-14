"""Stand-in for arboocr_demo, used only by test_engine.py. Reads its own
argv to decide what to emit, so tests can exercise both the happy path and
error paths without a real binary or models."""

import json
import sys

args = sys.argv[1:]

if "--fail" in args:
    print("simulated engine failure", file=sys.stderr)
    sys.exit(2)

if "--garbage" in args:
    print("not json")
    sys.exit(0)

if "--noisy-stderr" in args:
    # Past a pipe's OS buffer (~64KB), written before any stdout — the real
    # arboocr_demo does this via ONNXRuntime schema-registration warnings.
    sys.stderr.write("noise\n" * 20000)

if "--nonascii" in args:
    # "Ё" (U+0401) is 0xD1 0x81 in UTF-8; byte 0x81 has no mapping in
    # cp1252, the Windows locale default subprocess.run(text=True) falls
    # back to when no encoding is given explicitly.
    text_field = "Ёlka — café"
else:
    text_field = "hello"

image = ""
if "--image" in args:
    idx = args.index("--image")
    if idx + 1 < len(args):
        image = args[idx + 1]

line = {"text": text_field, "score": 0.9, "detScore": 0.8,
        "polygon": [{"x": 1.0, "y": 2.0}]}

# arboOCR v0.2.0: "words" is present in the JSON only when --word-boxes was
# asked for, and Engine emits bool flags in the single-token "--flag=value"
# form, so match on that exact token rather than a bare "--word-boxes".
if "--word-boxes=true" in args:
    line["words"] = [
        {"text": "hel", "score": 0.95, "polygon": [{"x": 1.0, "y": 2.0}]},
        {"text": "lo", "score": 0.85, "polygon": [{"x": 3.0, "y": 4.0}]},
    ]

import os

# --images-from is batch mode: one process over a newline-delimited list file,
# one JSON array on stdout in list order. Each path is echoed back as that
# page's line text, so tests can assert positional matching rather than
# assume it.
if "--images-from" in args:
    if "--batch-usage-error" in args:
        # What a bad flag actually does: exit 1 with no JSON on stdout, which
        # must not be confused with the ordinary "a page came back empty"
        # exit 1 that still carries the array.
        print("Option '--images-from' does not exist", file=sys.stderr)
        sys.exit(1)

    with open(args[args.index("--images-from") + 1], encoding="utf-8") as fh:
        paths = [
            l.strip() for l in fh
            if l.strip() and not l.strip().startswith("#")
        ]
    if "--batch-short" in args and paths:
        paths.pop()

    pages = [
        {"backend": "cpu", "image": os.path.basename(p), "elapsedMs": 12.5,
         "lines": [{"text": p, "score": 0.9, "detScore": 0.8,
                    "polygon": [{"x": 1.0, "y": 2.0}]}]}
        for p in paths
    ]
    sys.stdout.buffer.write((json.dumps(pages) + "\n").encode("utf-8"))
    # A batch exits 1 when any image came back empty — an ordinary outcome
    # that still carries the JSON the caller asked for.
    sys.exit(1 if "--batch-exit1" in args else 0)

payload = json.dumps({
    "backend": "cpu",
    "image": os.path.basename(image),
    "elapsedMs": 12.5,
    "lines": [line],
}, ensure_ascii=False)
# Write raw UTF-8 bytes directly, bypassing sys.stdout's text-mode encoding
# (cp1252 on a non-UTF-8 Windows console) - otherwise this script itself
# would crash trying to *write* text_field, before the parent's decode
# (the actual thing under test) ever comes into play.
sys.stdout.buffer.write((payload + "\n").encode("utf-8"))
