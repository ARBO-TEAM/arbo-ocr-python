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

import os
payload = json.dumps({
    "backend": "cpu",
    "image": os.path.basename(image),
    "elapsedMs": 12.5,
    "lines": [
        {"text": text_field, "score": 0.9, "detScore": 0.8,
         "polygon": [{"x": 1.0, "y": 2.0}]},
    ],
}, ensure_ascii=False)
# Write raw UTF-8 bytes directly, bypassing sys.stdout's text-mode encoding
# (cp1252 on a non-UTF-8 Windows console) - otherwise this script itself
# would crash trying to *write* text_field, before the parent's decode
# (the actual thing under test) ever comes into play.
sys.stdout.buffer.write((payload + "\n").encode("utf-8"))
