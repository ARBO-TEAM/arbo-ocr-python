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

image = ""
if "--image" in args:
    idx = args.index("--image")
    if idx + 1 < len(args):
        image = args[idx + 1]

import os
print(json.dumps({
    "backend": "cpu",
    "image": os.path.basename(image),
    "elapsedMs": 12.5,
    "lines": [
        {"text": "hello", "score": 0.9, "detScore": 0.8,
         "polygon": [{"x": 1.0, "y": 2.0}]},
    ],
}))
