# arbo-ocr-python

Python wrapper for [arboOCR](https://github.com/wafik/ArboOCR) — runs the
prebuilt `arboocr_demo` binary via `subprocess`, no C++ build required.

## Install

```bash
pip install arbo-ocr-python
arbo-ocr-install
```

Unlike Composer, a `pip install` cannot reliably run arbitrary code after
installing a wheel, so this package ships an explicit console script,
`arbo-ocr-install`, that you run once after installing (the same pattern
`playwright install` uses). It downloads the matching arboOCR release
binary (Windows or Linux, auto-detected) into `arbo_ocr/bin/<platform>/`.

As of [`v0.1.0-php1`](https://github.com/wafik/ArboOCR/releases/tag/v0.1.0-php1),
this is the pinned release. If auto-download fails (offline install,
unsupported OS), download a release manually from the
[arboOCR releases page](https://github.com/wafik/ArboOCR/releases) and pass
`bin_path` explicitly (see below).

You also need the OCR models — arboOCR does not bundle them. See
[Models](#models) below for exactly which files each `model_type` needs and
where to get them.

## Models

arboOCR doesn't bundle OCR models — you point `models_dir` at a folder of
PP-OCRv6 ONNX files. Only the recognizer has size variants; the detector is
always one file regardless of `model_type`:

| File | Needed for | Varies by `model_type`? |
|---|---|---|
| `PP-OCRv6_det.onnx` | detection | no — always this one file |
| `PP-OCRv6_rec_tiny.onnx` + `PP-OCRv6_rec_tiny_dict.txt` | `model_type: 'tiny'` | yes |
| `PP-OCRv6_rec_small.onnx` + `PP-OCRv6_rec_small_dict.txt` | `model_type: 'small'` (default) | yes |
| `PP-OCRv6_rec_medium.onnx` + `PP-OCRv6_rec_medium_dict.txt` | `model_type: 'medium'` | yes |
| `PP-OCRv6_cls.onnx` | angle classification, only if `use_angle_cls` | no |

You only need the recognizer size(s) you'll actually use — e.g. for
`model_type: 'small'` alone, `models_dir` just needs `PP-OCRv6_det.onnx` +
`PP-OCRv6_rec_small.onnx` + `PP-OCRv6_rec_small_dict.txt`. Switching sizes
later is just changing `model_type`; `models_dir` can hold all three sizes
side by side if you want to switch freely.

**Getting the files** — arboOCR doesn't host default download URLs (see its
own [Models section](https://github.com/wafik/ArboOCR#models)), so pick
whichever applies:
- Already have a Python `rapidocr` install? Copy its `models/` directory
  over, renaming files to match the layout above.
- Have your own PP-OCRv6 ONNX export? Place/rename the files as above.
- A local arboOCR checkout's `models/` directory already has the detector,
  classifier, and all three recognizer sizes — handy for local dev (see the
  tiny-model example below).

## Usage

```python
from arbo_ocr import Engine

engine = Engine(models_dir="/path/to/models", model_type="small")
result = engine.recognize("/path/to/image.png")

for line in result.lines:
    print(line.text, line.score)
```

Pass `bin_path` to point at a manually-downloaded binary instead of relying
on the auto-installed one:

```python
engine = Engine(bin_path="/custom/path/to/arboocr_demo", models_dir="/path/to/models")
```

## Quick example (tiny model, fastest)

For a fast local smoke test, use `model_type="tiny"` — the smallest/fastest
PP-OCRv6 recognizer. If you have an arboOCR checkout handy, its `models/`
folder already contains the tiny det/rec/cls ONNX files (no extra download):

```python
from arbo_ocr import Engine

engine = Engine(
    models_dir="/path/to/arboOCR/models",  # e.g. a local arboOCR checkout's models/ dir
    model_type="tiny",
)

result = engine.recognize("/path/to/receipt.jpg")

print(f"backend={result.backend} lines={len(result.lines)} elapsed_ms={result.elapsed_ms:.1f}")
for line in result.lines:
    print(f"  {line.text:<40} score={line.score:.3f}")
```

The `tiny` model trades some accuracy for speed — good for quick local
testing; switch to `small` (the default) or `medium` for production-quality
recognition.

## How it works

This package never builds or vendors arboOCR's C++ source. It downloads a
prebuilt, self-contained binary (binary + required shared libraries, no
source, no ONNX models) from arboOCR's GitHub Releases via the
`arbo-ocr-install` console script — run once after `pip install`, unlike
Composer's automatic post-install hook — and calls it as a subprocess per
image with a `--json` flag, parsing the JSON result.

## Benchmark

Wrapper-overhead benchmarks (subprocess spawn cost on top of
`arboocr_demo`'s own reported time) haven't been run for this package yet.
Accuracy is identical to the other language wrappers (PHP, Go, Rust, …),
since all of them call the same underlying `arboocr_demo` binary — only the
per-call wrapper overhead could differ, and that hasn't been measured here.

## Development

```bash
pip install -e ".[dev]"
pytest
```

## License

Apache-2.0
