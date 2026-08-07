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

As of [`v0.2.0`](https://github.com/wafik/ArboOCR/releases/tag/v0.2.0),
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

### Options

Every `Engine(**options)` keyword maps to one `arboocr_demo` CLI flag.
Anything not listed is ignored rather than passed through.

| Option | Type | CLI flag | Notes |
|---|---|---|---|
| `models_dir` | str | `--models-dir` | folder of PP-OCRv6 ONNX files |
| `ocr_version` | str | `--ocr-version` | default `PP-OCRv6` |
| `model_type` | str | `--model-type` | `tiny` / `small` (default) / `medium` |
| `det_model_path` | str | `--det-model` | override the detector file |
| `cls_model_path` | str | `--cls-model` | override the classifier file |
| `rec_model_path` | str | `--rec-model` | override the recognizer file |
| `dict_path` | str | `--dict` | override the recognizer dictionary |
| `use_angle_cls` | bool | `--angle` | 0°/180° orientation classification |
| `use_cuda` | bool | `--cuda` | |
| `use_tensorrt` | bool | `--tensorrt` | |
| `use_fp16` | bool | `--fp16` | |
| `use_clahe` | bool | `--clahe` | contrast pre-processing |
| `min_confidence` | float | `--min-confidence` | *v0.2.0* — drop lines below this score; `0` disables (default 0.5) |
| `rec_batch_num` | int | `--rec-batch-num` | *v0.2.0* — crops per recognition inference call (default 6) |
| `det_limit_side_len` | int | `--det-limit-side-len` | *v0.2.0* — longest image side for detection resize (default 960) |
| `word_boxes` | bool | `--word-boxes` | *v0.2.0* — also populate `line.words` |
| `log_level` | str | `--log-level` | *v0.2.0* — `debug` / `info` / `warn` / `error`; the binary is silent on stderr without it |

#### Word boxes

With `word_boxes=True`, each line carries a `words` list — one entry per
word (per character for CJK), each a plain dict with `text`, `score`, and
`polygon`. Without it arboOCR omits the key entirely and `line.words` is `[]`:

```python
engine = Engine(models_dir="/path/to/models", word_boxes=True)
result = engine.recognize("/path/to/receipt.jpg")

for line in result.lines:
    for word in line.words:
        print(word["text"], word["score"], word["polygon"])
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

`arbo-ocr-python` was compared against arbo-ocr-php, arbo-ocr-go, and
arbo-ocr-rust on a 40-image SROIE sample — all four call the identical
`arboocr_demo` binary, so accuracy is the same across all four; this
measures wrapper overhead only (subprocess spawn − arboocr_demo's own
reported time):

| Size | arbo-php | arbo-go | arbo-rust | arbo-python |
|--------|----------:|---------:|-----------:|-------------:|
| tiny | 192 ms | 138 ms | 131 ms | 203 ms |
| small | 234 ms | 184 ms | 174 ms | 249 ms |
| medium | 302 ms | 253 ms | 245 ms | 321 ms |

Same accuracy across all four (83.7/85.3/85.5% tiny/small/medium). Go and
Rust track each other closely (both compiled, no interpreter startup);
PHP and Python both pay their interpreter's own startup cost on top of the
same `arboocr_demo` spawn, landing in the same ballpark as each other
(Python's is slightly higher — its `import arbo_ocr` pulls in a few more
stdlib modules than PHP's autoloader touches per call).

Two real issues surfaced by this benchmark, both fixed:
- **UTF-8 decode crash**: `subprocess.run(text=True)` with no explicit
  `encoding` decodes with the Windows locale default (`cp1252`), not UTF-8.
  Some recognized text hits a byte `cp1252` has no mapping for, raising
  `UnicodeDecodeError` inside `subprocess`'s internal reader thread —
  silently swallowed there, surfacing only as `stdout`/`stderr` being `None`
  despite `returncode == 0`. Fixed by passing `encoding="utf-8"` explicitly
  in `Engine.recognize()`.
- **Avoidable import cost**: `installer.py` imported `urllib.request`,
  `zipfile`, and `tarfile` at module level, even though `engine.py` only
  ever needs `detect_platform()`/`default_bin_path()` from that module.
  Since those are heavy stdlib imports (`urllib.request` alone pulls in
  `ssl`, `http.client`, etc.), every `Engine` call paid ~50ms of interpreter
  startup it never used. Moved them into the functions that actually need
  them (`download_and_extract()`, `main()`) — cut measured overhead by
  ~35-40ms per call across all three sizes.

## Development

```bash
pip install -e ".[dev]"
pytest
```

## License

Apache-2.0
