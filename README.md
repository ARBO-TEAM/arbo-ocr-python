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

You also need the OCR models — arboOCR does not bundle them, and the
currently pinned `v0.2.0` binary cannot fetch them either, so you must put
them somewhere and point `models_dir` at them. See [Models](#models) below
for exactly which files each `model_type` needs and where to get them.

The **next arboOCR release** adds model auto-download: missing models get
fetched into a local cache and SHA-256-verified on first use. This package
already exposes the matching options (see
[Model auto-download](#model-auto-download)), but they only do anything once
you're running a binary from that release — `v0.2.0` rejects the underlying
flags outright. This README will lose the "next release" hedging when the
pinned version moves.

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

**Getting the files** — with the pinned `v0.2.0` binary you have to supply
them yourself (it hosts no default download URLs — see arboOCR's own
[Models section](https://github.com/wafik/ArboOCR#models)), so pick
whichever applies:
- Already have a Python `rapidocr` install? Copy its `models/` directory
  over, renaming files to match the layout above.
- Have your own PP-OCRv6 ONNX export? Place/rename the files as above.
- A local arboOCR checkout's `models/` directory already has the detector,
  classifier, and all three recognizer sizes — handy for local dev (see the
  tiny-model example below).
- From the **next arboOCR release** onward you can skip all of the above and
  let the binary fetch what's missing — see
  [Model auto-download](#model-auto-download).

### Model auto-download

> **Requires the next arboOCR release.** The pinned `v0.2.0` binary this
> package installs today has no auto-download and does not accept
> `--no-download`, `--models-url`, or `--download-models` — passing any of
> them makes it exit `1` with a usage error. Everything in this section
> applies once you point `bin_path` at a newer build, or once this package's
> pin moves to the release that ships it.

That release resolves each model file in a fixed order, per file:

1. An explicit path (`det_model_path` / `cls_model_path` / `rec_model_path` /
   `dict_path`) is used as given and is **never** substituted by a download.
2. Otherwise a file already present in `models_dir` wins — zero network.
3. Only then is it downloaded and SHA-256-verified into the model cache.

So an existing offline setup keeps behaving exactly as it does now; the
download only fills genuine gaps.

**Options**

```python
# Fail loudly instead of reaching for the network — good for air-gapped
# or reproducible-build environments.
engine = Engine(models_dir="/path/to/models", no_download=True)

# Fetch anything missing from an internal mirror instead of the default host.
engine = Engine(models_url="https://mirror.internal/arboocr/models/")
```

**Prefetching** — pull the models down ahead of time (a Docker build layer,
a CI warm-up step) so the first `recognize()` call does no network I/O:

```python
from arbo_ocr.installer import download_models

download_models(ocr_version="PP-OCRv6", model_type="small")
```

It shells out to `arboocr_demo --download-models`, which downloads, verifies,
and exits without running OCR; it raises `RuntimeError` on a non-zero exit.
Every argument is optional (`ocr_version`, `model_type`, `models_url`,
`bin_path`) and unset ones emit no flag, leaving arboOCR's defaults in charge.

**Environment variables** — read by the `arboocr_demo` child process, which
inherits the parent Python process's environment:

| Variable | Effect |
|---|---|
| `ARBOOCR_OFFLINE=1` | never download; same as `no_download=True` |
| `ARBOOCR_CACHE_DIR` | override the model cache directory |
| `ARBOOCR_MODELS_URL` | default directory URL to fetch from; same as `models_url` |

**Cache directory** — tag-scoped, so a future model-set revision lands beside
the current one instead of overwriting it:

| OS | Path |
|---|---|
| Windows | `%LOCALAPPDATA%\arboOCR\models\models-v1` |
| macOS | `~/Library/Caches/arboOCR/models/models-v1` |
| Linux | `$XDG_CACHE_HOME/arboOCR/models/models-v1`, else `~/.cache/arboOCR/models/models-v1` |

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
| `no_download` | bool | `--no-download` | *next arboOCR release* — never fetch missing models; fail instead |
| `models_url` | str | `--models-url` | *next arboOCR release* — directory URL to fetch missing models from (e.g. an internal mirror) |

An option you don't pass emits no flag at all, so the two *next arboOCR
release* rows above are inert — and safe — on the pinned `v0.2.0` binary.
Passing one to `v0.2.0` is not: it exits `1` with a usage error. See
[Model auto-download](#model-auto-download).

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
