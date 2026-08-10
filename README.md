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

Run it again after every `pip install --upgrade` too. The binary is fetched
at runtime, so it is not part of the wheel and pip leaves the old one in
place on an upgrade; `arbo-ocr-install` notices that it predates the newly
pinned release (it records the installed tag in
`arbo_ocr/bin/<platform>/.arboocr-version`) and replaces it. Re-running when
nothing changed is a no-op that downloads nothing.

[`v0.3.0`](https://github.com/wafik/ArboOCR/releases/tag/v0.3.0) is the
pinned release. If auto-download fails (offline install, unsupported OS),
download a release manually from the
[arboOCR releases page](https://github.com/wafik/ArboOCR/releases) and pass
`bin_path` explicitly (see below).

That is the whole setup — the OCR models are not a separate step. arboOCR
doesn't bundle them, but as of `v0.3.0` it fetches whatever is missing on
the first `recognize()` call, SHA-256-verifies it, and caches it locally so
later runs are offline. Pointing `models_dir` at a folder you populated
yourself still wins over any download and means zero network; it's now a
deliberate choice rather than a prerequisite. See [Models](#models) for the
file layout and [Model auto-download](#model-auto-download) for the cache,
the env vars, and how to turn the download off.

## Models

arboOCR doesn't bundle OCR models. It fetches the ones it needs on first
use (see [Model auto-download](#model-auto-download)), or you point
`models_dir` at a folder of PP-OCRv6 ONNX files you supply yourself. Either
way the layout is the same — only the recognizer has size variants, and the
detector is always one file regardless of `model_type`:

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

**Getting the files** — do nothing and let the binary fetch them (see
[Model auto-download](#model-auto-download)), or supply them yourself if you
would rather the first run touch no network at all:
- Already have a Python `rapidocr` install? Copy its `models/` directory
  over, renaming files to match the layout above.
- Have your own PP-OCRv6 ONNX export? Place/rename the files as above.
- A local arboOCR checkout's `models/` directory already has the detector,
  classifier, and all three recognizer sizes — handy for local dev (see the
  tiny-model example below).

### Model auto-download

Since arboOCR `v0.3.0` — the version this package pins — missing model files
are downloaded and SHA-256-verified automatically on first use, then cached.
No setup step, and nothing to do on a second run.

arboOCR resolves each model file in a fixed order, per file:

1. An explicit path (`det_model_path` / `cls_model_path` / `rec_model_path` /
   `dict_path`) is used as given and is **never** substituted by a download.
2. Otherwise a file already present in `models_dir` wins — zero network.
3. Only then is it downloaded and SHA-256-verified into the model cache.

So a populated `models_dir` still wins outright and means zero network; the
download only fills genuine gaps. The default source is
`https://github.com/ARBO-TEAM/arbo-ocr-models/releases/download/models-v1/`.

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

engine = Engine(model_type="small")
result = engine.recognize("/path/to/image.png")

for line in result.lines:
    print(line.text, line.score)
```

No `models_dir` needed — arboOCR downloads and caches what it's missing on
that first call. Pass `models_dir` when you want it to use files you already
have on disk instead:

```python
engine = Engine(models_dir="/path/to/models", model_type="small")
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
| `models_dir` | str | `--models-dir` | folder of PP-OCRv6 ONNX files; anything missing is downloaded |
| `ocr_version` | str | `--ocr-version` | default `PP-OCRv6` |
| `model_type` | str | `--model-type` | `tiny` / `small` (default) / `medium` |
| `det_model_path` | str | `--det-model` | override the detector file |
| `cls_model_path` | str | `--cls-model` | override the classifier file |
| `rec_model_path` | str | `--rec-model` | override the recognizer file |
| `dict_path` | str | `--dict` | override the recognizer dictionary |
| `use_angle_cls` | bool | `--angle` | 0°/180° orientation classification |
| `use_cuda` | bool | `--cuda` | CUDA execution provider — see [GPU](#gpu) |
| `use_tensorrt` | bool | `--tensorrt` | TensorRT execution provider — see [GPU](#gpu) |
| `use_fp16` | bool | `--fp16` | |
| `use_clahe` | bool | `--clahe` | contrast pre-processing |
| `min_confidence` | float | `--min-confidence` | *v0.2.0* — drop lines below this score; `0` disables (default 0.5) |
| `rec_batch_num` | int | `--rec-batch-num` | *v0.2.0* — crops per recognition inference call (default 6) |
| `det_limit_side_len` | int | `--det-limit-side-len` | *v0.2.0* — longest image side for detection resize (default 960) |
| `word_boxes` | bool | `--word-boxes` | *v0.2.0* — also populate `line.words` |
| `log_level` | str | `--log-level` | *v0.2.0* — `debug` / `info` / `warn` / `error`; the binary is silent on stderr without it |
| `no_download` | bool | `--no-download` | *v0.3.0* — never fetch missing models; fail instead |
| `models_url` | str | `--models-url` | *v0.3.0* — directory URL to fetch missing models from (e.g. an internal mirror) |

An option you don't pass emits no flag at all, which is what keeps the
`v0.3.0` rows above safe if you have pointed `bin_path` at an older binary
of your own. Passing one to a pre-`v0.3.0` build is not safe: it exits `1`
with a usage error. See [Model auto-download](#model-auto-download).

#### GPU

`use_cuda=True` / `use_tensorrt=True` require the matching ONNX Runtime
provider libraries alongside the binary. Release archives before `v0.3.0`
shipped without `onnxruntime_providers_shared`, so neither provider could
load from a release build at all — only from a locally compiled arboOCR.
`v0.3.0` includes it, so both flags now work with the binary
`arbo-ocr-install` fetches (given a suitable CUDA/TensorRT runtime on the
host).

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
PP-OCRv6 recognizer. The tiny model files are fetched and cached on the first
call, so there is nothing to set up:

```python
from arbo_ocr import Engine

engine = Engine(model_type="tiny")

# Or, to reuse ONNX files you already have — e.g. a local arboOCR checkout's
# models/ dir, which carries the tiny det/rec/cls files — and skip the fetch:
# engine = Engine(models_dir="/path/to/arboOCR/models", model_type="tiny")

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
image with a `--json` flag, parsing the JSON result. The ONNX models are not
in that archive and are not this package's job either: the binary fetches and
caches them itself on first use (see
[Model auto-download](#model-auto-download)).

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
