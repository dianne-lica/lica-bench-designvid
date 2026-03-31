# lica-bench

**lica-bench** is a structured evaluation suite for measuring how well vision-language models understand, edit, and generate graphic design artifacts. It covers layout reasoning, typography, visual hierarchy, SVG/vector understanding, template variants, animation, and more.

Benchmarks use the [Lica dataset](https://github.com/purvanshi/lica-dataset) (1,148 graphic design layouts). The release zip is unpacked as **`lica-benchmarks-dataset/`** with two parts: **`lica-data/`** holds the **core Lica files** (`metadata.csv`, `layouts/`, `images/`, `annotations/`). **`benchmarks/<domain>/`** holds **task-specific evaluation data** (manifests, JSON specs, prepared assets).

## Benchmarks

Each task is one of two types: **understanding** (answer a question or edit an artifact), or **generation** (produce a new artifact).

### Taxonomy (five domains)

Counts below are **evaluation task lines** (how the benchmark suite is scoped). Typography understanding lists ten lines by counting the five fields in `typography-3` separately. Layout understanding lists eight lines; **four** of those do not yet have a shipped benchmark (layer order, rotation, crop shape, frame detection).

| Domain | Understanding | Generation | Total |
|--------|--------------:|-----------:|------:|
| Layout | 8 | 4 | 12 |
| Typography | 10 | 2 | 12 |
| SVG & Vector | 5 | 5 | 10 |
| Template & Semantics | 5 | 3 | 8 |
| Animation | 3 | 3 | 6 |
| **Total** | **31** | **17** | **48** |

**Roll-up to code and data:** Layout → `layout-1`–`layout-7` and `image-1`–`image-2` (layer-aware inpainting). Typography → `typography-1`–`typography-8`. SVG & Vector → `svg-1`–`svg-8` and `lottie-1`–`lottie-2`. Template & Semantics → `category-1`–`category-2` and `template-1`–`template-6`. Animation → `temporal-1`–`temporal-6`.

The module table below sums to **45** task lines: **48 − 45 = 3** because four layout-understanding rows are still to ship as benchmarks, while two inpainting benchmarks (`image-1`, `image-2`) map to **one** generation row on the grid (−1), net **+3** vs shipped lines.

### By package module (41 benchmarks)

The registry exposes **41** benchmark IDs across **eight** Python modules (mirrors `benchmarks/` folders):

| Module | Task lines | Benchmarks | Description |
|--------|----------:|----------:|-------------|
| category | 2 | 2 | Design category classification and user intent prediction |
| image | 2 | 2 | Layer-aware object insertion (reference-guided and description-guided) |
| layout | 7 | 7 | Spatial reasoning over design canvases (aspect ratio, element counting, component type and detection) and layout generation (intent-to-layout, partial completion, aspect-ratio adaptation) |
| lottie | 2 | 2 | Lottie animation generation from text and image |
| svg | 8 | 8 | SVG reasoning and editing (perceptual and semantic Q/A, bug fixing, optimization, style editing) and generation (text-to-SVG, image-to-SVG, combined input) |
| template | 6 | 6 | Template matching, retrieval, clustering, and generation (style completion, color transfer, asset swap) |
| temporal | 6 | 6 | Keyframe ordering, motion type classification, animation property extraction, and generation (animation parameters, motion trajectory, short-form video) |
| typography | 12 | 8 | Font family, color, five fields in `typography-3`\*, style ranges, curved text, rotation, and generation (styled text element, styled text rendering to layout) |

> \* `typography-3` (Text Params Estimation) expects one JSON object with five fields: `font_size`, `font_weight`, `text_align`, `letter_spacing`, and `line_height`.

## Getting started

### 1. Install

```bash
git clone https://github.com/purvanshi/lica-bench.git
cd lica-bench
python3 -m venv .venv && source .venv/bin/activate
pip install -e .

# Add extras you need (pick any combination)
pip install -e ".[metrics]"          # scipy, sklearn, Pillow, cairosvg, etc.
pip install -e ".[openai]"           # OpenAI provider
pip install -e ".[gemini]"           # Gemini provider
pip install -e ".[anthropic]"        # Anthropic provider
pip install -e ".[svg-metrics]"      # Full SVG eval (metrics + LPIPS, CLIP)
pip install -e ".[layout-metrics]"   # Layout/image metrics (Linux + Python<3.12 recommended)
pip install -e ".[dev]"              # ruff linter
```

The PyPI/setuptools distribution is **lica-bench**; import the library as **`design_benchmarks`**.

### 2. Verify installation (no data, no API keys)

```bash
python scripts/run_benchmarks.py --list                     # enumerate tasks and readiness
```

### 3. Download data

```bash
python scripts/download_data.py                              # → data/lica-benchmarks-dataset/
```

Point `--data` at the domain folder under **`benchmarks/`** you need. `--dataset-root` always points at the **`lica-benchmarks-dataset/`** root.

### 4. Run benchmarks

```bash
# Stub model (no API keys; validates load_data + build_model_input on real data)
python scripts/run_benchmarks.py --stub-model --benchmarks category-1 \
    --data data/lica-benchmarks-dataset/benchmarks/category/CategoryClassification \
    --dataset-root data/lica-benchmarks-dataset --n 5

# Real model
python scripts/run_benchmarks.py --benchmarks svg-1 \
    --provider openai --model-id gpt-5.4 \
    --data data/lica-benchmarks-dataset/benchmarks/svg \
    --dataset-root data/lica-benchmarks-dataset

# Temporal benchmarks (video-based)
python scripts/run_benchmarks.py --benchmarks temporal-1 \
    --provider gemini \
    --data data/lica-benchmarks-dataset/benchmarks/temporal/KeyframeOrdering \
    --dataset-root data/lica-benchmarks-dataset
```

Use the same **`--dataset-root`** (Lica bundle root) for stub runs, API runs, and **`--batch-submit`** so paths inside CSVs/JSON resolve correctly.

See [scripts/README.md](scripts/README.md) for batch submit/collect, vLLM, HuggingFace, multi-model comparison, config files, and all CLI flags.

### 5. API keys

Set whichever provider(s) you need:

```bash
export OPENAI_API_KEY=sk-...
export ANTHROPIC_API_KEY=sk-ant-...
export GOOGLE_API_KEY=...            # Gemini (Google AI Studio / google-genai API key)
```

For **Gemini on Vertex AI** (service account), pass a JSON key file instead of relying on `GOOGLE_API_KEY`:

```bash
python scripts/run_benchmarks.py --benchmarks svg-1 --provider gemini \
    --credentials /path/to/service-account.json \
    --data ... --dataset-root ...
```

The file must be either a **service account** key (`type: service_account`) or JSON containing an `api_key` field.

**Batch submit** for Gemini also needs a GCS bucket (`--bucket` or `DESIGN_BENCHMARKS_GCS_BUCKET`); see [scripts/README.md](scripts/README.md).

## Benchmark dataset layout

Everything lives under one root directory **`lica-benchmarks-dataset/`** (e.g. `data/lica-benchmarks-dataset/` after `download_data.py`):

```
lica-benchmarks-dataset/
├── lica-data/                    # core Lica release (layouts, renders, metadata)
│   ├── metadata.csv              # one row per layout
│   ├── layouts/<template_id>/<layout_id>.json
│   ├── images/<template_id>/<layout_id>.{png,jpg,webp,mp4}
│   └── annotations/…             # optional
│
└── benchmarks/                   # evaluation inputs per domain
    ├── category/                 #   CategoryClassification/, UserIntentPrediction/
    ├── image/
    ├── layout/
    ├── lottie/
    ├── svg/
    ├── template/
    ├── temporal/                 #   KeyframeOrdering/, MotionTypeClassification/, etc.
    └── typography/
```

**Using this bundle:** Point **`--dataset-root`** at this directory (the parent of `lica-data/` and `benchmarks/`). Point **`--data`** at the inputs for the benchmark you run—typically a folder under **`benchmarks/<domain>/`** (sometimes a task subfolder; mirror the examples in [Getting started](#4-run-benchmarks) and `scripts/run_benchmarks.py --list`). Paths inside CSVs (for example `image_path`) and in template JSON (`data_root`) are interpreted relative to **`--dataset-root`**, not only relative to `--data`.

**What the two trees are:** **`lica-data/`** is the shared Lica corpus (layout JSON, renders, `metadata.csv`). **`benchmarks/`** holds evaluation payloads per domain (CSVs, JSON, manifests, copied assets). Exact filenames differ by task; see the module under `src/design_benchmarks/tasks/<domain>.py` or [docs/CONTRIBUTING.md](docs/CONTRIBUTING.md) when adding or packaging data.

## Project structure

```
lica-bench/
├── src/design_benchmarks/
│   ├── tasks/              # @benchmark classes — one file per domain
│   │   ├── category.py     #   category-1, category-2
│   │   ├── image.py        #   image-1, image-2
│   │   ├── layout.py       #   layout-1 … layout-7
│   │   ├── lottie.py       #   lottie-1, lottie-2
│   │   ├── svg.py          #   svg-1 … svg-8
│   │   ├── template.py     #   template-1 … template-6
│   │   ├── temporal.py     #   temporal-1 … temporal-6
│   │   └── typography.py   #   typography-1 … typography-8
│   ├── models/             # Provider wrappers (OpenAI, Anthropic, Gemini, HF, vLLM)
│   ├── metrics/            # Reusable metric functions (IoU, FID, SSIM, LPIPS, edit distance)
│   ├── evaluation/
│   │   ├── tracker.py      # Per-sample JSONL logger
│   │   └── reporting.py    # BenchmarkResult / RunReport (CSV + JSON)
│   ├── inference/          # Batch API runners, GCS helpers
│   ├── utils/              # Shared helpers (image, text, layout path resolution)
│   ├── base.py             # BaseBenchmark, BenchmarkMeta, TaskType, @benchmark
│   ├── registry.py         # Auto-discovery via pkgutil.walk_packages
│   └── runner.py           # BenchmarkRunner orchestration
├── scripts/
│   ├── download_data.py    # Fetch + unpack into lica-benchmarks-dataset/
│   └── run_benchmarks.py   # Unified CLI for list, stub, real, and batch runs
├── docs/
│   └── CONTRIBUTING.md     # How to add tasks and domains
└── pyproject.toml
```

## Quick start (Python API)

```python
from pathlib import Path
from design_benchmarks import BenchmarkRegistry, BenchmarkRunner
from design_benchmarks.models import load_model

registry = BenchmarkRegistry()
registry.discover()

runner = BenchmarkRunner(registry)
models = {"openai": load_model("openai", model_id="gpt-5.4")}
report = runner.run(
    benchmark_ids=["svg-1"],
    models=models,
    data_dir=Path("data/lica-benchmarks-dataset/benchmarks/svg"),
    dataset_root=Path("data/lica-benchmarks-dataset"),
    n=5,
)
print(report.summary())
report.save("outputs/report.json")
runner.tracker.save("outputs/tracker.jsonl")
```

`RunReport` includes both metric scores and reliability counters per benchmark/model:
`count`, `success_count`, `failure_count`, and `failure_rate`. This makes partial-run
failures visible in terminal summaries and saved JSON/CSV reports.

## Contributing

See **[docs/CONTRIBUTING.md](docs/CONTRIBUTING.md)** for:

- How to add a benchmark task to an existing domain
- How to create a new domain module
- Where benchmark inputs live in the Lica release and the PR checklist

## Limitations

- Some metrics (LPIPS, CLIP score, SSIM, CIEDE2000) need heavier extras (`.[svg-metrics]`, `.[layout-metrics]`). The full `.[layout-metrics]` stack is enabled on Linux with Python < 3.12. Tasks return `0.0` for metrics whose dependencies are unavailable.
- **`--provider`** picks which backend runs the model (OpenAI, Gemini, Anthropic, etc.); **`--model-id`** is only the catalog string for *that* backend (it does not select the provider). If you omit **`--model-id`**, the default for the chosen provider is used (see `DEFAULT_MODEL_IDS` in `scripts/run_benchmarks.py`). With **`--multi-models`**, each entry is **`provider:model_id`** so both are explicit. Use a **`--model-id`** your account actually exposes (README examples may name newer IDs such as `gpt-5.4`).

## Models

| Provider | Install extra | CLI flag |
|----------|--------------|----------|
| OpenAI | `.[openai]` | `--provider openai` |
| Anthropic | `.[anthropic]` | `--provider anthropic` |
| Gemini | `.[gemini]` | `--provider gemini` |
| HuggingFace | (torch) | `--provider hf --device auto` |
| vLLM | `.[vllm]` | `--provider vllm` |
| Diffusion | `.[vllm-omni]` | `--provider diffusion` |
| OpenAI Image | `.[openai]` | `--provider openai_image` |

### Evaluation extras

| Extra | Contents | Used by |
|-------|----------|---------|
| `.[metrics]` | scipy, sklearn, scikit-image, Pillow, cairosvg | All implemented tasks (clustering, color, SVG rendering) |
| `.[svg-metrics]` | metrics + torch, transformers, lpips | SVG generation (LPIPS, CLIP score) |
| `.[layout-metrics]` | torch, transformers (+ Linux/Python<3.12: pyiqa, hpsv2, hpsv3, dreamsim, image-reward) | Layout / image generation (FID, HPSv2/v3, DreamSim) |

## Dataset

The [Lica dataset](https://github.com/purvanshi/lica-dataset) underpins the initial benchmark release:

- 1,148 graphic design layouts across 9 design categories
- Structured JSON annotations (components, positions, styles, descriptions)
- Rendered images (PNG) and animations (MP4)
- Download: `python scripts/download_data.py`

## Citation

If you use this benchmark, please cite the original LICA dataset:

```bibtex
@misc{lica-dataset,
  author = {Mehta, Purvanshi and others},
  title  = {LICA: Open-Source Graphic Design Layout Dataset},
  year   = {2025},
  url    = {https://github.com/purvanshi/lica-dataset}
}
```
