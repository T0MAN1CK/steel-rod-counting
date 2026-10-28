# Steel Rod Counting

Localize and count visible steel-rod ends in photographs using an ImageNet-pretrained ResNet-50 U-Net. The model predicts a continuous Gaussian centre heatmap; peak extraction turns that heatmap into rod-centre coordinates and a count.

The project includes data preparation, grouped train/validation splitting, paired augmentation, training, saved-checkpoint validation, whole-photo inference, unit checks and local visual reports. Training runs are configured with `argparse`. Metrics, checkpoints and prediction images stay on disk; no hosted monitoring account is required.

## Contents

- [Project layout](#project-layout)
- [Installation and VS Code](#installation-and-vs-code)
- [Dataset and preparation](#dataset-and-preparation)
- [Model, targets and loss](#model-targets-and-loss)
- [Train the preferred configuration](#train-the-preferred-configuration)
- [Monitor training](#monitor-training)
- [Evaluate a saved checkpoint](#evaluate-a-saved-checkpoint)
- [Run whole-photo inference](#run-whole-photo-inference)
- [Controlled experiments](#controlled-experiments)
- [Saved results](#saved-results)
- [Command-line settings](#command-line-settings)
- [Verification and troubleshooting](#verification-and-troubleshooting)

## Project layout

```text
steel-rod-counting/
  README.md
  pyproject.toml
  src/rod_counting/
    config.py             # CLI settings, validation and seeding
    dataset.py            # One sample: RGB, float targets, points, normalization
    dataloader.py         # Loaders, source-photo isolation, optional scene validation
    augmentation.py       # Paired geometry and RGB appearance changes
    heatmaps.py           # Gaussian centre target generation
    model.py              # U-Net construction and saved-checkpoint loading
    losses.py             # Weighted heatmap MSE
    train.py              # Train one epoch, validate, save and stop
    validation.py         # Validation pass and preview collection
    metrics.py            # Peak extraction, point matching and count metrics
    reporting.py          # Local CSV/JSONL logs, PNGs, HTML and run settings
    test_loader.py        # No-update CPU/CUDA preflight
    test.py               # Evaluate saved weights on validation
    inference.py          # Count rods in complete labeled photos
  scripts/
    data_preparation/     # Clean patches, split, reconstruct scenes, make targets
    compare_runs.py       # Compare saved run metrics and settings
    audit_dataset.py     # Optional audit of the complete original archive
  tests/                  # Unit checks for data, geometry, loss, metrics and controls
  data/                   # Local only
    raw/segmentation_dataset/{train_data,test_data}/
    prepared/{training_v1,larger_scenes_v1}/
  runs/                   # Local only: settings, logs, previews and best checkpoints
    index.html            # Browse the archived experiments and evaluations
  artifacts/              # Local only: saved validation and inference results
```

The Git repository contains source, tests, package settings and this README. Datasets, weights, environments and generated reports are ignored by Git. A local project archive can include these ignored directories, but cloning the repository alone does not download them. The supplied dataset is not redistributed; its source and usage rights need to be established before any redistribution.

## Installation and VS Code

Use Python **3.12 or 3.13**. The verified GPU environment uses Python 3.13, PyTorch 2.11.0 with CUDA 12.8, torchvision 0.26.0 and an NVIDIA RTX 4060 Ti with 8 GB VRAM. Direct dependency versions are pinned in `pyproject.toml`.

Open this project folder in VS Code and open **Terminal > New Terminal**. From the project root, create an environment:

For a source-only checkout, first run `git clone https://github.com/T0MAN1CK/steel-rod-counting.git` and `cd steel-rod-counting`. Supply data separately as described below. If using the complete local archive, open that folder directly.

```powershell
py -3.13 -m venv .venv
$python = Join-Path $PWD ".venv\Scripts\python.exe"

& $python -m pip install --upgrade pip
& $python -m pip install torch==2.11.0 torchvision==0.26.0 --index-url https://download.pytorch.org/whl/cu128
& $python -m pip install -e .
& $python -m pip check
& $python -c "import torch; print(torch.__version__); print('CUDA:', torch.cuda.is_available()); print(torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'CPU')"
```

If a suitable environment already exists, set `$python` to that environment's `Scripts\python.exe` and run `& $python -m pip install -e .` here. An editable installation points at the project folder where it was installed. Reinstall it after moving the project.

In VS Code, use **Python: Select Interpreter** to select this interpreter. The explicit `$python` commands do not require activating the environment. Each new PowerShell terminal needs its own `$python` assignment.

The pretrained encoder is downloaded on first use unless already cached. Optional `HF_HOME` and `TORCH_HOME` environment variables select cache directories. Cached pretrained initialization is unnecessary for loading an existing project checkpoint.

## Dataset and preparation

The raw inputs are JPG photographs/patches and JSON point annotations. Each JSON record has a `filename` and a `regions` mapping; each region contains `shape_attributes.cx` and `shape_attributes.cy`. These coordinates mark rod centres, not object outlines. Preserve raw annotations.

```text
data/raw/segmentation_dataset/
  train_data/               # 4,573 original 256 x 256 patches + train_labels.json
  test_data/                # 19 complete 4,160 x 3,120 photos + test_labels.json

data/prepared/training_v1/
  train_data/               # 3,394 retained patches; one shared image directory
  splits/
    train_labels.json
    validation_labels.json
    split_summary.json
  heatmaps/                 # Sigma 2 targets
  heatmaps_sigma3/          # Sigma 3 targets used by the preferred configuration
```

The local archive already contains prepared data. Skip these commands when those output directories exist. To prepare a fresh copy from raw inputs, run:

```powershell
& $python scripts/data_preparation/remove_black_patches.py --source data/raw/segmentation_dataset/train_data --output data/prepared/training_v1/train_data

& $python scripts/data_preparation/split_train_validation.py --train-dir data/prepared/training_v1/train_data --output data/prepared/training_v1/splits

& $python scripts/data_preparation/create_heatmaps.py --train-dir data/prepared/training_v1/train_data --splits data/prepared/training_v1/splits --output data/prepared/training_v1/heatmaps_sigma3 --sigma 3
```

Each preparation output directory must be new. To generate sigma 2 targets as well, repeat the last command with `--output data/prepared/training_v1/heatmaps --sigma 2`.

Preparation excludes a patch only if its suffix is 12–15, its RGB pixels are entirely black and it has no annotations. It removes 1,179 padding patches and preserves all 107,007 annotated points. Patches from a source photo stay in the same split: the final `_index` is removed to determine the photo group, then groups are split with seed 42. Augmentation occurs after this split.

| Prepared split | Source photos | Patches | Annotated points |
| --- | ---: | ---: | ---: |
| Training | 235 | 2,721 | 84,257 |
| Validation | 59 | 673 | 22,750 |

There are 762 retained empty patches, including 136 in validation. Counts always come from the annotations. When an actual image dimension is 256, a coordinate equal to 256 is corrected to 255 in the prepared split files only; 19 coordinates require this adjustment. Other out-of-bounds coordinates are rejected.

Photo grouping prevents patches of the same named photo from crossing splits. It does not prove that different photo names represent independent capture sessions. Near-duplicate captures and annotation completeness remain dataset limitations. Keep new test data separate from training and parameter selection.

## Model, targets and loss

`model.py` constructs `segmentation_models_pytorch.Unet` with an ImageNet-pretrained ResNet-50 encoder, three RGB input channels and one output-logit channel. Both encoder and decoder are trained. RGB inputs use ImageNet channel normalization; there is no classification centre crop.

Targets are float32 Gaussian maps with a peak value of 1 at each integer centre. Overlapping Gaussians are combined by their maximum. They are continuous localization targets, not binary segmentation masks or unit-mass density maps. Gaussian width is selected when preparing targets: `--sigma 3` means a standard deviation of three target-image pixels.

TIFF preserves float target values. A filename such as `Photo (101)_0.jpg.tiff` intentionally keeps the full RGB filename and adds the target extension. Some editors cannot display float TIFFs; view the PNG previews or HTML reports instead. Display limitations do not affect training.

The loss in `losses.py` is:

```text
prediction = sigmoid(logits)
loss = mean((1 + alpha * target) * (prediction - target)^2)
alpha = 10
```

The extra weight emphasizes rod-centre targets over the large background. Loss arithmetic uses float32 even with mixed-precision convolution. This is a heatmap regression loss; count MAE is an evaluation and checkpoint-selection metric, not a differentiable training loss.

To count rods, `metrics.py` finds 3 x 3 local maxima meeting the configured score threshold, reduces equal-score plateaus to one location and suppresses lower-scoring centres closer than `--min-distance` pixels (default 3). Scores are not calibrated probabilities. Point precision/recall/F1 use maximum-cardinality one-to-one matching within `--match-tolerance` pixels (default 5).

The peak threshold and Gaussian sigma are different parameters. Changing the threshold does not regenerate targets or alter loss gradients. During training it can change the selected epoch and early-stop time because it changes validation counts. Changing sigma changes regression targets and requires a separate training experiment.

## Train the preferred configuration

Run dependency/unit checks and a no-update CUDA preflight first:

```powershell
& $python -m pip check
& $python -m unittest discover -s tests -v

& $python -m rod_counting.train --check-only --data data/prepared/training_v1 --heatmaps data/prepared/training_v1/heatmaps_sigma3 --batch-size 8 --seed 42 --peak-threshold 0.5 --stretch 0 --zoom 0
```

The CUDA preflight performs a real batch forward/backward and checks finite gradients, but takes no optimizer step and writes no checkpoint. It does not establish convergence or memory fit after AdamW state allocation. A CPU forward check is also available with `--device cpu --check-only --batch-size 2`.

Start training manually in a fresh directory:

```powershell
$run = "runs/sigma3/patches_sigma3_seed42_thr050_pat30_$(Get-Date -Format yyyyMMdd_HHmmss)"

& $python -m rod_counting.train --data data/prepared/training_v1 --heatmaps data/prepared/training_v1/heatmaps_sigma3 --batch-size 8 --seed 42 --peak-threshold 0.5 --stretch 0 --zoom 0 --patience 30 --epochs 100 --run-dir $run
```

This trains on 2,721 patches and validates on all 673 validation patches every epoch. AdamW uses learning rate `0.0001` and weight decay `0.0001`. CUDA AMP is enabled. Basic augmentation uses paired horizontal/vertical flips and 90-degree rotations, followed by RGB-only brightness, contrast and colour changes in the range 0.9–1.1. Zoom and stretching are disabled in this configuration. Validation is unaugmented.

`best.pt` is updated whenever validation count MAE reaches a new minimum. Training stops after 30 consecutive epochs without an improvement greater than `--min-delta 0.001`, or at 100 epochs. Saving the best checkpoint and the early-stop tolerance are separate: even a smaller improvement can update `best.pt`. There is no learning-rate scheduler or interrupted-run resume command. Restart an interrupted experiment with a new run directory.

Batch size affects optimization as well as memory use. Preserve batch size, seed, data split, targets and other settings for controlled comparisons. Seeding requests deterministic execution but does not guarantee identical numbers across different hardware or package versions.

## Monitor training

The terminal prints the absolute path of `progress.html`. Open it after the first epoch finishes:

```powershell
Invoke-Item "$run/progress.html"
```

The page refreshes every 30 seconds and shows loss, count metrics, localization metrics and four fixed validation examples spanning annotation counts. Previous epoch pages remain available. Preview panels show RGB, target heatmap, predicted heatmap, prediction overlay and predicted green circles against annotated yellow crosses. Heatmaps use fixed 0–1 brightness; a faint heatmap may still contain peaks below the counting threshold.

Each new training run writes:

| File | Contents |
| --- | --- |
| `config.json` | Arguments, data checksums, Gaussian settings, runtime and source revision |
| `environment.txt` | Installed package versions |
| `metrics.csv`, `metrics.jsonl` | One metric row per completed epoch |
| `progress.html` | Latest plots and validation previews |
| `epoch_001.html`, etc. | Saved per-epoch pages |
| `previews/*.png` | Viewable validation prediction contact sheets |
| `best.pt` | Lowest validation count-MAE checkpoint |
| `last.pt` | Latest checkpoint, which may perform worse |

The archived run collection retains `best.pt`, settings and all logs/previews; redundant `last.pt` files are omitted. New training still writes both checkpoints. Checkpoints include model, optimizer/scaler state, settings, epoch and validation metrics; the supplied CLI evaluates them but does not implement resuming.

Open `runs/index.html` for the local experiment archive. Each entry shows the experiment type, sigma, seed, batch size, patience, counting threshold, epochs and best validation MAE, with links to progress pages, settings, metrics and weights. Run folders retain their saved names so historical metadata remains intact. Runs with matching settings are separate recorded attempts.

## Evaluate a saved checkpoint

The preferred saved checkpoint is:

```text
runs/sigma3/seed42_patience30_20261028_133154/best.pt
```

Evaluate it on the prepared validation patches at threshold 0.65:

```powershell
$checkpoint = "runs/sigma3/seed42_patience30_20261028_133154/best.pt"
$validation = "artifacts/validation_thr065_$(Get-Date -Format yyyyMMdd_HHmmss)"

& $python -m rod_counting.test $checkpoint --peak-threshold 0.65 --output $validation
Invoke-Item $validation
```

`test.py` loads saved weights without pretrained initialization or training. It prints count MAE, RMSE, signed bias, point precision/recall/F1 and heatmap loss. The new output directory receives `metrics.json` and fixed validation preview PNGs. Omit `--output` to print metrics only. Add `--device cpu` for CPU evaluation.

By default, data and model settings come from the checkpoint. After relocating external data, supply `--data <prepared-root> --heatmaps <target-root>`. Checkpoints selected using scene validation automatically use their saved scene root; `--validation-scenes <scene-root>` explicitly selects complete-scene validation.

Choose any further threshold comparisons on validation. Repeated evaluation of test images makes those results exploratory rather than an untouched estimate of performance.

## Run whole-photo inference

Use the preferred checkpoint, an explicit **1024 x 768** whole-image resize and threshold **0.65**:

```powershell
$checkpoint = "runs/sigma3/seed42_patience30_20261028_133154/best.pt"
$prediction = "artifacts/test_1024_thr065_$(Get-Date -Format yyyyMMdd_HHmmss)"

& $python -m rod_counting.inference $checkpoint --size 1024 768 --peak-threshold 0.65 --save-previews --output $prediction
if ($LASTEXITCODE -eq 0) { Invoke-Item $prediction }
```

Default input is `data/raw/segmentation_dataset/test_data` with `test_labels.json`. For another annotated collection, use `--data <photo-directory> --labels <labels.json>`. Every image must have exactly one matching annotation record. This entry point currently requires labels; it is not an unlabeled prediction command.

Inference processes each complete image in one forward pass, without tiling. It pads dimensions to multiples of 32 internally and removes the padding from the output. `--size` takes **width, then height**. Omitting it uses native resolution. The verified 1024 x 768 scale performs substantially better than native 4K or resizing an entire photo to 256 x 256. Matching the patch dimensions by shrinking a whole dense scene does not preserve rod size.

The output contains `counts.csv`, `metrics.json` and, with `--save-previews`, one PNG per photo. Each preview contains RGB, the predicted heatmap and green predicted centres with yellow reference centres. The metrics are in **rods per full photo**, not rods per patch. Positive count bias means overcounting; negative bias means undercounting. An accurate total count can still conceal false detections and missed rods that cancel each other.

## Controlled experiments

Use separate `--run-dir` paths for every training attempt, reuse the fixed split and change one setting at a time where possible. Keep alternate Gaussian targets in separate directories. Existing output directories are refused so earlier runs are preserved.

### Gaussian width

For example, create new sigma-4 targets without changing sigma-3 targets:

```powershell
& $python scripts/data_preparation/create_heatmaps.py --train-dir data/prepared/training_v1/train_data --splits data/prepared/training_v1/splits --output data/prepared/training_v1/heatmaps_sigma4 --sigma 4
```

Select them with `--heatmaps data/prepared/training_v1/heatmaps_sigma4`. Compare count/localization metrics at the same evaluation threshold; heatmap losses across different target widths are not directly comparable.

### Patch zoom/crop or stretching

Add `--zoom 0.2 --stretch 0` for a separate patch experiment. On 50% of training samples, it applies a uniform zoom in the range 0.8–1.2, with random crop/padding back to 256 x 256. Centre coordinates are transformed, out-of-frame points are discarded and fixed-sigma heatmaps are regenerated. Validation stays unchanged.

Alternatively use `--stretch 0.15 --zoom 0` for independent axis compression and centred padding. It transforms RGB and the continuous heatmap together, changing target widths. Zoom and stretching cannot be combined by this CLI. Both default to zero and neither is part of the preferred configuration.

### Reconstructed scenes

For complete groups of twelve patches, scene preparation uses a 4-column x 3-row, row-major layout. It offsets centre coordinates according to patch placement, then regenerates heatmaps. Incomplete groups retain individual patches; missing patches are never filled with invented content.

The existing `data/prepared/larger_scenes_v1` contains 240 complete 1024 x 768 scenes plus 514 individual patches, with all original points and photo splits retained: 194 scenes + 393 patches for training, and 46 scenes + 121 patches for validation.

To create this data in a fresh output directory:

```powershell
& $python scripts/data_preparation/reconstruct_scenes.py --data data/prepared/training_v1 --output data/prepared/larger_scenes_v1
& $python scripts/data_preparation/create_heatmaps.py --train-dir data/prepared/larger_scenes_v1/train_data --splits data/prepared/larger_scenes_v1/splits --output data/prepared/larger_scenes_v1/heatmaps --sigma 3
```

Train on mixed reconstructed scenes and leftover patches by selecting `--data data/prepared/larger_scenes_v1 --batch-size 1 --peak-threshold 0.65 --stretch 0 --zoom 0` with a separate run directory. Batch size 1 accommodates variable dimensions and 90-degree rotations. Logs include separate scene and patch MAE; checkpoint selection uses their combined per-sample MAE.

For patch training with complete-scene checkpoint selection, retain the normal patch arguments and add `--validation-scenes data/prepared/larger_scenes_v1`. Training still uses batches of 8 patches; validation evaluates all 46 complete validation scenes one at a time. `val/*` controls checkpoint selection and early stopping. All 673 original patch-validation metrics are also logged under `patch_val/*`. Omitting the flag retains normal patch validation.

These scene variants remain available, but their recorded test performance did not improve on patch training. Scene MAE, mixed-sample MAE and patch MAE have different units and should not be compared directly.

### Compare saved runs

```powershell
& $python scripts/compare_runs.py runs/sigma3/refactored_thr05_seed42_20261026_224434 runs/sigma3/seed42_patience30_20261028_133154 --output artifacts/run_comparison.csv
```

This reads saved settings and epoch metrics without loading weights. It compares the best selection-MAE epoch and warns when controlled settings differ. Choose a new CSV output path under `artifacts/` or `runs/`. The local run index also links saved validation and inference evaluations, including comparisons at alternate thresholds.

## Saved results

The preferred checkpoint is epoch **83** of `runs/sigma3/seed42_patience30_20261028_133154`. Training used sigma 3 targets, seed 42, batch size 8, threshold 0.5 for checkpoint selection, patience 30, up to 100 epochs, and no zoom/stretch.

At evaluation threshold **0.65**, it achieved validation patch count MAE **1.5082**, RMSE **2.8261** and localization F1 **0.9438**. On 19 full test photos resized to 1024 x 768, it achieved count MAE **8.0526 rods/photo**, RMSE **9.4618** and bias **+2.8947 rods/photo**: 8,334 predicted against 8,279 annotated rods.

| Configuration | Test threshold | Full-photo count MAE | RMSE |
| --- | ---: | ---: | ---: |
| Patch sigma 3, seed 42, patience 30 — preferred | 0.65 | **8.0526** | **9.4618** |
| Patch sigma 3, seed 42, patience 15 | 0.65 | 8.6842 | 11.3994 |
| Patch sigma 3, seed 43, patience 30 | 0.65 | 9.1579 | 12.9371 |
| Patch sigma 3, seed 43, patience 15 | 0.65 | 9.6316 | 12.3267 |
| Patch sigma 3, zoom 0.2, seed 42 | 0.65 | 10.9474 | 14.8288 |
| Patch training / scene validation, seed 43 | 0.60 | 17.7895 | 21.7643 |
| Mixed scenes + patches, seed 42 | 0.65 | 19.6842 | 29.5083 |

These are observed comparisons on the same repeatedly inspected 19 photos. They are not an unbiased estimate on a new test set, and the scene-selection row also uses a different threshold. Dataset labels can be incomplete or offset, and shaft highlights can resemble rod ends. Inspect overlays alongside counts. Do not assume every unmatched centre is a real false positive or that a near-correct count implies correct localization.

Exact per-photo results and previews for the preferred checkpoint are saved locally under `artifacts/seed42_repeat_test_20261028_173510/`. Browse the other archived experiments through `runs/index.html`.

## Command-line settings

Run `& $python -m rod_counting.train --help` for the full interface. Defaults are preserved for existing experiments; use the explicit preferred command above instead of assuming every default matches it.

| Option | Default | Meaning |
| --- | --- | --- |
| `--data` | `data/prepared/training_v1` | Prepared image/split root |
| `--heatmaps` | `<data>/heatmaps` | Override target root; preferred uses `heatmaps_sigma3` |
| `--run-dir` | `runs/resnet50_baseline` | New training output directory |
| `--validation-scenes` | Unset | Optional complete-scene validation root |
| `--encoder` | `resnet50` | `resnet18`, `resnet34` or `resnet50` |
| `--batch-size` / `--workers` | `8` / `0` | Training batch size and loader workers |
| `--epochs` / `--patience` | `100` / `15` | Epoch cap and stale-epoch allowance |
| `--min-delta` | `0.001` | MAE improvement required to reset early-stop patience |
| `--lr` / `--weight-decay` | `0.0001` / `0.0001` | AdamW settings |
| `--loss-alpha` | `10` | Foreground weighting for MSE |
| `--peak-threshold` | `0.3` | Counting threshold; preferred training explicitly uses `0.5` |
| `--min-distance` / `--match-tolerance` | `3` / `5` | Peak suppression / point-match distance in current image pixels |
| `--seed` | `42` | Initialization, order and augmentation RNG seed; saved split stays fixed |
| `--stretch` / `--zoom` | `0` / `0` | Optional geometry experiments |
| `--preview-count` | `4` | Fixed validation examples saved per epoch |
| `--no-augmentation` | Off | Disable training augmentation |
| `--no-amp` | Off | Disable CUDA mixed precision |
| `--no-pretrained` | Off | Random encoder initialization for deliberate separate checks/experiments |
| `--check-only` | Off | No-update preflight instead of training |
| `--device` | `cuda` | `cuda` or `cpu`; full training requires CUDA |

## Verification and troubleshooting

The unit suite checks paired augmentation, target precision and normalization, loss gradients, peak detection and point matching, source-photo isolation, scene reconstruction, checkpoint selection, local reporting and inference geometry. Use the preflight before a new training configuration; full training is started separately.

- **PowerShell reports an invalid expression after `&`:** set `$python` again in this terminal. Confirm `Test-Path $python`. Variables such as `$run` and `$checkpoint` are also local to the current terminal.
- **The wrong checkout is imported:** run `& $python -c "import rod_counting; print(rod_counting.__file__)"`, then reinstall with `& $python -m pip install -e .` from the intended folder.
- **A checkpoint path is missing:** use the actual folder name from `runs/index.html`; shell variables do not survive closing the terminal. The preferred checkpoint's literal path is shown above.
- **An output directory already exists:** select a fresh timestamped name. Existing experiments are deliberately protected.
- **CUDA runs out of memory:** close other GPU workloads and preflight a smaller batch size. A changed batch size is a new optimization configuration. Use batch size 1 for mixed-dimension scene training.
- **TIFF looks black or is unsupported in the editor:** inspect generated PNGs/HTML. Targets are continuous float maps and should not be converted to binary masks to fix a display issue.
- **Full-photo counts are unexpectedly poor:** check `--size 1024 768`, the checkpoint, threshold and label collection. Native 4K and whole-photo 256 x 256 inference are different input scales.
- **No unlabeled predictions:** the current inference CLI expects reference JSON labels. Count comparison and annotated overlays depend on them.

To use this repository without the local archive, provide the raw dataset, run preparation, install dependencies and train. Source plus the same raw inputs reproduces the preparation/training procedure; identical numerical results across all runtimes are not guaranteed.
