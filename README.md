# RLS Tiny VLM — Entrance Challenge

A small vision-language model built from scratch in PyTorch for the RLS Entrance Challenge.

The model receives a 64×64 RGB image containing one or two colored geometric shapes and generates the scene description one character at a time.

Example:

    image
      ↓
    CNN encoder
      ↓
    visual tokens
      ↓
    transformer decoder
      ↓
    next-character prediction
      ↓
    generated word

Example output:

    largeredcircle

## Project structure

    src/
    ├── tokenizer.py
    ├── data.py
    ├── train.py
    ├── evaluate.py
    ├── generate.py
    └── model/
        ├── encoder.py
        ├── attention.py
        ├── decoder.py
        └── vlm.py

    tests/
    ├── test_attention.py
    └── test_shapes.py

    experiments/
    └── E1_blind/
        └── notes.md

    benchmarks/
    └── S1_throughput/
        ├── benchmark.py
        ├── timings.csv
        ├── throughput.png
        ├── hardware.txt
        └── notes.md

## Main results

| Experiment | Configuration | Metric | Result |
|---|---|---|---:|
| Main model | `configs/baseline.yaml` | Exact match, test | 70.35% |
| Main model | `configs/baseline.yaml` | Size accuracy, test | 84.14% |
| Main model | `configs/baseline.yaml` | Color accuracy, test | 74.40% |
| Main model | `configs/baseline.yaml` | Shape accuracy, test | 74.24% |
| Main model | `configs/baseline.yaml` | Relation accuracy, test | 43.43% |
| Main model | `configs/baseline.yaml` | Letter accuracy, test | 69.76% |
| E1 Blind | `configs/blind.yaml` | Exact match, test | 1.65% |
| E1 Blind | `configs/blind.yaml` | Size accuracy, test | 33.28% |
| E1 Blind | `configs/blind.yaml` | Color accuracy, test | 20.10% |
| E1 Blind | `configs/blind.yaml` | Shape accuracy, test | 20.63% |
| E1 Blind | `configs/blind.yaml` | Relation accuracy, test | 0.00% |
| E1 Blind | `configs/blind.yaml` | Letter accuracy, test | 17.05% |

## E1 — Blind baseline

Question:

> Does the model use the image?

The normal model was trained with the original images.

The blind model used the same model architecture and training setup, but the images were replaced by zero tensors during training and validation.

The normal model achieved 70.35% exact match on the test split, while the blind model achieved 1.65%.

This large drop indicates that the normal model relies substantially on visual information.

The blind model still had some non-zero accuracy, showing that it could learn some structure from the character sequences and dataset grammar without useful visual input.

The complete E1 experiment log is in:

    experiments/E1_blind/notes.md

## S1 — Training throughput

Training throughput was measured for batch sizes:

    1, 16, 64, 256

The benchmark measured:

    forward → loss → backward → optimizer step

Data loading was NOT included in the timing.

### CPU results

| Batch size | Step time (ms) | Throughput (images/s) |
|---:|---:|---:|
| 1 | 21.94 | 45.63 ± 1.48 |
| 16 | 140.67 | 114.44 ± 8.74 |
| 64 | 493.92 | 129.62 ± 2.44 |
| 256 | 1988.28 | 128.83 ± 3.07 |

Throughput increased from batch size 1 to 64, then remained approximately unchanged at batch size 256.

The benchmark used warm-up steps and CUDA synchronization when CUDA was available.

CUDA was not available in the current PyTorch installation, so the recorded benchmark is CPU-only.

Benchmark files are in:

    benchmarks/S1_throughput/

## Additional evaluation: test-heldout

The project also evaluates the special held-out split.

The held-out split contains color-shape combinations that do not appear in the training split, while the individual colors and shapes are still present.

Results:

| Metric | Test | Test-heldout |
|---|---:|---:|
| Exact match | 70.35% | 0.10% |
| Size accuracy | 84.14% | 83.76% |
| Color accuracy | 74.40% | 76.44% |
| Shape accuracy | 74.24% | 26.14% |
| Relation accuracy | 43.43% | 47.37% |

The largest drop is in shape accuracy.

## Correctness checks

The provided tests were run successfully:

    100 passed, 1 skipped

The single skipped test was the CUDA test because CUDA was not available.

## Reproducing the project

### 1. Activate the virtual environment

PowerShell:

    .venv\Scripts\Activate.ps1

### 2. Run the tests

    python -m pytest -q

### 3. Run E0

    python -m src.train --config configs/baseline.yaml --sanity-check

E0 checks whether the training pipeline can overfit one batch.

### 4. Train the normal model

    python -m src.train --config configs/baseline.yaml

### 5. Evaluate the normal model

    python -m src.evaluate --checkpoint checkpoints\best.pt --split test

### 6. Evaluate the held-out split

    python -m src.evaluate --checkpoint checkpoints\best.pt --split test_heldout

### 7. Train the blind baseline

    python -m src.train --config configs/blind.yaml

The blind checkpoint is saved to:

    checkpoints\blind\best.pt

### 8. Evaluate the blind baseline

    python -m src.evaluate --checkpoint checkpoints\blind\best.pt --split test --blind

### 9. Run the S1 benchmark

From the project root:

    python -m benchmarks.S1_throughput.benchmark

The benchmark writes:

    benchmarks/S1_throughput/timings.csv
    benchmarks/S1_throughput/throughput.png
    benchmarks/S1_throughput/hardware.txt

## Hardware

CPU:

    AMD Ryzen 5 7520U with Radeon Graphics

GPU:

    AMD Radeon(TM) Graphics

PyTorch:

    2.4.1+cpu

Python:

    3.12.0

OS:

    Windows 11

CUDA:

    Not available in the installed PyTorch build.

## Notes

The model uses handwritten multi-head self-attention rather than PyTorch's high-level transformer attention modules.

The project focuses on the Level 1 requirements of the RLS challenge.