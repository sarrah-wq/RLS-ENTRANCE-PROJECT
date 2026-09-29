# S1 — Training Throughput Benchmark

## Question

How does training throughput change with batch size on my CPU?

## Method

I measured one training step consisting of:

forward pass → loss → backward pass → optimizer step

I tested the following batch sizes:

- 1
- 16
- 64
- 256

Warm-up steps: 5

Timed steps per measurement: 20

The same model architecture was used for every batch size.

## Data loading

Data loading was NOT included in the timing.

The benchmark created the input tensors before timing and kept them on the selected device.

## CPU Results

| Batch size | Step time (ms) | Throughput (images/s) |
|---:|---:|---:|
| 1 | 21.94 | 45.63 ± 1.48 |
| 16 | 140.67 | 114.44 ± 8.74 |
| 64 | 493.92 | 129.62 ± 2.44 |
| 256 | 1988.28 | 128.83 ± 3.07 |

## GPU

A GPU benchmark was not performed because the installed PyTorch version is CPU-only and CUDA was not available.

GPU detected by the operating system:
AMD Radeon(TM) Graphics

PyTorch:
2.4.1+cpu

## Hardware

CPU:
AMD Ryzen 5 7520U with Radeon Graphics

GPU:
AMD Radeon(TM) Graphics

Operating system:
Windows 11

Python:
3.12.0

PyTorch:
2.4.1+cpu

## Interpretation

Throughput increased substantially from batch size 1 to batch size 64.

Increasing the batch size from 64 to 256 did not increase throughput further; the measured throughput was approximately the same.

Therefore, for this CPU and this model, increasing the batch size beyond 64 did not provide a clear throughput improvement.

The benchmark did not include data-loading time, so these measurements describe model training computation rather than the complete data-to-training pipeline.