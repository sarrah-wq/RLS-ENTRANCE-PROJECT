import csv
import platform
import time
from pathlib import Path

import matplotlib.pyplot as plt
import torch
from torch import nn

from src.model.vlm import TinyVLM


BATCH_SIZES = [1, 16, 64, 256]

NUM_WARMUP = 5
NUM_BATCHES = 20
NUM_REPEATS = 3

SEQ_LEN = 15
VOCAB_SIZE = 27


def synchronize(device):
    """Wait for GPU work to finish before measuring time."""
    if device.type == "cuda":
        torch.cuda.synchronize()


def benchmark_training(batch_size, device):
    """
    Measure training throughput.

    Data loading is NOT included.
    The batch is created before timing and stays on the device.
    """

    model = TinyVLM().to(device)
    model.train()

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=0.001,
    )

    criterion = nn.CrossEntropyLoss()

    # Fake batch, already on the selected device.
    images = torch.randn(
        batch_size,
        3,
        64,
        64,
        device=device,
    )

    input_ids = torch.randint(
        0,
        VOCAB_SIZE,
        (batch_size, SEQ_LEN),
        device=device,
    )

    targets = torch.randint(
        0,
        VOCAB_SIZE,
        (batch_size, SEQ_LEN),
        device=device,
    )

    # ---------------------------------------------------------
    # Warm-up
    # ---------------------------------------------------------

    for _ in range(NUM_WARMUP):
        optimizer.zero_grad()

        logits = model(images, input_ids)

        # Use the text part of the sequence.
        text_logits = logits[:, -SEQ_LEN:, :]

        loss = criterion(
            text_logits.reshape(-1, VOCAB_SIZE),
            targets.reshape(-1),
        )

        loss.backward()
        optimizer.step()

    synchronize(device)

    # ---------------------------------------------------------
    # Timed steps
    # ---------------------------------------------------------

    times = []

    for _ in range(NUM_BATCHES):
        synchronize(device)

        start = time.perf_counter()

        optimizer.zero_grad()

        logits = model(images, input_ids)

        text_logits = logits[:, -SEQ_LEN:, :]

        loss = criterion(
            text_logits.reshape(-1, VOCAB_SIZE),
            targets.reshape(-1),
        )

        loss.backward()
        optimizer.step()

        synchronize(device)

        end = time.perf_counter()

        times.append(end - start)

    avg_step_time = sum(times) / len(times)

    images_per_second = batch_size / avg_step_time

    return avg_step_time, images_per_second


def save_hardware_info(output_dir):
    """Save basic hardware/software information."""

    path = output_dir / "hardware.txt"

    with open(path, "w", encoding="utf-8") as f:
        f.write(f"OS: {platform.platform()}\n")
        f.write(f"CPU: {platform.processor()}\n")
        f.write(f"Python: {platform.python_version()}\n")
        f.write(f"PyTorch: {torch.__version__}\n")

        if torch.cuda.is_available():
            f.write(f"GPU: {torch.cuda.get_device_name(0)}\n")
            f.write(f"CUDA: {torch.version.cuda}\n")
        else:
            f.write("GPU: CUDA not available\n")


def save_plot(results, output_dir):
    """Create throughput plot."""

    path = output_dir / "throughput.png"

    plt.figure(figsize=(8, 5))

    devices = sorted(set(row["device"] for row in results))

    for device in devices:
        device_rows = [
            row for row in results
            if row["device"] == device
        ]

        batch_sizes = [
            row["batch_size"]
            for row in device_rows
        ]

        throughput = [
            row["mean_images_per_second"]
            for row in device_rows
        ]

        plt.plot(
            batch_sizes,
            throughput,
            marker="o",
            label=device,
        )

    plt.xlabel("Batch size")
    plt.ylabel("Training throughput (images/s)")
    plt.title("S1 Training Throughput")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()

    plt.savefig(path)
    plt.close()


def main():
    output_dir = Path("benchmarks/S1_throughput")
    output_dir.mkdir(parents=True, exist_ok=True)

    save_hardware_info(output_dir)

    print("=== S1 Training Throughput Benchmark ===")
    print("Data loading included: NO")
    print(f"Warm-up steps: {NUM_WARMUP}")
    print(f"Timed steps per repeat: {NUM_BATCHES}")
    print(f"Repeats: {NUM_REPEATS}")
    print()

    devices = ["cpu"]

    if torch.cuda.is_available():
        devices.append("cuda")

    results = []

    for device_name in devices:

        device = torch.device(device_name)

        print(f"Device: {device_name}")
        print()

        for batch_size in BATCH_SIZES:

            repeat_throughputs = []
            repeat_step_times = []

            for repeat in range(NUM_REPEATS):

                step_time, images_per_second = benchmark_training(
                    batch_size=batch_size,
                    device=device,
                )

                repeat_step_times.append(step_time)
                repeat_throughputs.append(images_per_second)

            mean_step_time = sum(repeat_step_times) / len(repeat_step_times)
            mean_throughput = sum(repeat_throughputs) / len(repeat_throughputs)

            variance = sum(
                (x - mean_throughput) ** 2
                for x in repeat_throughputs
            ) / len(repeat_throughputs)

            std_throughput = variance ** 0.5

            print(
                f"Batch {batch_size:3d} | "
                f"Step: {mean_step_time * 1000:8.2f} ms | "
                f"Throughput: "
                f"{mean_throughput:8.2f} ± "
                f"{std_throughput:.2f} images/s"
            )

            results.append(
                {
                    "device": device_name,
                    "batch_size": batch_size,
                    "mean_step_time_ms": mean_step_time * 1000,
                    "mean_images_per_second": mean_throughput,
                    "std_images_per_second": std_throughput,
                }
            )

        print()

    # ---------------------------------------------------------
    # Save CSV
    # ---------------------------------------------------------

    csv_path = output_dir / "timings.csv"

    with open(csv_path, "w", newline="", encoding="utf-8") as f:

        writer = csv.DictWriter(
            f,
            fieldnames=[
                "device",
                "batch_size",
                "mean_step_time_ms",
                "mean_images_per_second",
                "std_images_per_second",
            ],
        )

        writer.writeheader()
        writer.writerows(results)

    save_plot(results, output_dir)

    print("Saved:")
    print(f"  {csv_path}")
    print(f"  {output_dir / 'throughput.png'}")
    print(f"  {output_dir / 'hardware.txt'}")


if __name__ == "__main__":
    main()