"""Training loop.

Forward pass, cross-entropy loss with padding excluded via ignore_index,
backward pass, optimizer step, periodic validation, and checkpointing.
"""
import argparse
import os
import random

import numpy as np
import torch
import yaml
from torch import nn, optim
from torch.utils.data import DataLoader

from src.data import ShapeScenesDataset, collate_fn
from src.model.vlm import TinyVLM
from src.tokenizer import CharacterTokenizer


def set_seed(seed: int) -> None:
    """Make training as reproducible as practical."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def prepare_inputs(
    targets: torch.Tensor,
    tokenizer: CharacterTokenizer,
) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Convert padded target sequences into:
      input_ids  -> valid token IDs for the model
      expected_ids -> original targets, including -100 padding

    Example:

        targets:
        [l, a, r, e, eos, -100]

        input_ids:
        [l, a, r, e]

        expected_ids:
        [l, a, r, e, eos, -100]
    """
    input_ids = targets[:, :-1].clone()

    # -100 is valid for the loss but invalid for nn.Embedding.
    input_ids[input_ids == -100] = tokenizer.eos_id

    expected_ids = targets

    return input_ids, expected_ids


def compute_loss(
    model: nn.Module,
    images: torch.Tensor,
    targets: torch.Tensor,
    tokenizer: CharacterTokenizer,
    criterion: nn.Module,
) -> torch.Tensor:
    """Run the model and calculate the next-token loss."""

    input_ids, expected_ids = prepare_inputs(targets, tokenizer)

    logits = model(images, input_ids)

    # Number of visual tokens is inferred from:
    #
    # total sequence length = visual tokens + input text length
    #
    t_vis = logits.shape[1] - input_ids.shape[1]

    # The LAST visual token predicts the FIRST target token.
    #
    # Then:
    # text position 0 predicts target 1
    # text position 1 predicts target 2
    # ...
    #
    # Therefore we need exactly len(expected_ids) predictions.
    shift_logits = logits[:, t_vis - 1 :, :].contiguous()

    if shift_logits.shape[1] != expected_ids.shape[1]:
        raise RuntimeError(
            "Prediction/target length mismatch: "
            f"logits={shift_logits.shape}, "
            f"targets={expected_ids.shape}"
        )

    loss = criterion(
        shift_logits.view(-1, tokenizer.vocab_size),
        expected_ids.view(-1),
    )

    return loss


@torch.no_grad()
def evaluate_loss(
    model: nn.Module,
    loader: DataLoader,
    tokenizer: CharacterTokenizer,
    criterion: nn.Module,
    device: torch.device,
    blind_baseline: bool,
) -> float:
    """Calculate average validation loss."""
    model.eval()

    total_loss = 0.0
    total_batches = 0

    for images, targets in loader:
        images = images.to(device)
        targets = targets.to(device)

        if blind_baseline:
            images = torch.zeros_like(images)

        loss = compute_loss(
            model,
            images,
            targets,
            tokenizer,
            criterion,
        )

        total_loss += loss.item()
        total_batches += 1

    return total_loss / total_batches


def train() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config",
        type=str,
        default="configs/baseline.yaml",
    )
    parser.add_argument(
        "--sanity-check",
        action="store_true",
        help="E0: overfit one batch.",
    )
    args = parser.parse_args()

    # ------------------------------------------------------------
    # 1. Load configuration
    # ------------------------------------------------------------
    with open(args.config, "r", encoding="utf-8") as f:
        config = yaml.safe_load(f)

    seed = config.get("seed", 42)
    set_seed(seed)

    # ------------------------------------------------------------
    # 2. Choose CPU or GPU
    # ------------------------------------------------------------
    device = torch.device(
        "cuda" if torch.cuda.is_available() else "cpu"
    )

    print(f"Using device: {device}")
    print(f"Seed: {seed}")

    # ------------------------------------------------------------
    # 3. Tokenizer + datasets
    # ------------------------------------------------------------
    tokenizer = CharacterTokenizer()

    train_dataset = ShapeScenesDataset(
        f"{config['data_path']}/train.pt",
        tokenizer,
    )

    val_dataset = ShapeScenesDataset(
        f"{config['data_path']}/val.pt",
        tokenizer,
    )

    train_loader = DataLoader(
        train_dataset,
        batch_size=config["batch_size"],
        shuffle=True,
        collate_fn=collate_fn,
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=config["batch_size"],
        shuffle=False,
        collate_fn=collate_fn,
    )

    # ------------------------------------------------------------
    # 4. Model
    # ------------------------------------------------------------
    model = TinyVLM(
        vocab_size=tokenizer.vocab_size,
        d_model=config.get("d_model", 128),
        n_heads=config.get("n_heads", 4),
        n_layers=config.get("n_layers", 2),
    ).to(device)

    optimizer = optim.AdamW(
        model.parameters(),
        lr=config["learning_rate"],
    )

    criterion = nn.CrossEntropyLoss(
        ignore_index=-100
    )

    blind_baseline = config.get("blind_baseline", False)

    # ------------------------------------------------------------
    # 5. E0: overfit exactly one batch
    # ------------------------------------------------------------
    if args.sanity_check:
        print("\n=== E0 SANITY CHECK ===")
        print("Goal: overfit one batch.")
        print("This is a bug check, not a final result.\n")

        sanity_batch = next(iter(train_loader))

        epochs = 100

        model.train()

        for epoch in range(epochs):
            images, targets = sanity_batch

            images = images.to(device)
            targets = targets.to(device)

            if blind_baseline:
                images = torch.zeros_like(images)

            optimizer.zero_grad()

            loss = compute_loss(
                model,
                images,
                targets,
                tokenizer,
                criterion,
            )

            loss.backward()
            optimizer.step()

            if epoch % 10 == 0 or epoch == epochs - 1:
                print(
                    f"E0 epoch {epoch:3d} | "
                    f"loss = {loss.item():.6f}"
                )

        print("\nE0 complete.")

        return

    # ------------------------------------------------------------
    # 6. Normal training
    # ------------------------------------------------------------
    epochs = config["epochs"]
    if blind_baseline:
        checkpoint_dir = "checkpoints/blind"
    else:
        checkpoint_dir = "checkpoints"

    os.makedirs(checkpoint_dir, exist_ok=True)

    best_val_loss = float("inf")



    for epoch in range(epochs):

        model.train()

        total_train_loss = 0.0
        train_batches = 0

        for batch_idx, (images, targets) in enumerate(train_loader):

            images = images.to(device)
            targets = targets.to(device)

            if blind_baseline:
                images = torch.zeros_like(images)

            optimizer.zero_grad()

            loss = compute_loss(
                model,
                images,
                targets,
                tokenizer,
                criterion,
            )

            loss.backward()
            optimizer.step()

            total_train_loss += loss.item()
            train_batches += 1

            if batch_idx % 50 == 0:
                print(
                    f"Epoch {epoch} | "
                    f"Batch {batch_idx} | "
                    f"Loss {loss.item():.4f}"
                )

        avg_train_loss = total_train_loss / train_batches

        # --------------------------------------------------------
        # Validation
        # --------------------------------------------------------
        avg_val_loss = evaluate_loss(
            model,
            val_loader,
            tokenizer,
            criterion,
            device,
            blind_baseline,
        )

        print(
            f"Epoch {epoch} | "
            f"Train Loss {avg_train_loss:.4f} | "
            f"Val Loss {avg_val_loss:.4f}"
        )

        # --------------------------------------------------------
        # Save checkpoint every epoch
        # --------------------------------------------------------
        checkpoint_path = f"{checkpoint_dir}/model_epoch_{epoch}.pt"

        torch.save(
            model.state_dict(),
                checkpoint_path,
)


        # Save best model separately
        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            best_checkpoint_path = f"{checkpoint_dir}/best.pt"

            torch.save(
                model.state_dict(),
                 best_checkpoint_path,
            )

    print(f"Saved new best checkpoint to {best_checkpoint_path}")


if __name__ == "__main__":
    train()