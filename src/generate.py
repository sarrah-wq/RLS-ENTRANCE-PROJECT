"""Greedy decoding.

Generates one letter at a time from the visual tokens until <eos> or
45 letters, for inference and for evaluation.
"""
import torch

from src.tokenizer import CharacterTokenizer
from src.model.vlm import TinyVLM
from src.data import ShapeScenesDataset


@torch.no_grad()
def generate_word(model,image,tokenizer,max_letters=45,):
    """
    Generate a word one token at a time using greedy decoding.

    The model starts with the image only.
    The last visual token predicts the first character.
    After that, each generated character is fed back in.
    """

    model.eval()

    device = image.device

    # No start token.
    # The image alone predicts the first character.
    input_ids = torch.empty(
        (1, 0),
        dtype=torch.long,
        device=device,
    )

    generated_ids = []

    # Maximum = 45 letters + <eos>
    for _ in range(max_letters + 1):

        # Image + previously generated letters
        logits = model(image, input_ids)

        # The last position predicts the next token.
        next_token_logits = logits[:, -1, :]

        # Greedy decoding:
        # choose the token with the largest logit.
        next_token = torch.argmax(
            next_token_logits,
            dim=-1,
        )

        token_id = next_token.item()

        generated_ids.append(token_id)

        # Stop when the model says "I'm finished."
        if token_id == tokenizer.eos_id:
            break
        
        if len(generated_ids) >= max_letters:

            break
        # Give the newly generated token to the model
        # during the next iteration.
        input_ids = torch.cat(
            [
                input_ids,
                next_token.unsqueeze(0),
            ],
            dim=1,
        )

    return tokenizer.decode(generated_ids)



def main():
    import argparse

    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--checkpoint",
        type=str,
        default="checkpoints/best.pt",
    )

    parser.add_argument(
        "--split",
        type=str,
        default="test",
    )

    parser.add_argument(
        "--index",
        type=int,
        default=0,
    )

    args = parser.parse_args()

    # CPU or GPU
    device = torch.device(
        "cuda" if torch.cuda.is_available() else "cpu"
    )

    print("Using device:", device)

    # Tokenizer
    tokenizer = CharacterTokenizer()

    # Load the real test dataset
    dataset = ShapeScenesDataset(
        f"data/{args.split}.pt",
        tokenizer,
    )

    # Get one real image
    image, _ = dataset[args.index]

    # Ground-truth word
    ground_truth = dataset.words[args.index]

    # Add batch dimension:
    # (3, 64, 64) -> (1, 3, 64, 64)
    image = image.unsqueeze(0).to(device)

    # Create the same architecture used during training
    model = TinyVLM(
        vocab_size=tokenizer.vocab_size,
        d_model=128,
        n_heads=4,
        n_layers=2,
    ).to(device)

    # Load the trained weights
    model.load_state_dict(
        torch.load(
            args.checkpoint,
            map_location=device,
            weights_only=True,
        )
    )

    # Generate prediction
    prediction = generate_word(
        model,
        image,
        tokenizer,
    )

    print()
    print("Ground truth :", ground_truth)
    print("Prediction    :", prediction)


if __name__ == "__main__":
    main()