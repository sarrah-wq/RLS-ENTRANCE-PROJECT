"""Evaluation: exact-match and per-attribute accuracy.

Includes the parser that splits a (possibly misspelled) generated word
back into size / color / shape [/ relation / size / color / shape].
"""
import argparse
import re

import torch
from tqdm import tqdm

from src.tokenizer import CharacterTokenizer
from src.data import ShapeScenesDataset
from src.model.vlm import TinyVLM
from src.generate import generate_word


# ---------------------------------------------------------
# The words used by the dataset
# ---------------------------------------------------------

SIZE_NAMES = ["small", "large"]
COLOR_NAMES = ["red", "green", "blue", "yellow"]
SHAPE_NAMES = ["circle", "square", "triangle", "cross"]
RELATIONS = ["leftof", "rightof", "above", "below"]


# ---------------------------------------------------------
# Exact parser for correct ground-truth words
# ---------------------------------------------------------

OBJECT_RE = (
    rf"({'|'.join(SIZE_NAMES)})"
    rf"({'|'.join(COLOR_NAMES)})"
    rf"({'|'.join(SHAPE_NAMES)})"
)

WORD_RE = re.compile(
    rf"^{OBJECT_RE}"
    rf"(?:"
    rf"({'|'.join(RELATIONS)})"
    rf"{OBJECT_RE}"
    rf")?$"
)


def parse_ground_truth(word):
    """
    Parse a correct dataset word.

    Example:
        largeredcircle

    becomes:
        {
            "obj1": {
                "size": "large",
                "color": "red",
                "shape": "circle"
            },
            "relation": None,
            "obj2": None
        }
    """

    match = WORD_RE.fullmatch(word)

    if match is None:
        return None

    groups = match.groups()

    # First object
    obj1 = {
        "size": groups[0],
        "color": groups[1],
        "shape": groups[2],
    }

    # Single object
    if groups[3] is None:
        return {
            "obj1": obj1,
            "relation": None,
            "obj2": None,
        }

    # Two objects
    obj2 = {
        "size": groups[4],
        "color": groups[5],
        "shape": groups[6],
    }

    return {
        "obj1": obj1,
        "relation": groups[3],
        "obj2": obj2,
    }


# ---------------------------------------------------------
# Small edit-distance helper
# ---------------------------------------------------------

def edit_distance(a, b):
    """
    Number of character edits needed to turn a into b.

    Example:
        cirle -> circle
        distance = 1
    """

    rows = len(a) + 1
    cols = len(b) + 1

    dp = [[0] * cols for _ in range(rows)]

    # Empty string -> b
    for i in range(rows):
        dp[i][0] = i

    # a -> empty string
    for j in range(cols):
        dp[0][j] = j

    for i in range(1, rows):
        for j in range(1, cols):
            if a[i - 1] == b[j - 1]:
                cost = 0
            else:
                cost = 1

            dp[i][j] = min(
                dp[i - 1][j] + 1,       # delete
                dp[i][j - 1] + 1,       # insert
                dp[i - 1][j - 1] + cost,  # replace
            )

    return dp[-1][-1]


# ---------------------------------------------------------
# Fuzzy parser for model predictions
# ---------------------------------------------------------

def _match_sequence(word, parts):
    """
    Try to split a predicted word into known pieces.

    Example:
        largeredcirle

    can become:
        large + red + cirle

    and "cirle" is matched to "circle" with edit distance 1.
    """

    from functools import lru_cache

    @lru_cache(None)
    def solve(part_index, position):
        # We used every expected part
        if part_index == len(parts):
            if position == len(word):
                return 0, []
            return None

        candidates = parts[part_index]

        best_score = None
        best_tokens = None

        for candidate in candidates:
            # Allow:
            #   one missing character
            #   correct length
            #   one extra character
            min_len = max(1, len(candidate) - 1)
            max_len = len(candidate) + 1

            for piece_len in range(min_len, max_len + 1):
                end = position + piece_len

                if end > len(word):
                    continue

                piece = word[position:end]
                distance = edit_distance(piece, candidate)

                # Do not allow a very broken piece.
                if distance > 1:
                    continue

                result = solve(part_index + 1, end)

                if result is None:
                    continue

                next_score, next_tokens = result
                total_score = distance + next_score

                if best_score is None or total_score < best_score:
                    best_score = total_score
                    best_tokens = [candidate] + next_tokens

        if best_score is None:
            return None

        return best_score, best_tokens

    return solve(0, 0)


def parse_broken_word(word):
    """
    Parse a model prediction even when it contains small spelling errors.

    Example:
        largeredcirle

    is treated as:

        large + red + circle
    """

    if not isinstance(word, str):
        return None

    # Keep only lowercase letters.
    word = "".join(c for c in word.lower() if c.isalpha())

    if not word:
        return None

    # First try exact parsing.
    exact = parse_ground_truth(word)

    if exact is not None:
        return exact

    # Try a one-object scene.
    one_object_parts = [
        tuple(SIZE_NAMES),
        tuple(COLOR_NAMES),
        tuple(SHAPE_NAMES),
    ]

    one_result = _match_sequence(word, one_object_parts)

    # Try a two-object scene.
    two_object_parts = [
        tuple(SIZE_NAMES),
        tuple(COLOR_NAMES),
        tuple(SHAPE_NAMES),
        tuple(RELATIONS),
        tuple(SIZE_NAMES),
        tuple(COLOR_NAMES),
        tuple(SHAPE_NAMES),
    ]

    two_result = _match_sequence(word, two_object_parts)

    # Choose the parse with fewer mistakes.
    candidates = []

    if one_result is not None:
        candidates.append((one_result[0], one_result[1], 1))

    if two_result is not None:
        candidates.append((two_result[0], two_result[1], 2))

    if not candidates:
        return None

    candidates.sort(key=lambda x: x[0])
    score, tokens, object_count = candidates[0]

    # If the prediction is too badly broken, give up.
    if score > 2:
        return None

    obj1 = {
        "size": tokens[0],
        "color": tokens[1],
        "shape": tokens[2],
    }

    if object_count == 1:
        return {
            "obj1": obj1,
            "relation": None,
            "obj2": None,
        }

    obj2 = {
        "size": tokens[4],
        "color": tokens[5],
        "shape": tokens[6],
    }

    return {
        "obj1": obj1,
        "relation": tokens[3],
        "obj2": obj2,
    }


# ---------------------------------------------------------
# Check one attribute
# ---------------------------------------------------------

def same_attribute(predicted, target, key):
    """
    Return True only when both dictionaries exist
    and the requested attribute is the same.
    """

    if predicted is None or target is None:
        return False

    return predicted.get(key) == target.get(key)


# ---------------------------------------------------------
# Evaluation
# ---------------------------------------------------------

@torch.no_grad()
def evaluate_model(
    model,
    dataset,
    tokenizer,
    device,
    max_samples=None,
    blind=False,
):
    """
    Evaluate:
        1. exact-match accuracy
        2. size accuracy
        3. color accuracy
        4. shape accuracy
        5. relation accuracy
    """

    model.eval()

    total = len(dataset)

    if max_samples is not None:
        total = min(total, max_samples)

    exact_correct = 0

    size_correct = 0
    color_correct = 0
    shape_correct = 0

    size_total = 0
    color_total = 0
    shape_total = 0

    relation_correct = 0
    relation_total = 0

    unparsed_predictions = 0
    letter_correct = 0
    letter_total = 0
    for i in tqdm(range(total), desc="Evaluating"):
        image, _ = dataset[i]

        ground_truth_word = dataset.words[i]

        image = image.unsqueeze(0).to(device)
        if blind:
            image=torch.zeros_like(image)

        # Generate the model's word.
        prediction = generate_word(
            model,
            image,
            tokenizer,
            max_letters=45,
        )

        # Exact match
        if prediction == ground_truth_word:
            exact_correct += 1
        for j in range(len(ground_truth_word)):
            letter_total += 1

            if j < len(prediction) and prediction[j] == ground_truth_word[j]:
             letter_correct += 1
        # Parse both words.
        ground_truth = parse_ground_truth(ground_truth_word)
        predicted = parse_broken_word(prediction)

        if predicted is None:
            unparsed_predictions += 1

        # -------------------------
        # First object
        # -------------------------

        size_total += 1
        color_total += 1
        shape_total += 1

        if same_attribute(predicted.get("obj1") if predicted else None,
                          ground_truth["obj1"],
                          "size"):
            size_correct += 1

        if same_attribute(predicted.get("obj1") if predicted else None,
                          ground_truth["obj1"],
                          "color"):
            color_correct += 1

        if same_attribute(predicted.get("obj1") if predicted else None,
                          ground_truth["obj1"],
                          "shape"):
            shape_correct += 1

        # -------------------------
        # Second object
        # -------------------------

        if ground_truth["obj2"] is not None:
            size_total += 1
            color_total += 1
            shape_total += 1

            if same_attribute(
                predicted.get("obj2") if predicted else None,
                ground_truth["obj2"],
                "size",
            ):
                size_correct += 1

            if same_attribute(
                predicted.get("obj2") if predicted else None,
                ground_truth["obj2"],
                "color",
            ):
                color_correct += 1

            if same_attribute(
                predicted.get("obj2") if predicted else None,
                ground_truth["obj2"],
                "shape",
            ):
                shape_correct += 1

            # Relation only matters for 2-object scenes.
            relation_total += 1

            if (
                predicted is not None
                and predicted.get("relation") == ground_truth["relation"]
            ):
                relation_correct += 1

    results = {
        "exact_match": exact_correct / total if total > 0 else 0.0,
        "size_accuracy": size_correct / size_total if size_total > 0 else 0.0,
        "color_accuracy": color_correct / color_total if color_total > 0 else 0.0,
        "shape_accuracy": shape_correct / shape_total if shape_total > 0 else 0.0,
        "relation_accuracy": (
            relation_correct / relation_total
            if relation_total > 0
            else 0.0
        ),
        "unparsed_predictions": unparsed_predictions,
        "num_samples": total,
        "letter_accuracy": (
        letter_correct / letter_total
            if letter_total > 0
            else 0.0
                    ),
    }

    return results


# ---------------------------------------------------------
# Load checkpoint
# ---------------------------------------------------------

def load_model(checkpoint_path, device):
    """
    Create the same TinyVLM architecture used during training
    and load the saved weights.
    """

    model = TinyVLM(
        vocab_size=27,
        d_model=128,
        n_heads=4,
        n_layers=2,
    ).to(device)

    checkpoint = torch.load(
        checkpoint_path,
        map_location=device,
        weights_only=False,
    )

    # Our training script saves a dictionary containing "model".
    if isinstance(checkpoint, dict) and "model" in checkpoint:
        state_dict = checkpoint["model"]
    else:
        # Also support a checkpoint that is just a state_dict.
        state_dict = checkpoint

    model.load_state_dict(state_dict)

    return model


# ---------------------------------------------------------
# Main
# ---------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="Evaluate TinyVLM on ShapeScenes."
    )

    parser.add_argument(
        "--checkpoint",
        default="checkpoints/best.pt",
        help="Path to the model checkpoint.",
    )

    parser.add_argument(
        "--data-dir",
        default="data",
        help="Folder containing train.pt, val.pt, test.pt, test_heldout.pt.",
    )

    parser.add_argument(
        "--split",
        default="test",
        choices=["train", "val", "test", "test_heldout"],
        help="Dataset split to evaluate.",
    )

    parser.add_argument(
        "--max-samples",
        type=int,
        default=None,
        help="Evaluate only the first N samples.",
    )

    parser.add_argument(
        "--device",
        default=None,
        choices=["cpu", "cuda"],
        help="Device to use. Default: CUDA if available, otherwise CPU.",
    )
    parser.add_argument(
    "--blind",
    action="store_true",
    help="Replace images with zeros during evaluation.",
)
    args = parser.parse_args()

    # Choose device.
    if args.device is None:
        device = torch.device(
            "cuda" if torch.cuda.is_available() else "cpu"
        )
    else:
        device = torch.device(args.device)

    print(f"Device: {device}")
    print(f"Split: {args.split}")
    print(f"Checkpoint: {args.checkpoint}")

    # Tokenizer
    tokenizer = CharacterTokenizer()

    # Dataset
    data_path = f"{args.data_dir}/{args.split}.pt"

    dataset = ShapeScenesDataset(
        data_path,
        tokenizer,
    )

    print(f"Number of samples: {len(dataset)}")

    # Model
    model = load_model(
        args.checkpoint,
        device,
    )

    # Evaluate
    results = evaluate_model(
        model=model,
        dataset=dataset,
        tokenizer=tokenizer,
        device=device,
        max_samples=args.max_samples,
        blind=args.blind,
    )

    # Print results
    print()
    print("========== RESULTS ==========")

    print(
        f"Exact match:       "
        f"{results['exact_match'] * 100:.2f}%"
    )

    print(
        f"Size accuracy:     "
        f"{results['size_accuracy'] * 100:.2f}%"
    )

    print(
        f"Color accuracy:    "
        f"{results['color_accuracy'] * 100:.2f}%"
    )

    print(
        f"Shape accuracy:    "
        f"{results['shape_accuracy'] * 100:.2f}%"
    )
    print(
    f"Letter accuracy:   "
    f"{results['letter_accuracy'] * 100:.2f}%"
)
    print(
        f"Relation accuracy: "
        f"{results['relation_accuracy'] * 100:.2f}%"
    )

    print(
        f"Unparsed predictions: "
        f"{results['unparsed_predictions']} / "
        f"{results['num_samples']}"
    )


if __name__ == "__main__":
    main()