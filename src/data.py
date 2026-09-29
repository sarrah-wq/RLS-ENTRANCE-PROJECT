"""PyTorch Dataset and collate function for ShapeScenes.

Loads the image + word pairs written by generate_data.py and returns
(image tensor, token id tensor) pairs, batched with a collate function
that pads to the longest word in the batch.
"""
import torch
from torch.utils.data import Dataset
from src.tokenizer import CharacterTokenizer

class ShapeScenesDataset(Dataset):
    def __init__(self, pt_path: str, tokenizer: CharacterTokenizer):
        data = torch.load(pt_path, weights_only=True)
        self.images = data["images"]  # (N, 3, 64, 64) uint8
        self.words = data["words"]
        self.tokenizer = tokenizer

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        # Convert image to float32 and normalize to [0, 1]
        img = self.images[idx].to(torch.float32) / 255.0
        # Encode the word
        token_ids = torch.tensor(self.tokenizer.encode(self.words[idx]), dtype=torch.long)
        return img, token_ids

def collate_fn(batch, pad_id=-100):
    images, tokens = zip(*batch)
    images = torch.stack(images)
    # Pad sequences to the maximum length in this batch
    tokens_padded = torch.nn.utils.rnn.pad_sequence(
        tokens, batch_first=True, padding_value=pad_id
    )
    return images, tokens_padded