"""Character-level tokenizer: 26 letters + <eos> (27 tokens).

Implements encode/decode between words and letter-index sequences, and
batches words of different lengths (padding + ignore_index for the loss).
"""
class CharacterTokenizer:
    def __init__(self):
        # 26 letters a-z plus <eos>
        self.chars = list("abcdefghijklmnopqrstuvwxyz")
        self.eos_token = "<eos>"
        self.vocab = self.chars + [self.eos_token]
        
        self.char_to_id = {c: i for i, c in enumerate(self.vocab)}
        self.id_to_char = {i: c for i, c in enumerate(self.vocab)}
        
        self.eos_id = self.char_to_id[self.eos_token]
        self.vocab_size = len(self.vocab)

    def encode(self, word: str) -> list[int]:
        """Converts a string word into a list of token IDs, appending <eos>."""
        return [self.char_to_id[c] for c in word] + [self.eos_id]

    def decode(self, ids: list[int]) -> str:
        """Converts token IDs back to a string, stopping at <eos>."""
        chars = []
        for i in ids:
            if i == self.eos_id:
                break
            chars.append(self.id_to_char[i])
        return "".join(chars)