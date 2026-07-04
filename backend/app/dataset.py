import torch


class SortDataset:
    """Toy task: given `length` digits from {0..num_digits-1}, predict the
    sorted sequence. Mirrors the sorting demo used by Karpathy's minGPT and
    bbycroft's llm-viz nano demo model (there sorting letters A/B/C).

    A training example concatenates the unsorted input with its sorted
    version into one sequence of length 2*L. The model is trained to predict
    the next token everywhere, but loss is only counted on the sorted half
    (the input half is given, not something the model should be scored on).
    """

    def __init__(self, length: int = 6, num_digits: int = 3):
        self.length = length
        self.num_digits = num_digits

    def get_vocab_size(self) -> int:
        return self.num_digits

    def get_block_size(self) -> int:
        # sequence of length 2*L, trained as (x = seq[:-1], y = seq[1:])
        return self.length * 2 - 1

    def sample_batch(self, batch_size: int):
        inp = torch.randint(self.num_digits, size=(batch_size, self.length), dtype=torch.long)
        sol, _ = torch.sort(inp, dim=1)
        cat = torch.cat([inp, sol], dim=1)
        x = cat[:, :-1]
        y = cat[:, 1:].clone()
        y[:, : self.length - 1] = -1  # ignore_index: don't score the given input half
        return x, y
