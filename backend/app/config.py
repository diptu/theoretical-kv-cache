TRACE_FORMAT_VERSION = "1.0"

# Toy task: sort a fixed-length sequence of digits, mirroring the sorting
# demo model used by Karpathy's minGPT / bbycroft's llm-viz nano demo.
SORT_LENGTH = 6
NUM_DIGITS = 3
VOCAB_SYMBOLS = ["A", "B", "C"]


class GPTConfig:
    def __init__(self, vocab_size: int, block_size: int, n_layer: int, n_head: int, n_embd: int):
        self.vocab_size = vocab_size
        self.block_size = block_size
        self.n_layer = n_layer
        self.n_head = n_head
        self.n_embd = n_embd
