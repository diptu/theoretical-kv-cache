import argparse
import json
from pathlib import Path

import torch
import torch.nn.functional as F

from .config import GPTConfig, NUM_DIGITS, SORT_LENGTH
from .dataset import SortDataset
from .model import GPT

CHECKPOINT_DIR = Path(__file__).resolve().parent.parent / "checkpoints"


@torch.no_grad()
def evaluate(model: GPT, dataset: SortDataset, batch_size: int = 128) -> float:
    model.eval()
    x, y = dataset.sample_batch(batch_size)
    logits, _, _ = model(x, collect_trace=False)
    preds = logits[:, dataset.length - 1 :].argmax(-1)
    targets = y[:, dataset.length - 1 :]
    acc = (preds == targets).float().mean().item()
    model.train()
    return acc


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--steps", type=int, default=3000)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--log-every", type=int, default=200)
    args = parser.parse_args()

    torch.manual_seed(0)

    dataset = SortDataset(length=SORT_LENGTH, num_digits=NUM_DIGITS)
    config = GPTConfig(
        vocab_size=dataset.get_vocab_size(),
        block_size=dataset.get_block_size(),
        n_layer=3,
        n_head=3,
        n_embd=48,
    )
    model = GPT(config)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr)

    for step in range(1, args.steps + 1):
        x, y = dataset.sample_batch(args.batch_size)
        logits, _, _ = model(x, collect_trace=False)
        loss = F.cross_entropy(logits.reshape(-1, logits.size(-1)), y.reshape(-1), ignore_index=-1)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if step % args.log_every == 0 or step == 1:
            acc = evaluate(model, dataset)
            print(f"step {step:5d} | loss {loss.item():.4f} | sort_accuracy {acc:.3f}")

    CHECKPOINT_DIR.mkdir(parents=True, exist_ok=True)
    torch.save(model.state_dict(), CHECKPOINT_DIR / "model.pt")
    with open(CHECKPOINT_DIR / "config.json", "w") as f:
        json.dump(vars(config), f, indent=2)
    print(f"Saved checkpoint to {CHECKPOINT_DIR}")


if __name__ == "__main__":
    main()
