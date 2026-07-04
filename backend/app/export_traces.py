import json
from pathlib import Path

from .main import EXAMPLES, config_dict, get_model
from .tracing import trace_payload

TRACES_DIR = Path(__file__).resolve().parent.parent / "traces"


def main():
    model = get_model()
    cfg = config_dict(model)
    TRACES_DIR.mkdir(parents=True, exist_ok=True)
    for i, tokens in enumerate(EXAMPLES):
        max_steps = model.config.block_size - len(tokens)
        payload = trace_payload(model, cfg, tokens, max_steps)
        out_path = TRACES_DIR / f"example_{i}.json"
        with open(out_path, "w") as f:
            json.dump(payload, f)
        print(f"wrote {out_path}")


if __name__ == "__main__":
    main()
