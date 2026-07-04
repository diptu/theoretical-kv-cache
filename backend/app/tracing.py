import torch

from .config import TRACE_FORMAT_VERSION
from .model import GPT


def _to_list(t: torch.Tensor):
    return t.detach().cpu().tolist()


def _squeeze_batch(t: torch.Tensor):
    return t[0]  # batch=1 for tracing/inference


def _build_layer_trace(layer_trace: dict) -> dict:
    return {
        "ln1_out": _to_list(_squeeze_batch(layer_trace["ln1_out"])),
        "q": _to_list(_squeeze_batch(layer_trace["q"])),
        "k_new": _to_list(_squeeze_batch(layer_trace["k_new"])),
        "v_new": _to_list(_squeeze_batch(layer_trace["v_new"])),
        "attn_scores": _to_list(_squeeze_batch(layer_trace["attn_scores"])),
        "attn_pattern": _to_list(_squeeze_batch(layer_trace["attn_pattern"])),
        "attn_out": _to_list(_squeeze_batch(layer_trace["attn_out"])),
        "resid_after_attn": _to_list(_squeeze_batch(layer_trace["resid_after_attn"])),
        "mlp_hidden": _to_list(_squeeze_batch(layer_trace["mlp_hidden"])),
        "mlp_out": _to_list(_squeeze_batch(layer_trace["mlp_out"])),
        "resid_after_mlp": _to_list(_squeeze_batch(layer_trace["resid_after_mlp"])),
    }


@torch.no_grad()
def _run_prefill(model: GPT, prompt_ids: list[int]):
    idx = torch.tensor([prompt_ids], dtype=torch.long)
    logits, kvs, extra = model(idx, collect_trace=True)
    prefill = {
        "tokens": prompt_ids,
        "token_emb": _to_list(_squeeze_batch(extra["tok_emb"])),
        "pos_emb": _to_list(extra["pos_emb"][0]),
        "embed_sum": _to_list(_squeeze_batch(extra["embed_sum"])),
        "layers": [_build_layer_trace(lt) for lt in extra["layer_traces"]],
        "final_ln_out": _to_list(_squeeze_batch(extra["final_ln_out"])),
        "logits": _to_list(_squeeze_batch(logits)),
    }
    return prefill, kvs, logits


@torch.no_grad()
def _run_decode_steps(model: GPT, kvs, logits: torch.Tensor, cur_len: int, steps: int):
    decode_steps = []
    for step in range(steps):
        next_token = int(torch.argmax(logits[0, -1]).item())
        next_idx = torch.tensor([[next_token]], dtype=torch.long)
        logits, kvs, extra = model(next_idx, past_kvs=kvs, pos_offset=cur_len, collect_trace=True)
        cur_len += 1
        decode_steps.append(
            {
                "step": step,
                "new_token": next_token,
                "cache_size_after": cur_len,
                "layers_new_kv": [
                    {
                        "k_new": _to_list(_squeeze_batch(lt["k_new"])),
                        "v_new": _to_list(_squeeze_batch(lt["v_new"])),
                        "attn_pattern": _to_list(_squeeze_batch(lt["attn_pattern"])),
                    }
                    for lt in extra["layer_traces"]
                ],
                "logits": _to_list(_squeeze_batch(logits)),
            }
        )
    return decode_steps


def generate_with_trace(model: GPT, prompt_ids: list[int], steps: int):
    """Prefill on `prompt_ids`, then autoregressively decode `steps` more
    tokens, recording the KV cache entries newly added at each step (not the
    full cache each time) so the trace stays small and the frontend can
    reconstruct cache growth by accumulating `layers_new_kv` over steps.
    """
    model.eval()
    prefill, kvs, logits = _run_prefill(model, prompt_ids)
    decode_steps = _run_decode_steps(model, kvs, logits, cur_len=len(prompt_ids), steps=steps)
    return prefill, decode_steps


def trace_payload(model: GPT, config_dict: dict, prompt_ids: list[int], steps: int) -> dict:
    prefill, decode_steps = generate_with_trace(model, prompt_ids, steps)
    return {
        "version": TRACE_FORMAT_VERSION,
        "config": config_dict,
        "prefill": prefill,
        "decode_steps": decode_steps,
    }
