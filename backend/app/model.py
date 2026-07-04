import math

import torch
import torch.nn as nn
import torch.nn.functional as F

from .config import GPTConfig


class CausalSelfAttention(nn.Module):
    def __init__(self, config: GPTConfig):
        super().__init__()
        assert config.n_embd % config.n_head == 0
        self.n_head = config.n_head
        self.n_embd = config.n_embd
        self.head_dim = config.n_embd // config.n_head
        self.c_attn = nn.Linear(config.n_embd, 3 * config.n_embd)
        self.c_proj = nn.Linear(config.n_embd, config.n_embd)
        self.register_buffer(
            "causal_mask",
            torch.tril(torch.ones(config.block_size, config.block_size)).view(
                1, 1, config.block_size, config.block_size
            ),
        )

    def _split_heads(self, x: torch.Tensor, T: int) -> torch.Tensor:
        B = x.size(0)
        return x.view(B, T, self.n_head, self.head_dim).transpose(1, 2)  # B, nh, T, hd

    def forward(self, x: torch.Tensor, past_kv=None, trace: dict | None = None):
        B, T, C = x.shape
        qkv = self.c_attn(x)
        q, k_new, v_new = qkv.split(self.n_embd, dim=2)
        q = self._split_heads(q, T)
        k_new = self._split_heads(k_new, T)
        v_new = self._split_heads(v_new, T)

        if past_kv is not None:
            past_k, past_v = past_kv
            k = torch.cat([past_k, k_new], dim=2)
            v = torch.cat([past_v, v_new], dim=2)
        else:
            k, v = k_new, v_new

        Tk = k.size(2)
        att = (q @ k.transpose(-2, -1)) / math.sqrt(self.head_dim)  # B, nh, T, Tk
        if past_kv is None:
            # Prefill over T>1 positions: mask future tokens. When decoding one
            # new token against an existing cache, every cached position is
            # already causally valid so no mask is needed.
            mask = self.causal_mask[:, :, :T, :Tk]
            att = att.masked_fill(mask == 0, float("-inf"))
        att_pattern = F.softmax(att, dim=-1)
        y = att_pattern @ v  # B, nh, T, hd
        y = y.transpose(1, 2).contiguous().view(B, T, C)
        out = self.c_proj(y)

        if trace is not None:
            trace["q"] = q
            trace["k_new"] = k_new
            trace["v_new"] = v_new
            # masked positions are -inf, which isn't valid JSON; clamp for serialization
            trace["attn_scores"] = torch.nan_to_num(att, neginf=-1e9, posinf=1e9)
            trace["attn_pattern"] = att_pattern
            trace["attn_out"] = y

        return out, (k, v)


class MLP(nn.Module):
    def __init__(self, config: GPTConfig):
        super().__init__()
        self.c_fc = nn.Linear(config.n_embd, 4 * config.n_embd)
        self.c_proj = nn.Linear(4 * config.n_embd, config.n_embd)

    def forward(self, x: torch.Tensor, trace: dict | None = None):
        h = F.gelu(self.c_fc(x))
        out = self.c_proj(h)
        if trace is not None:
            trace["mlp_hidden"] = h
            trace["mlp_out"] = out
        return out


class Block(nn.Module):
    def __init__(self, config: GPTConfig):
        super().__init__()
        self.ln_1 = nn.LayerNorm(config.n_embd)
        self.attn = CausalSelfAttention(config)
        self.ln_2 = nn.LayerNorm(config.n_embd)
        self.mlp = MLP(config)

    def forward(self, x: torch.Tensor, past_kv=None, trace: dict | None = None):
        ln1_out = self.ln_1(x)
        attn_out, new_kv = self.attn(ln1_out, past_kv=past_kv, trace=trace)
        x = x + attn_out
        resid_after_attn = x
        ln2_out = self.ln_2(x)
        mlp_out = self.mlp(ln2_out, trace=trace)
        x = x + mlp_out
        if trace is not None:
            trace["ln1_out"] = ln1_out
            trace["resid_after_attn"] = resid_after_attn
            trace["resid_after_mlp"] = x
        return x, new_kv


class GPT(nn.Module):
    def __init__(self, config: GPTConfig):
        super().__init__()
        self.config = config
        self.wte = nn.Embedding(config.vocab_size, config.n_embd)
        self.wpe = nn.Embedding(config.block_size, config.n_embd)
        self.blocks = nn.ModuleList([Block(config) for _ in range(config.n_layer)])
        self.ln_f = nn.LayerNorm(config.n_embd)
        self.lm_head = nn.Linear(config.n_embd, config.vocab_size, bias=False)

    def forward(
        self,
        idx: torch.Tensor,
        past_kvs: list | None = None,
        pos_offset: int = 0,
        collect_trace: bool = False,
    ):
        B, T = idx.shape
        pos = torch.arange(pos_offset, pos_offset + T, device=idx.device)
        tok_emb = self.wte(idx)
        pos_emb = self.wpe(pos)[None, :, :].expand(B, -1, -1)
        embed_sum = tok_emb + pos_emb

        x = embed_sum
        new_kvs = []
        layer_traces = [] if collect_trace else None
        for i, block in enumerate(self.blocks):
            past_kv = past_kvs[i] if past_kvs is not None else None
            layer_trace = {} if collect_trace else None
            x, kv = block(x, past_kv=past_kv, trace=layer_trace)
            new_kvs.append(kv)
            if collect_trace:
                layer_traces.append(layer_trace)

        final_ln_out = self.ln_f(x)
        logits = self.lm_head(final_ln_out)

        extra = None
        if collect_trace:
            extra = {
                "tok_emb": tok_emb,
                "pos_emb": pos_emb,
                "embed_sum": embed_sum,
                "layer_traces": layer_traces,
                "final_ln_out": final_ln_out,
            }
        return logits, new_kvs, extra
