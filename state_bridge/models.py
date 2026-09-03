"""Loading frozen sender / receiver models and extracting sender hidden states."""

from __future__ import annotations

from dataclasses import dataclass

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

DTYPES = {"bfloat16": torch.bfloat16, "float16": torch.float16, "float32": torch.float32}


@dataclass
class LoadedModel:
    name: str
    model: torch.nn.Module
    tokenizer: object
    device: torch.device
    dtype: torch.dtype
    hidden_size: int
    num_layers: int

    def chat_prompt(self, user: str, assistant_prefix: str | None = None) -> str:
        """Render a single-turn chat prompt for this model (thinking disabled).

        ``assistant_prefix`` continues the assistant turn with already-written
        text (used by the text hand-off baseline)."""
        messages = [{"role": "user", "content": user}]
        try:
            text = self.tokenizer.apply_chat_template(
                messages, tokenize=False, add_generation_prompt=True, enable_thinking=False
            )
        except TypeError:  # template without enable_thinking support
            text = self.tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        if assistant_prefix:
            text = text + assistant_prefix
        return text

    def embed(self, ids: torch.Tensor) -> torch.Tensor:
        return self.model.get_input_embeddings()(ids.to(self.device))

    @property
    def embedding_rms(self) -> float:
        w = self.model.get_input_embeddings().weight
        return w.detach().float().pow(2).mean(dim=-1).sqrt().mean().item()

    @property
    def num_params(self) -> int:
        return sum(p.numel() for p in self.model.parameters())

    @property
    def non_embedding_params(self) -> int:
        emb = self.model.get_input_embeddings().weight.numel()
        out = 0
        head = getattr(self.model, "lm_head", None)
        if head is not None and head.weight is not self.model.get_input_embeddings().weight:
            out = head.weight.numel()
        return self.num_params - emb - out


def _text_config(config):
    return getattr(config, "text_config", None) or config


def load_model(path: str, device: str = "cuda:0", dtype: str = "bfloat16", name: str | None = None) -> LoadedModel:
    """Load a causal LM, freeze it, and put it in eval mode."""
    torch_dtype = DTYPES[dtype]
    tokenizer = AutoTokenizer.from_pretrained(path)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    # load straight onto the target device (avoids a transient copy on the default GPU)
    model = AutoModelForCausalLM.from_pretrained(path, dtype=torch_dtype, device_map=device)
    model.to(device)
    model.eval()
    model.requires_grad_(False)
    tc = _text_config(model.config)
    return LoadedModel(
        name=name or path,
        model=model,
        tokenizer=tokenizer,
        device=torch.device(device),
        dtype=torch_dtype,
        hidden_size=tc.hidden_size,
        num_layers=tc.num_hidden_layers,
    )


def tokenize_batch(lm: LoadedModel, texts: list[str], padding_side: str, max_length: int | None = None):
    tok = lm.tokenizer
    old = tok.padding_side
    tok.padding_side = padding_side
    try:
        enc = tok(
            texts,
            return_tensors="pt",
            padding=True,
            truncation=max_length is not None,
            max_length=max_length,
            add_special_tokens=False,
        )
    finally:
        tok.padding_side = old
    return enc


class SenderEncoder:
    """Runs the frozen sender's prefill and returns the hidden states that cross the bridge.

    The sender never generates in the basic setting: it reads the prompt, its
    hidden states at ``layers`` are concatenated on the feature axis, and it stops.
    Right padding is used so that positions of real tokens are exact.
    """

    def __init__(self, lm: LoadedModel, layers: list[int], max_prompt_tokens: int = 512):
        self.lm = lm
        self.layers = [l if l >= 0 else lm.num_layers + 1 + l for l in layers]
        for l in self.layers:
            if not 0 <= l <= lm.num_layers:
                raise ValueError(f"sender layer {l} out of range for {lm.num_layers} layers")
        self.max_prompt_tokens = max_prompt_tokens

    @property
    def out_dim(self) -> int:
        return self.lm.hidden_size * len(self.layers)

    @torch.no_grad()
    def encode_ids(self, input_ids: torch.Tensor, attention_mask: torch.Tensor):
        out = self.lm.model(
            input_ids=input_ids.to(self.lm.device),
            attention_mask=attention_mask.to(self.lm.device),
            output_hidden_states=True,
            use_cache=False,
        )
        hs = torch.cat([out.hidden_states[l] for l in self.layers], dim=-1)
        return hs, attention_mask.to(self.lm.device)

    @torch.no_grad()
    def encode(self, texts: list[str]):
        """``texts`` are fully rendered sender prompts.  Returns (hidden [B,T,D], mask [B,T])."""
        enc = tokenize_batch(self.lm, texts, padding_side="right", max_length=self.max_prompt_tokens)
        return self.encode_ids(enc["input_ids"], enc["attention_mask"])

    @torch.no_grad()
    def all_layer_pooled(self, texts: list[str]) -> torch.Tensor:
        """Mean-pooled hidden state of every layer: [num_layers+1, B, d]. Used by geometry analysis."""
        enc = tokenize_batch(self.lm, texts, padding_side="right", max_length=self.max_prompt_tokens)
        out = self.lm.model(
            input_ids=enc["input_ids"].to(self.lm.device),
            attention_mask=enc["attention_mask"].to(self.lm.device),
            output_hidden_states=True,
            use_cache=False,
        )
        m = enc["attention_mask"].to(self.lm.device).unsqueeze(-1).float()
        pooled = [(h.float() * m).sum(1) / m.sum(1).clamp(min=1) for h in out.hidden_states]
        return torch.stack(pooled, 0)
