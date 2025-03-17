"""simple attention"""

import time
from typing import Optional, Callable
import numpy as np

import mindspore
from mindspore import ops, nn, Tensor


class Config:
    attention_dropout = 0.0
    hidden_size = 4096
    num_attention_heads = 32
    num_key_value_heads = 8
    max_position_embeddings = 8192
    rope_theta = 500000.0
    attention_bias = False


class LlamaRotaryEmbedding(nn.Cell):
    """LlamaRotaryEmbedding"""
    def __init__(self, dim, max_position_embeddings=2048, base=10000, scaling_factor=1.0):
        super().__init__()
        self.scaling_factor = scaling_factor
        self.dim = dim
        self.max_position_embeddings = max_position_embeddings
        self.base = base
        inv_freq = 1.0 / (self.base ** (np.arange(0, self.dim, 2).astype(np.float32) / self.dim))
        self.inv_freq = Tensor(inv_freq, mindspore.float32)
        # For BC we register cos and sin cached
        self.max_seq_len_cached = max_position_embeddings

    def construct(self, x, position_ids):
        """construct function"""

        # x: [bs, num_attention_heads, seq_len, head_size]
        inv_freq_expanded = self.inv_freq[None, :, None].to(mindspore.float32).broadcast_to(
            (position_ids.shape[0], -1, 1))
        position_ids_expanded = position_ids[:, None, :].to(mindspore.float32)
        # Force float32 since bfloat16 loses precision on long contexts
        # See https://github.com/huggingface/transformers/pull/29285
        freqs = ops.matmul(inv_freq_expanded, position_ids_expanded).swapdims(1, 2)
        emb = ops.cat((freqs, freqs), axis=-1)
        cos = emb.cos()
        sin = emb.sin()
        cos, sin = cos.to(dtype=x.dtype), sin.to(dtype=x.dtype)
        cos, sin = ops.stop_gradient(cos), ops.stop_gradient(sin)
        return cos, sin


class LlamaAttention(nn.Cell):
    """LlamaAttention"""
    def __init__(self, config: Config):
        super().__init__()
        self.config = config
        self.attention_dropout = config.attention_dropout
        self.hidden_size = config.hidden_size
        self.num_heads = config.num_attention_heads
        self.head_dim = self.hidden_size // self.num_heads
        self.num_key_value_heads = config.num_key_value_heads
        self.num_key_value_groups = self.num_heads // self.num_key_value_heads
        self.max_position_embeddings = config.max_position_embeddings
        self.rope_theta = config.rope_theta
        self.is_causal = True

        if (self.head_dim * self.num_heads) != self.hidden_size:
            raise ValueError(
                f"hidden_size must be divisible by num_heads (got `hidden_size`: {self.hidden_size}"
                f" and `num_heads`: {self.num_heads})."
            )

        self.q_proj = nn.Dense(self.hidden_size, self.num_heads * self.head_dim, has_bias=config.attention_bias)
        self.k_proj = nn.Dense(
            self.hidden_size, self.num_key_value_heads * self.head_dim, has_bias=config.attention_bias
        )
        self.v_proj = nn.Dense(
            self.hidden_size, self.num_key_value_heads * self.head_dim, has_bias=config.attention_bias
        )
        self.o_proj = nn.Dense(self.hidden_size, self.hidden_size, has_bias=config.attention_bias)

        self.rotary_emb = LlamaRotaryEmbedding(
            self.head_dim,
            max_position_embeddings=self.max_position_embeddings,
            base=self.rope_theta,
        )

    def rotate_half(self, x):
        x1 = x[..., : x.shape[-1] // 2]
        x2 = x[..., x.shape[-1] // 2 :]
        return ops.cat((-x2, x1), axis=-1)


    def apply_rotary_pos_emb(self, q, k, cos, sin, position_ids=None, unsqueeze_dim=1):
        cos = cos.unsqueeze(unsqueeze_dim)
        sin = sin.unsqueeze(unsqueeze_dim)
        q_embed = (q * cos) + (self.rotate_half(q) * sin)
        k_embed = (k * cos) + (self.rotate_half(k) * sin)
        return q_embed, k_embed


    def repeat_kv(self, hidden_states: mindspore.Tensor, n_rep: int) -> mindspore.Tensor:
        batch, num_key_value_heads, slen, head_dim = hidden_states.shape
        if n_rep == 1:
            return hidden_states
        hidden_states = hidden_states[:, :, None, :, :].broadcast_to(
            (batch, num_key_value_heads, n_rep, slen, head_dim))
        return hidden_states.reshape(batch, num_key_value_heads * n_rep, slen, head_dim)


    def construct(
            self,
            hidden_states: mindspore.Tensor,
            attention_mask: Optional[mindspore.Tensor] = None,
            position_ids: Optional[mindspore.Tensor] = None,
            **kwargs,
    ):
        """construct function"""
        bsz, q_len, _ = hidden_states.shape

        query_states = self.q_proj(hidden_states)
        key_states = self.k_proj(hidden_states)
        value_states = self.v_proj(hidden_states)

        query_states = query_states.view(bsz, q_len, self.num_heads, self.head_dim).swapdims(1, 2)
        key_states = key_states.view(bsz, q_len, self.num_key_value_heads, self.head_dim).swapdims(1, 2)
        value_states = value_states.view(bsz, q_len, self.num_key_value_heads, self.head_dim).swapdims(1, 2)

        cos, sin = self.rotary_emb(value_states, position_ids)
        query_states, key_states = self.apply_rotary_pos_emb(query_states, key_states, cos, sin)

        key_states = self.repeat_kv(key_states, self.num_key_value_groups)
        value_states = self.repeat_kv(value_states, self.num_key_value_groups)

        attn_weights = ops.matmul(query_states, key_states.swapdims(2, 3)) / (self.head_dim**0.5)

        attn_weights = ops.cast(attn_weights, mindspore.float32)
        if attention_mask is not None:  # no matter the length, we just slice it
            causal_mask = attention_mask[:, :, :, : key_states.shape[-2]]
            attn_weights = attn_weights + ops.cast(causal_mask, attn_weights.dtype)

        # upcast attention to fp32
        attn_weights = ops.softmax(attn_weights, axis=-1, dtype=mindspore.float32).to(query_states.dtype)
        attn_weights = ops.dropout(attn_weights, p=self.attention_dropout, training=self.training)
        attn_output = ops.matmul(attn_weights, value_states)

        # assert attn_output.shape == (bsz, self.num_heads, q_len, self.head_dim)
        attn_output = attn_output.swapdims(1, 2)
        attn_output = attn_output.reshape(bsz, q_len, -1)
        attn_output = self.o_proj(attn_output)

        return attn_output



block = LlamaAttention(Config())
grad_fn = mindspore.value_and_grad(block, None, block.trainable_params(), has_aux=False)


def fp(*args, **kwargs):
    out = block(*args, **kwargs)
    return out


def fp_and_bp(*args, **kwargs):
    out, grads = grad_fn(*args, **kwargs)
    return out, grads


def run_func(f: Callable, des: str = "function"):
    """run func"""

    s_time = time.time()

    x = Tensor(np.random.randn(1, 512, 4096), mindspore.float32)
    position_ids = Tensor(np.arange(512).reshape(1, -1), mindspore.float32)
    out = f(x, None, position_ids)

    time_to_prepare = time.time() - s_time
    s_time = time.time()

    for i in range(1000):
        out = f(x*(i/1000), None, position_ids)

    time_to_run_thousand_times = time.time() - s_time

    s_out_shape = f"{out.shape}" if isinstance(out, Tensor) else f"{out[0].shape}, grad[0] shape is: {out[1][0].shape}"
    print(f"{des}, output shape is: {s_out_shape}, time to prepare: {time_to_prepare:.2f}s, "
          f"time to run thousand times: {time_to_run_thousand_times:.2f}s")


# run fp
run_func(fp, des="origin block fp")
run_func(mindspore.jit(fp), des="jitted block by default fp")
run_func(mindspore.jit(fp, jit_level="O1"), des="jitted block by O1 fp")

# run fp+bp
run_func(fp_and_bp, des="origin block fp+bp")
run_func(mindspore.jit(fp_and_bp), des="jitted block by default fp+bp")
run_func(mindspore.jit(fp_and_bp, jit_level="O1"), des="jitted block by O1 fp+bp")
