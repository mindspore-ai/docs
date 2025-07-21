# Copyright 2025 Huawei Technologies Co., Ltd
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ============================================================================
"""Begin Qwen2 large language model inference network development."""

import json
from glob import glob
from dataclasses import dataclass
from typing import Optional
from typing import Type
from typing import List
from typing import Tuple
from typing import Union

import math
from collections import deque
import numpy as np

from mindspore import Tensor
from mindspore import dtype
from mindspore import nn
from mindspore import ops
from mindspore import mint
from mindspore import Parameter
from mindspore import load_checkpoint
from mindspore import load_param_into_net
from mindspore import mutable
from mindspore import jit


@dataclass
class Qwen2Config:
    """Qwen2 Config, the key-value is almost the same with config.json in Hugging Face"""
    architectures: Optional[List[str]] = None
    attention_dropout: float = 0.0
    bos_token_id: int = 151643
    eos_token_id: int = 151645
    hidden_act: str = "silu"
    hidden_size: int = 3584
    initializer_range: float = 0.02
    intermediate_size: int = 18944
    max_position_embeddings: int = 32768
    max_window_layers: int = 28
    model_type: str = "qwen2"
    num_attention_heads: int = 28
    num_hidden_layers: int = 28
    num_key_value_heads: int = 4
    rms_norm_eps: float = 1e-06
    rope_theta: float = 1000000.0
    sliding_window: Optional[int] = 131072
    tie_word_embeddings: bool = False
    torch_dtype: str = "bfloat16"
    transformers_version: str = "4.41.2"
    use_cache: bool = True
    use_sliding_window: bool = False
    vocab_size: int = 152064
    param_dtype: Optional[Type] = dtype.bfloat16   # this is mindspore datatype as hugging face use str as dtype

    @classmethod
    def from_json(cls, json_path: str) -> 'Qwen2Config':
        """Get Qwen2Config from json file"""
        with open(json_path) as f:
            data = json.load(f)
        config = cls(**data)
        return config


@dataclass
class Qwen2ModelInput:
    """Qwen2 Model Input, the packed input struct for qwen2"""
    input_ids: Tensor
    positions: Tensor
    batch_valid_length: Tensor
    is_prefill: bool
    attn_mask: Tensor
    k_caches: List[Tensor]
    v_caches: List[Tensor]
    slot_mapping: Tensor = None
    block_tables: Tensor = None
    hidden_state: Optional[Tensor] = None
    residual: Optional[Tensor] = None
    q_seq_lens: Optional[Tensor] = None


class RmsNorm(nn.Cell):
    """Common rmsnorm layer"""
    def __init__(self, config: Qwen2Config) -> None:
        super().__init__()

        self.rms_norm = ops.RmsNorm(config.rms_norm_eps)

        self.weight = Parameter(
            mint.ones(
                config.hidden_size,
                dtype=config.param_dtype
            ),
            requires_grad=False
        )

    def construct(self, x: Tensor, residual: Optional[Tensor] = None) -> Union[Tensor, Tuple[Tensor, Tensor]]:
        """layer compute"""
        if residual is not None:
            x = x + residual
            residual = x
        output = self.rms_norm(x, self.weight)[0]
        if residual is None:
            return output
        return output, residual


class Qwen2Linear(nn.Cell):
    """Qwen2 linear layer"""
    def __init__(self, input_size: int, output_size: int, param_dtype: Optional[Type], enable_bias: bool) -> None:
        super().__init__()

        self.param_dtype = param_dtype
        self.input_size = input_size
        self.output_size = output_size
        self.enable_bias = enable_bias

        self.matmul = ops.MatMul(transpose_b=True)
        self.weight = Parameter(
            mint.zeros(
                (self.output_size, self.input_size),
                dtype=self.param_dtype
            ),
            requires_grad=False
        )

        if self.enable_bias:
            self.bias_add = ops.Add()
            self.bias = Parameter(
                mint.zeros(self.output_size, dtype=self.param_dtype)
            )

    def construct(self, input: Tensor):
        """layer compute"""
        origin_shape = input.shape
        x = self.matmul(input.view(-1, origin_shape[-1]), self.weight)
        if self.enable_bias:
            x = self.bias_add(x, self.bias)
        return x.view(*origin_shape[:-1], -1)


class VocabEmbedding(nn.Cell):
    """Common vocab embedding layer"""
    def __init__(self, config: Qwen2Config) -> None:
        super().__init__()

        self.num_embeddings = config.vocab_size
        self.embedding_dim = config.hidden_size

        self.gather = ops.Gather()

        self.weight = Parameter(
            mint.zeros(
                (self.num_embeddings, self.embedding_dim),
                dtype=config.param_dtype
            ),
            requires_grad=False
        )

    def construct(self, input_ids: Tensor):
        """layer compute"""
        return self.gather(self.weight, input_ids, 0)


class Qwen2RotaryEmbedding(nn.Cell):
    """Qwen2 rotary embedding layer"""
    def __init__(self, head_size: int, rotary_dim: int,
                 max_position_embeddings: int, base: int,
                 dtype: Optional[Type]) -> None:
        super().__init__()

        self.head_size = head_size
        self.rotary_dim = rotary_dim
        self.max_position_embeddings = max_position_embeddings
        self.base = base
        self.dtype = dtype

        # format 2 is neox style
        self.rotary_embedding_op = ops.ApplyRotaryPosEmb(2)
        self.gather = ops.Gather()

        self.freqs_cos, self.freqs_sin = self._compute_cos_sin_cache()

    def _compute_inv_freq(self) -> Tensor:
        """compute inv freq for rope"""
        freqs_base = mint.arange(0, self.rotary_dim, 2).astype(np.float32)
        freqs = 1.0 / (self.base ** (freqs_base / self.rotary_dim))
        return freqs

    def _compute_cos_sin_cache(self) -> Tuple[Tensor, Tensor]:
        """compute cos sin for rope"""
        freqs = self._compute_inv_freq()
        t = np.arange(0, self.max_position_embeddings, 1).astype(np.float32)
        freqs = np.outer(t, freqs)
        emb = np.concatenate((freqs, freqs), axis=1)
        freqs_cos = np.cos(emb)
        freqs_sin = np.sin(emb)

        freqs_cos = Tensor(freqs_cos, dtype=self.dtype)
        freqs_sin = Tensor(freqs_sin, dtype=self.dtype)
        return freqs_cos, freqs_sin

    def construct(self, positions: Tensor, query: Tensor, key: Tensor, batch_valid_length: Tensor, is_prefill: bool):
        """layer compute"""
        query = query.contiguous()
        key = key.contiguous()

        if is_prefill:
            freqs_cos = self.freqs_cos
            freqs_sin = self.freqs_sin
        else:
            freqs_cos = self.gather(self.freqs_cos, positions.view(-1), 0)
            freqs_sin = self.gather(self.freqs_sin, positions.view(-1), 0)

        return self.rotary_embedding_op(query, key, freqs_cos, freqs_sin, batch_valid_length)


class FlashAttention(nn.Cell):
    """Common flash attention layer"""
    def __init__(self, scale: float, num_heads: int) -> None:
        super().__init__()

        input_layout = "TH"
        scale = scale
        pre_tokens = 2147483647
        next_tokens = 2147483647
        self.flash_attention = \
                ops.operations.nn_ops.FlashAttentionScore(head_num=num_heads,
                                                            scale_value=scale,
                                                            pre_tokens=pre_tokens,
                                                            next_tokens=next_tokens,
                                                            input_layout=input_layout)

    def construct(self, q: Tensor, k: Tensor, v: Tensor, attn_mask: Tensor, batch_valid_length: Tensor) -> Tensor:
        """layer compute"""
        _, _, _, output = self.flash_attention(
            q,
            k,
            v,
            None,
            None,
            None,
            attn_mask,
            None,
            batch_valid_length,
            batch_valid_length
        )
        return output


class PagedAttention(nn.Cell):
    """Common paged attention layer"""
    def __init__(self, head_num: int, scale: float, num_kv_heads: int) -> None:
        super().__init__()

        self.head_num = head_num
        self.num_kv_heads = num_kv_heads

        self.paged_attention = ops.auto_generate.PagedAttention(
            head_num=head_num,
            scale_value=scale,
            kv_head_num=num_kv_heads
        )

    def construct(self, q: Tensor, k_cache: Tensor, v_cache: Tensor,
                        block_tables: Tensor, batch_valid_length: Tensor,
                        attn_mask: Tensor, q_seq_lens: Tensor) -> Tensor:
        """layer compute"""
        output = self.paged_attention(q, k_cache, v_cache, block_tables,
                                        batch_valid_length, None, None,
                                        attn_mask, q_seq_lens)
        return output


class Qwen2Attention(nn.Cell):
    """Qwen2 attention layer"""
    def __init__(self, config: Qwen2Config) -> None:
        super().__init__()

        self.hidden_size = config.hidden_size
        self.num_heads = config.num_attention_heads
        self.num_kv_heads = config.num_key_value_heads
        self.head_dim =config.hidden_size // self.num_heads
        self.q_size = self.head_dim * self.num_heads
        self.kv_size = self.head_dim * self.num_kv_heads
        self.scaling = float(self.head_dim ** -0.5)
        self.rope_theta = int(config.rope_theta)
        self.param_dtype = config.param_dtype
        self.max_position = config.max_position_embeddings

        self.flash_attn = FlashAttention(self.scaling, self.num_heads)
        self.paged_attn = PagedAttention(self.num_heads, self.scaling, self.num_kv_heads)
        self.reshape_and_cache = ops.auto_generate.ReshapeAndCache()

        self.q_proj = Qwen2Linear(
            input_size=self.hidden_size,
            output_size=self.q_size,
            param_dtype=self.param_dtype,
            enable_bias=True
        )
        self.k_proj = Qwen2Linear(
            input_size=self.hidden_size,
            output_size=self.kv_size,
            param_dtype=self.param_dtype,
            enable_bias=True
        )
        self.v_proj = Qwen2Linear(
            input_size=self.hidden_size,
            output_size=self.kv_size,
            param_dtype=self.param_dtype,
            enable_bias=True
        )
        self.o_proj = Qwen2Linear(
            input_size=self.q_size,
            output_size=self.hidden_size,
            param_dtype=self.param_dtype,
            enable_bias=False
        )

        self.rotary_emb = Qwen2RotaryEmbedding(
            head_size=self.head_dim,
            rotary_dim=self.head_dim,
            max_position_embeddings=self.max_position,
            base=self.rope_theta,
            dtype=self.param_dtype
        )

    def construct(self, hidden_state: Tensor, positions: Tensor, batch_valid_length: Tensor,
                        is_prefill: bool, layer_idx: int, k_cache: Tensor, v_cache: Tensor,
                        slot_mapping: Tensor, block_tables: Tensor, attn_mask: Tensor,
                        q_seq_lens: Tensor) -> Tensor:
        """layer compute"""
        bs, seq_len, hidden_dim = hidden_state.shape

        q = self.q_proj(hidden_state).view(-1, self.q_size)
        k = self.k_proj(hidden_state).view(-1, self.kv_size)
        v = self.v_proj(hidden_state).view(-1, self.kv_size)

        q, k = self.rotary_emb(
            positions,
            q,
            k,
            batch_valid_length,
            is_prefill
        )

        k = k.contiguous()
        v = v.contiguous()

        cache_out = self.reshape_and_cache(
            k,
            v,
            k_cache,
            v_cache,
            slot_mapping
        )
        q = ops.depend(q, cache_out)

        if is_prefill:
            attn_output = self.flash_attn(
                q,
                k,
                v,
                attn_mask,
                batch_valid_length
            )
        else:
            attn_output = self.paged_attn(
                q,
                k_cache,
                v_cache,
                block_tables,
                batch_valid_length,
                attn_mask,
                q_seq_lens
            )

        output = self.o_proj(attn_output).view(bs, seq_len, -1)
        return output


class Qwen2MLP(nn.Cell):
    """Qwen2 mlp layer"""
    def __init__(self, config: Qwen2Config) -> None:
        super().__init__()

        self.up_proj = Qwen2Linear(
            input_size=config.hidden_size,
            output_size=config.intermediate_size,
            param_dtype=config.param_dtype,
            enable_bias=False
        )
        self.gate_proj = Qwen2Linear(
            input_size=config.hidden_size,
            output_size=config.intermediate_size,
            param_dtype=config.param_dtype,
            enable_bias=False
        )
        self.down_proj = Qwen2Linear(
            input_size=config.intermediate_size,
            output_size=config.hidden_size,
            param_dtype=config.param_dtype,
            enable_bias=False
        )
        self.act_fn = ops.silu

    def construct(self, x: Tensor) -> Tensor:
        """layer compute"""
        output = self.down_proj(self.act_fn(self.gate_proj(x)) * self.up_proj(x))
        return output


class Qwen2DecoderLayer(nn.Cell):
    """Qwen2 decoder layer"""
    def __init__(self, config: Qwen2Config) -> None:
        super().__init__()

        self.hidden_size = config.hidden_size

        self.self_attn = Qwen2Attention(config=config)
        self.mlp = Qwen2MLP(config=config)
        self.input_layernorm = RmsNorm(config=config)
        self.post_attention_layernorm = RmsNorm(config=config)

    def construct(self, hidden_state: Tensor, residual: Tensor, positions: Tensor,
                        batch_valid_length: Tensor, is_prefill: bool, layer_idx: int,
                        k_cache: Tensor, v_cache: Tensor, slot_mapping: Tensor,
                        block_tables: Tensor, attn_mask: Tensor, q_seq_lens: Tensor) -> Tuple[Tensor, Tensor]:
        """layer compute"""
        if residual is None:
            residual = hidden_state
            hidden_state = self.input_layernorm(hidden_state)
        else:
            hidden_state, residual = self.input_layernorm(hidden_state, residual)

        hidden_state = self.self_attn(hidden_state, positions, batch_valid_length, is_prefill,
                                        layer_idx, k_cache, v_cache, slot_mapping, block_tables,
                                        attn_mask, q_seq_lens)
        hidden_state, residual = self.post_attention_layernorm(hidden_state, residual)
        hidden_state = self.mlp(hidden_state)

        return hidden_state, residual


class Qwen2Model(nn.Cell):
    """Qwen2 model"""
    def __init__(self, config: Qwen2Config) -> None:
        super().__init__()

        self.vocab_size = config.vocab_size
        self.hidden_size = config.hidden_size
        self.num_hidden_layers = config.num_hidden_layers

        self.embed_tokens = VocabEmbedding(config=config)
        self.layers = nn.CellList()
        for i in range(config.num_hidden_layers):
            layer = Qwen2DecoderLayer(config=config)
            self.layers.append(layer)
        self.norm = RmsNorm(config=config)

    @jit(jit_level="O0", infer_boost="on")
    def construct(self, input_ids: Tensor, positions: Tensor, batch_valid_length: Tensor,
                        is_prefill: bool, k_caches: List[Tensor], v_caches: List[Tensor],
                        slot_mapping: Tensor, block_tables: Tensor, attn_mask: Tensor,
                        q_seq_lens: Tensor) -> Tensor:
        """layer compute"""
        hidden_state = self.embed_tokens(input_ids)
        residual = None

        for i in range(self.num_hidden_layers):
            layer = self.layers[i]
            hidden_state, residual = layer(hidden_state, residual, positions, batch_valid_length,
                                           is_prefill, i, k_caches[i], v_caches[i], slot_mapping,
                                           block_tables, attn_mask, q_seq_lens)

        hidden_state, _ = self.norm(hidden_state, residual)

        return hidden_state


class Qwen2ForCausalLM(nn.Cell):
    """Qwen2 causal model"""
    def __init__(self, config: Qwen2Config) -> None:
        super().__init__()

        self.model = Qwen2Model(config=config)
        self.lm_head = Qwen2Linear(
            input_size=config.hidden_size,
            output_size=config.vocab_size,
            param_dtype=config.param_dtype,
            enable_bias=False
        )

    def load_weight(self, weight_path: str) -> None:
        """load model weight from hugging face"""
        weight_dict = {}
        for path in glob(weight_path + "/*.safetensors"):
            weight_dict.update(load_checkpoint(path, format="safetensors"))

        load_param_into_net(self, weight_dict, strict_load=False)

    def construct(self, model_input: Qwen2ModelInput) -> Tensor:
        """layer compute"""
        hidden_state = self.model(model_input.input_ids, model_input.positions,
                                  model_input.batch_valid_length, model_input.is_prefill,
                                  model_input.k_caches, model_input.v_caches, model_input.slot_mapping,
                                  model_input.block_tables, model_input.attn_mask, model_input.q_seq_lens)
        logits = self.lm_head(hidden_state)[:, -1]
        return logits


class CacheManager:
    """KVCache Manager"""
    def __init__(self, config: Qwen2Config, block_num: int, block_size: int, batch_size: int) -> None:
        self.block_num = block_num
        self.block_size = block_size
        self.batch_size = batch_size

        head_dim = config.hidden_size // config.num_attention_heads

        self.k_caches = mutable([ops.zeros((block_num, block_size,
                                            config.num_key_value_heads, head_dim),
                                            dtype=config.param_dtype)
                                    for _ in range(config.num_hidden_layers)])
        self.v_caches = mutable([ops.zeros((block_num, block_size,
                                            config.num_key_value_heads, head_dim),
                                            dtype=config.param_dtype)
                                    for _ in range(config.num_hidden_layers)])
        self.block_tables = [[] for _ in range(batch_size)]
        self.acc_slot_mapping = [[] for _ in range(batch_size)]
        self.free_block_ids = deque(range(block_num))

    def step(self, start_pos_idx: int, token_num_per_batch: int) -> Tuple[Tensor, Tensor]:
        """step compute model inputs"""
        for i in range(self.batch_size):
            block_table = self.block_tables[i]
            total_block_num = math.ceil((start_pos_idx + token_num_per_batch) / self.block_size)
            now_block_num = len(block_table)
            for _ in range(total_block_num - now_block_num):
                block_id = self.free_block_ids.popleft()
                block_table.append(block_id)
                start_slot_id = block_id * self.block_size
                self.acc_slot_mapping[i].extend(list(range(start_slot_id, start_slot_id + self.block_size)))


        now_block_tables = Tensor(self.block_tables, dtype=dtype.int32)
        now_slot_mapping = Tensor([self.acc_slot_mapping[i][start_pos_idx: start_pos_idx + token_num_per_batch]
                                for i in range(self.batch_size)], dtype=dtype.int32).view(-1)

        return now_block_tables, now_slot_mapping


def sample(logits: Tensor) -> Tensor:
    """argmax sample function"""
    next_token = logits.argmax(axis=-1, keepdims=True)
    return next_token
