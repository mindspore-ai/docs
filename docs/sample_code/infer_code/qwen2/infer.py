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
"""Begin Qwen2 large language model inference flow development."""

import os
from typing import List

import mindspore as ms

from mindspore import Tensor
from mindspore import mint
from mindspore import ops
from mindspore import dtype

from transformers import AutoTokenizer

from qwen2 import Qwen2Config
from qwen2 import Qwen2ForCausalLM
from qwen2 import Qwen2ModelInput
from qwen2 import CacheManager
from qwen2 import sample

# set mindspore context and envs
os.environ["MS_INTERNAL_DISABLE_CUSTOM_KERNEL_LIST"] = "PagedAttention"

ms.set_context(infer_boost="on")
ms.set_context(mode=ms.context.PYNATIVE_MODE)

model_path = "/path/to/model"
input_str = ["I love Beijing, because", "Hello, Qwen2"]
batch_size = len(input_str)
max_new_tokens = 64
block_size = 128
max_seq_lens = block_size * 10
block_num = (max_seq_lens * batch_size) // block_size

config = Qwen2Config.from_json(model_path + "/config.json")

model = Qwen2ForCausalLM(config)
# load weight
model.load_weight(model_path)

cache_manager = CacheManager(config, block_num, block_size, batch_size)

tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)

input_str = ["I love Beijing, because", "Hello, Qwen2"]

input_ids = tokenizer(input_str)["input_ids"]

print(input_ids)

def generate(model: Qwen2ForCausalLM, cache_manager: CacheManager, input_ids: List, max_new_tokens: int, max_seq_lens: int, eos_token_id: int):
    batch_size = len(input_ids)
    assert max_seq_lens >= max(map(len, input_ids))

    cur = min(map(len, input_ids))
    is_prefill = True
    it = 0

    decode_q_seq_lens = Tensor([1 for _ in range(batch_size)], dtype=dtype.int32)
    decode_mask = ops.zeros((1, 1), dtype=config.param_dtype)
    attn_mask = None
    q_seq_lens = None

    while cur <= max_seq_lens and it < max_new_tokens:
        batch_valid_length = Tensor([cur for _ in range(batch_size)], dtype=dtype.int32)
        if is_prefill:
            inp = Tensor([input_ids[i][:cur] for i in range(batch_size)], dtype=dtype.int32)
            pos = mint.arange(cur).astype(dtype.int32)
            block_tables, slot_mapping = cache_manager.step(0, cur)
            attn_mask = ops.logical_not(ops.sequence_mask(pos + 1, cur)).astype(config.param_dtype)
            q_seq_lens = None
        else:
            inp = Tensor([[input_ids[i][cur - 1]] for i in range(batch_size)], dtype=dtype.int32)
            pos = Tensor([[cur - 1] for _ in range(batch_size)], dtype=dtype.int32).view(-1)
            block_tables, slot_mapping = cache_manager.step(cur - 1, 1)
            attn_mask = decode_mask
            q_seq_lens = decode_q_seq_lens

        model_input = Qwen2ModelInput(
            input_ids=inp,
            positions=pos,
            batch_valid_length=batch_valid_length,
            is_prefill=is_prefill,
            attn_mask=attn_mask,
            k_caches=cache_manager.k_caches,
            v_caches=cache_manager.v_caches,
            block_tables=block_tables,
            slot_mapping=slot_mapping,
            q_seq_lens=q_seq_lens
        )

        logits = model(model_input)

        next_tokens = sample(logits)

        for i in range(batch_size):
            if cur >= len(input_ids[i]):
                input_ids[i].append(int(next_tokens[i]))

        cur += 1
        it += 1
        if is_prefill:
            is_prefill = False

    for i in range(batch_size):
        if eos_token_id in input_ids[i]:
            eos_idx = input_ids[i].index(eos_token_id)
            input_ids[i] = input_ids[i][: eos_idx + 1]

    return input_ids

output = generate(
    model=model,
    cache_manager=cache_manager,
    input_ids=input_ids,
    max_new_tokens=max_new_tokens,
    eos_token_id=tokenizer.eos_token_id,
    max_seq_lens=max_seq_lens
)

result = [tokenizer.decode(a) for a in output]
print(result)
