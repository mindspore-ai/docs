# Copyright 2021 Huawei Technologies Co., Ltd
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
"""servable config."""
import os
import sys
from importlib import import_module
import numpy as np
from mindspore_serving.server import register

sys.path.append(os.path.dirname(__file__) + os.sep + "../../")
poetry_dataset = import_module("src.poetry_dataset")

_, tokenizer, _ = poetry_dataset.create_tokenizer()
model = register.declare_model(model_file="poetry.mindir", model_format="MindIR")

def generate_random_poetry(s):
    """generate random poetry"""
    token_ids, segment_ids = tokenizer.encode(s)
    token_ids = token_ids[:-1]
    segment_ids = segment_ids[:-1]
    target_ids = []
    MAX_LEN = 64
    length = 128
    while len(token_ids) + len(target_ids) < MAX_LEN:
        _target_ids = token_ids + target_ids
        _segment_ids = segment_ids + [0 for _ in target_ids]
        index = len(_target_ids)
        _target_ids = poetry_dataset.padding(np.array(_target_ids), length=length)
        _segment_ids = poetry_dataset.padding(np.array(_segment_ids), length=length)
        pad_mask = (_target_ids != 0).astype(np.float32)
        _target_ids = _target_ids.astype(np.int32)
        _segment_ids = _segment_ids.astype(np.int32)

        _probas = model.call(_target_ids, _segment_ids, pad_mask)

        _probas = _probas[index-1, 3:]
        p_args = _probas.argsort()[::-1][:100]
        p = _probas[p_args]
        p = p / sum(p)
        target_index = np.random.choice(len(p), p=p)
        target = p_args[target_index] + 3
        target_ids.append(target)
        if target == 3:
            break
    poetry = tokenizer.decode(token_ids + target_ids)
    return poetry

def generate_hidden(head):
    """generate hidden"""
    token_ids, segment_ids = tokenizer.encode('')
    token_ids = token_ids[:-1]
    segment_ids = segment_ids[:-1]

    punctuations = ['，', '。']
    punctuation_ids = [tokenizer.token_to_id[token] for token in punctuations]
    poetry = []
    length = 128

    for ch in head:
        poetry.append(ch)
        token_id = tokenizer.token_to_id[ch]
        token_ids.append(token_id)
        segment_ids.append(0)
        while True:
            index = len(token_ids)
            _target_ids = poetry_dataset.padding(np.array(token_ids), length=length)
            _segment_ids = poetry_dataset.padding(np.array(segment_ids), length=length)
            pad_mask = (_target_ids != 0).astype(np.float32)
            _target_ids = _target_ids.astype(np.int32)
            _segment_ids = _segment_ids.astype(np.int32)

            _probas = model.call(_target_ids, _segment_ids, pad_mask)
            _probas = _probas[index - 1, 3:]
            p_args = _probas.argsort()[::-1][:100]
            p = _probas[p_args]
            p = p / sum(p)
            target_index = np.random.choice(len(p), p=p)
            target = p_args[target_index] + 3
            token_ids.append(target)
            segment_ids.append(0)
            if target > 3:
                poetry.append(tokenizer.id_to_token[target])
            if target in punctuation_ids:
                break
    return ''.join(poetry)

def generate(s, types):
    if types in [0, 1]:
        poetry = generate_random_poetry(s)
    else:
        poetry = generate_hidden(s)
    return poetry

@register.register_method(output_names=["poetry"])
def predict(inputs, types):
    poetry = register.add_stage(generate, inputs, types, outputs_count=1)
    return poetry
