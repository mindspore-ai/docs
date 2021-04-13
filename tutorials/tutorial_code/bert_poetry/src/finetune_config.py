# Copyright 2020 Huawei Technologies Co., Ltd
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

"""
config settings, will be used in finetune.py
"""

from easydict import EasyDict as edict
import mindspore.common.dtype as mstype
from .bert_model import BertConfig

bs = 16

cfg = edict({
    'dict_path': './vocab.txt',
    'disallowed_words': ['（', '）', '(', ')', '__', '《', '》', '【', '】', '[', ']'],
    'max_len': 64,
    'min_word_frequency': 8,
    'dataset_path': './poetry.txt',
    'batch_size': bs,
    'epoch_num': 20,
    'ckpt_prefix': 'poetry',
    'ckpt_dir': None,
    'pre_training_ckpt': './bert_converted.ckpt',
    'optimizer': 'AdamWeightDecayDynamicLR',
    'AdamWeightDecay': edict({
        'learning_rate': 3e-5,
        'end_learning_rate': 1e-10,
        'power': 1.0,
        'weight_decay': 1e-5,
        'eps': 1e-6,
    }),
    'Lamb': edict({
        'start_learning_rate': 2e-5,
        'end_learning_rate': 1e-7,
        'power': 1.0,
        'weight_decay': 0.01,
        'decay_filter': lambda x: False,
    }),
    'Momentum': edict({
        'learning_rate': 2e-5,
        'momentum': 0.9,
    }),
})

bert_net_cfg = BertConfig(
    batch_size=bs,
    seq_length=128,
    vocab_size=3191,
    hidden_size=768,
    num_hidden_layers=12,
    num_attention_heads=12,
    intermediate_size=3072,
    hidden_act="gelu",
    hidden_dropout_prob=0.1,
    attention_probs_dropout_prob=0.1,
    max_position_embeddings=512,
    type_vocab_size=2,
    initializer_range=0.02,
    use_relative_positions=False,
    input_mask_from_dataset=True,
    token_type_ids_from_dataset=True,
    dtype=mstype.float32,
    compute_type=mstype.float16,
)
