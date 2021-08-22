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
"""
Dataset for training transformers
"""
import numpy as np


class Iterator:
    """
      The dataset for generating the random inputs
    """
    def __init__(self, batch_size, vocab_size, src_len, tgt_len):
        self.index = 0
        self.len = batch_size
        np.random.seed(1)
        self.encoder_input_value = np.random.randint(low=0, high=vocab_size,
                                                     size=(batch_size, src_len)).astype(np.int32)
        self.encoder_input_mask = np.ones((batch_size, src_len, src_len)).astype(np.float16)

        np.random.seed(1)
        ids = np.random.randint(low=0, high=vocab_size, size=(batch_size, tgt_len + 1))
        self.decoder_input_value = ids[:, :-1].astype(np.int32)
        self.memory_mask = np.ones((batch_size, tgt_len, src_len)).astype(np.float16)
        self.label = ids[:, 1:].astype(np.int32)

    def __iter__(self):
        self.index = 0
        return self

    def __len__(self):
        return self.len

    def __next__(self):
        if self.index >= self.len:
            raise StopIteration

        item = (self.encoder_input_value[self.index], self.encoder_input_mask[self.index],
                self.decoder_input_value[self.index], self.memory_mask[self.index], self.label[self.index])
        self.index += 1
        return item

    def reset(self):
        self.index = 0
