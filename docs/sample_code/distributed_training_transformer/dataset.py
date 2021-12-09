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
from collections import Counter
import numpy as np
import mindspore.dataset as ds
import mindspore.dataset.transforms.c_transforms as C
from mindspore import dtype as mstype
class Tokenzier:
    """Space tokenizer"""
    def __init__(self, vocab=None):
        self.field_vocab = dict(src=dict(), tgt=dict())

    def tokenize(self, sentence):
        return sentence.split()

    def convert_token_to_ids(self, sentence, key):
        vocab = self.field_vocab[key]
        return [vocab.get(item, 0) for item in sentence]

    def build_vocab(self, data, key):
        """Build the vocabuary"""
        vocab = Counter()
        for line in data:
            if line.strip():
                words = line.split()
                vocab.update(words)

        k, _ = zip(*vocab.most_common())
        self.special_vocab = ['UNK', 'PAD', 'EOT']
        vocab = {k: (i+len(self.special_vocab)) for i, k in enumerate(k)}
        for i, w in enumerate(self.special_vocab):
            vocab[w] = i
        self.field_vocab[key] = vocab

class ToyDataset():
    """An example of translating datasets"""
    def __init__(self, file_path, seq_length, tokenizer):
        self.file_path = file_path
        self.tokenizer = tokenizer
        self.seq_length = seq_length

    def _read_data(self):
        """Read the source data from the disk"""
        source_seq = []
        target_seq = []
        with open(self.file_path, 'r') as fp:
            for line in fp:
                line = line.strip()
                if line:
                    src, tgt = line.split('\t')
                    source_seq.append(src)
                    target_seq.append(tgt)

        return (source_seq, target_seq)

    def get_dataset(self, batch_size):
        """Return the dataset according to the batch_size"""
        src_sentence, target_sentence = self._read_data()
        self.tokenizer.build_vocab(src_sentence, key='src')
        self.tokenizer.build_vocab(target_sentence, key='tgt')
        print('Source vocab size:{}'.format(len(self.tokenizer.field_vocab['src'])))
        print('Target vocab size:{}'.format(len(self.tokenizer.field_vocab['tgt'])))

        type_cast_op = C.TypeCast(mstype.int32)
        type_cast_op_float = C.TypeCast(mstype.float16)

        def generator_seq2seq():
            src_seq = self.seq_length[0]
            tgt_seq = self.seq_length[1]
            for src, tgt in zip(src_sentence, target_sentence):
                src_words = self.tokenizer.tokenize(src)
                tgt_words = self.tokenizer.tokenize(tgt)
                if len(src_words) > src_seq + 1:
                    src_ids = self.tokenizer.convert_token_to_ids(src_words, key='src')
                    src_ids = np.array(src_ids[:src_seq])
                    src_position = np.arange(0, self.seq_length[0])
                    src_attention_mask = np.ones((self.seq_length[0], self.seq_length[0]))

                    if len(tgt_words) < tgt_seq:
                        tgt_words += ['PAD'] * (tgt_seq - len(tgt_words))
                    else:
                        tgt_words = tgt_words[:tgt_seq]

                    tgt_ids = self.tokenizer.convert_token_to_ids(tgt_words, key='tgt')
                    tgt_position = np.arange(0, tgt_seq).astype(np.int32)
                    memory_mask = np.ones((tgt_seq, src_seq)).astype(np.float16)

                    label = tgt_ids[1:] + self.tokenizer.convert_token_to_ids(["EOT"], key="tgt")
                    tgt_ids = np.array(tgt_ids).astype(np.int32)
                    label = np.array(label).astype(np.int32)

                    yield (src_ids,
                           src_position,
                           src_attention_mask,
                           tgt_ids,
                           tgt_position,
                           memory_mask,
                           label)


        ts = ds.GeneratorDataset(generator_seq2seq, column_names=['src_ids',
                                                                  'src_position',
                                                                  'attention_mask',
                                                                  'tgt_ids',
                                                                  'tgt_position',
                                                                  'memory_mask',
                                                                  'label'])
        ts = ts.map(input_columns="src_ids", operations=type_cast_op)
        ts = ts.map(input_columns="src_position", operations=type_cast_op)
        ts = ts.map(input_columns="attention_mask", operations=type_cast_op_float)
        ts = ts.batch(batch_size, drop_remainder=True)

        return ts
