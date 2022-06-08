"""
poetry dataset processing method
"""
from collections import defaultdict
import numpy as np
import mindspore.dataset as de
import mindspore.dataset.transforms as transforms
from mindspore import dtype as mstype
from .finetune_config import cfg
from .poetry_utils import Tokenizer

def load_vocab(vacab_path):
    token_to_id = {}
    with open(vacab_path) as f:
        lines = f.readlines()
    for i, line in enumerate(lines):
        token = line.strip()
        token_to_id[token] = i
    return token_to_id

def create_tokenizer(length=128):
    """tokenizer processing method"""
    dict_path = cfg.dict_path
    forbidden_words = cfg.disallowed_words
    max_len = cfg.max_len
    frequency_threshold = cfg.min_word_frequency

    poetry_src = []
    with open(cfg.dataset_path) as f:
        lines = f.readlines()

    for line in lines:
        if line.count(':') != 1:
            continue
        poem = line.split(':')
        if len(poem) > 2:
            continue
        poem = poem[1]
        forbidden_poem = [word in poem for word in forbidden_words]
        if sum(forbidden_poem) > 0 or len(poem) > max_len-2:
            continue
        poetry_src.append(poem)

    token_to_id = load_vocab(dict_path)
    _tokenizer = Tokenizer(token_to_id, do_lower_case=True)

    token_num_dict = defaultdict(int)
    for poem in poetry_src:
        for token in _tokenizer.tokenize(poem):
            token_num_dict[token] += 1


    kept_token = []
    for token, num in token_num_dict.items():
        if num < int(frequency_threshold):
            continue
        kept_token.append((token, num))

    kept_token = [token for token, _ in sorted(kept_token, key=lambda x: -x[1])]

    kept_token_id = []
    tokens_id_dict = {}
    for i, token in enumerate(['[PAD]', '[UNK]', '[CLS]', '[SEP]']):
        tokens_id_dict[token] = i
        kept_token_id.append(token_to_id[token])


    for i, token in enumerate(kept_token):
        if token in token_to_id and token not in tokens_id_dict:
            tokens_id_dict[token] = len(tokens_id_dict)
            kept_token_id.append(token_to_id[token])

    tokenizer = Tokenizer(tokens_id_dict, do_lower_case=True)

    return poetry_src, tokenizer, kept_token_id


def padding(input_data, length=None):
    input_data = np.array(input_data)
    padding_length = length - input_data.shape[-1]
    output = np.pad(input_data, ((0, padding_length)), 'constant', constant_values=0)
    return output

class PoetryDataGenerator():
    """Reconstructing the PoetryDataGenerator processing method"""
    def __init__(self, batch_size, poetry, tokenizer, length=128):
        self.data = poetry
        self.batch_size = batch_size
        self.tokenizer = tokenizer
        self.length = length

    def __getitem__(self, index):
        np.random.shuffle(self.data)
        current_data = self.data[index]

        token_ids, segment_ids = self.tokenizer.encode(current_data)
        batch_token_ids = padding(token_ids, length=self.length)
        batch_segment_ids = padding(segment_ids, length=self.length)
        pad_mask = (batch_token_ids != 0).astype(np.float32)
        return (batch_token_ids, batch_segment_ids, pad_mask)

    def __len__(self):
        return len(self.data)


def create_poetry_dataset(batch_size, poetry, tokenizer):
    """create poetry dataset method"""
    dt = PoetryDataGenerator(batch_size, poetry, tokenizer)
    ds = de.GeneratorDataset(dt, ["input_ids", "token_type_id", "pad_mask"])
    #ds.set_dataset_size(dt.__len__())
    int_type_cast_op = transforms.TypeCast(mstype.int32)
    float_type_cast_op = transforms.TypeCast(mstype.float32)
    ds = ds.map(input_columns="input_ids", operations=int_type_cast_op)
    ds = ds.map(input_columns="token_type_id", operations=int_type_cast_op)
    ds = ds.map(input_columns="pad_mask", operations=float_type_cast_op)
    ds = ds.batch(batch_size, drop_remainder=True)
    return ds
