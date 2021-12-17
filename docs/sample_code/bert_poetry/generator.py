"""bert generator"""
import numpy as np
from mindspore import Tensor
from mindspore import dtype as mstype
from src.finetune_config import cfg as settings
from src.poetry_dataset import padding, create_tokenizer

_, tokenizer, _ = create_tokenizer()

def generate_random_poetry(model, s=''):
    """generate random poetry"""
    token_ids, segment_ids = tokenizer.encode(s)
    token_ids = token_ids[:-1]
    segment_ids = segment_ids[:-1]
    target_ids = []
    length = 128
    while len(token_ids) + len(target_ids) < settings.max_len:
        _target_ids = token_ids + target_ids
        _segment_ids = segment_ids + [0 for _ in target_ids]

        index = len(_target_ids)

        _target_ids = padding(np.array(_target_ids), length=length)
        _segment_ids = padding(np.array(_segment_ids), length=length)

        pad_mask = (_target_ids != 0).astype(np.float32)

        _target_ids = Tensor([_target_ids], mstype.int32)
        _segment_ids = Tensor([_segment_ids], mstype.int32)
        pad_mask = Tensor([pad_mask], mstype.float32)

        _probas = model(_target_ids, _segment_ids, pad_mask).asnumpy()
        _probas = _probas[0, index-1, 3:]
        p_args = _probas.argsort()[::-1][:100]
        p = _probas[p_args]
        p = p / sum(p)
        target_index = np.random.choice(len(p), p=p)
        target = p_args[target_index] + 3
        target_ids.append(target)
        if target == 3:
            break
    return s + tokenizer.decode(target_ids)

def generate_hidden(model, head=""):
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
        try:
            token_id = tokenizer.token_to_id[ch]
        except KeyError:
            return "'{}'不在词表中，请换一个字试试。".format(ch)
        token_ids.append(token_id)
        segment_ids.append(0)
        while True:
            index = len(token_ids)
            _target_ids = padding(np.array(token_ids), length=length)
            _segment_ids = padding(np.array(segment_ids), length=length)
            pad_mask = (_target_ids != 0).astype(np.float32)

            _target_ids = Tensor([_target_ids], mstype.int32)
            _segment_ids = Tensor([_segment_ids], mstype.int32)
            pad_mask = Tensor([pad_mask], mstype.float32)
            _probas = model(_target_ids, _segment_ids, pad_mask).asnumpy()

            _probas = _probas[0, index-1, 3:]
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
