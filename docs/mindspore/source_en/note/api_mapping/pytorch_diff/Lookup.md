# Differences with torchtext.data.functional.numericalize_tokens_from_iterator

[![View Source On Gitee](https://mindspore-website.obs.cn-north-4.myhuaweicloud.com/website-images/br_base/resource/_static/logo_source_en.svg)](https://gitee.com/mindspore/docs/blob/br_base/docs/mindspore/source_en/note/api_mapping/pytorch_diff/Lookup.md)

## torchtext.data.functional.numericalize_tokens_from_iterator

```python
torchtext.data.functional.numericalize_tokens_from_iterator(
    vocab,
    iterator,
    removed_tokens=None
)
```

For more information, see [torchtext.data.functional.numericalize_tokens_from_iterator](https://pytorch.org/text/0.9.0/data_functional.html#numericalize-tokens-from-iterator).

## mindspore.dataset.text.Lookup

```python
class mindspore.dataset.text.Lookup(
    vocab,
    unknown_token=None,
    data_type=mstype.int32
)
```

For more information, see [mindspore.dataset.text.Lookup](https://mindspore.cn/docs/en/br_base/api_python/dataset_text/mindspore.dataset.text.Lookup.html#mindspore.dataset.text.Lookup).

## Differences

PyTorch: Generate the id list corresponding to the vocabulary from the word segmentation iterator, input the mapping table corresponding to the vocabulary and the id, the vocabulary iterator, and return the created iterator object, from which the id of the corresponding vocabulary can be obtained.

MindSpore: Look up a word into an id according to the input vocabulary table.

| Categories | Subcategories |PyTorch | MindSpore | Difference |
| --- | ---   | ---   | ---        |---  |
|Parameter | Parameter1 | vocab     | vocab     | - |
|     | Parameter2 | iterator   |-     | Strings to be tokenized, see usage below in MindSpore |
|     | Parameter3 | removed_tokens    |-     | Removed tokens from output dataset, not support by MindSpore |
|     | Parameter4 | -   |unknown_token    | Word is used in case of the target word is out of vocabulary |
|     | Parameter5 | -   |data_type    | The data type output by lookup |

## Code Example

```python
# PyTorch
from torchtext.data.functional import numericalize_tokens_from_iterator

def gen():
    yield ["Sentencepiece", "as", "encode"]

vocab = {'Sentencepiece' : 0, 'encode' : 1, 'as' : 2, 'pieces' : 3}
ids_iter = numericalize_tokens_from_iterator(vocab, gen())
for ids in ids_iter:
    print([num for num in ids])
# Out: [0, 2, 1]


# MindSpore
import mindspore.dataset.text as text

vocab = text.Vocab.from_dict({'Sentencepiece' : 0, 'encode' : 1, 'as' : 2, 'pieces' : 3})
result = text.Lookup(vocab)(["Sentencepiece", "as", "encode"])
print(result)
# Out: [0 2 1]
```
