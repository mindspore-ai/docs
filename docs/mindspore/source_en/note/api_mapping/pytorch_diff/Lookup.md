# Function differences with torchtext.data.functional.numericalize_tokens_from_iterator

<a href="https://gitee.com/mindspore/docs/blob/r1.7/docs/mindspore/source_en/note/api_mapping/pytorch_diff/Lookup.md" target="_blank"><img src="https://mindspore-website.obs.cn-north-4.myhuaweicloud.com/website-images/r1.7/resource/_static/logo_source_en.png"></a>

## torchtext.data.functional.numericalize_tokens_from_iterator

```python
torchtext.data.functional.numericalize_tokens_from_iterator(
    vocab,
    iterator,
    removed_tokens=None
)
```

For more information, see [torchtext.data.functional.numericalize_tokens_from_iterator](https://pytorch.org/text/0.10.0/data_functional.html#numericalize-tokens-from-iterator).

## mindspore.dataset.text.transforms.Lookup

```python
class mindspore.dataset.text.transforms.Lookup(
    vocab,
    unknown_token=None,
    data_type=mstype.int32
)
```

For more information, see [mindspore.dataset.text.transforms.Lookup](https://mindspore.cn/docs/en/r1.7/api_python/dataset_text/mindspore.dataset.text.transforms.Lookup.html#mindspore.dataset.text.transforms.Lookup).

## Differences

PyTorch: Generate the id list corresponding to the vocabulary from the word segmentation iterator, input the mapping table corresponding to the vocabulary and the id, the vocabulary iterator, and return the created iterator object, from which the id of the corresponding vocabulary can be obtained.

MindSpore: Look up a word into an id according to the input vocabulary table.

## Code Example

```python
import mindspore.dataset as ds
from mindspore.dataset import text
import torch as T
from torchtext.data.functional import simple_space_split, numericalize_tokens_from_iterator

# In MindSpore, return id of given word with looking up the vocab.

Vocab_file_path = '/path/to/testVocab/vocab_list.txt'

vocab = text.Vocab.from_file(Vocab_file_path, ",", None, ["<pad>", "<unk>"], True)
lookup = text.Lookup(vocab)

text_file_dataset_dir = '/path/to/testVocab/words.txt'

text_file_dataset = ds.TextFileDataset(dataset_files=text_file_dataset_dir)
text_file_dataset = text_file_dataset.map(operations=lookup,  input_columns=["text"])
for d in text_file_dataset.create_dict_iterator(num_epochs=1, output_numpy=True):
   print(d["text"])
# Out:
# 14
# 12
# 13
# 10
# 15
# 11

# In torch, return the ids iterator with looking up the vocab.

vocab = {'Sentencepiece' : 0, 'encode' : 1, 'as' : 2, 'pieces' : 3}
ids_iter = numericalize_tokens_from_iterator(vocab, simple_space_split(["Sentencepiece as pieces", "as pieces"]))
for ids in ids_iter:
    print([num for num in ids])
# Out:
# [0, 2, 3]
# [2, 3]
```
