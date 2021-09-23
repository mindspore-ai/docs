# 比较与torchtext.data.functional.numericalize_tokens_from_iterator的功能差异

<a href="https://gitee.com/mindspore/docs/blob/master/docs/mindspore/migration_guide/source_zh_cn/api_mapping/pytorch_diff/Lookup.md" target="_blank"><img src="https://gitee.com/mindspore/docs/raw/master/resource/_static/logo_source.png"></a>

## torchtext.data.functional.numericalize_tokens_from_iterator

```python
torchtext.data.functional.numericalize_tokens_from_iterator(
    vocab,
    iterator,
    removed_tokens=None
    )
```

更多内容详见[torchtext.data.functional.numericalize_tokens_from_iterator](https://pytorch.org/docs/1.5.0/data.html#torchtext.data.functional.numericalize_tokens_from_iterator)。

## mindspore.dataset.text.transforms.Lookup

```python
class mindspore.dataset.text.transforms.Lookup(
    vocab,
    unknown_token=None,
    data_type=mstype.int32
    )
```

更多内容详见[mindspore.dataset.text.transforms.Lookup](https://mindspore.cn/docs/api/zh-CN/master/api_python/dataset_text/mindspore.dataset.text.transforms.Lookup.html#mindspore.dataset.text.transforms.Lookup)。

## 使用方式

PyTorch：从分词迭代器中生成词汇表对应的id列表，输入为词汇与id对应的映射表、词汇迭代器，返回创建好的迭代器对象，可从中获取对应词汇的id。

MindSpore：依据词汇与id的映射表，查找词汇对应的id。

## 代码示例

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
