# 比较与torchtext.data.utils.ngrams_iterator的功能差异

## torchtext.data.utils.ngrams_iterator

```python
torchtext.data.utils.ngrams_iterator(
    token_list,
    ngrams
    )
```

更多内容详见[torchtext.data.utils.ngrams_iterator](https://pytorch.org/docs/1.5.0/data.html#torchtext.data.utils.ngrams_iterator)。

## mindspore.dataset.text.transforms.Ngram

```python
class mindspore.dataset.text.transforms.Ngram(
    n,
    left_pad=("", 0),
    right_pad=("", 0),
    separator=" "
    )
```

更多内容详见[mindspore.dataset.text.transforms.Ngram](https://mindspore.cn/docs/api/zh-CN/r1.3/api_python/dataset_text/mindspore.dataset.text.transforms.Ngram.html#mindspore.dataset.text.transforms.Ngram)。

## 使用方式

PyTorch：返回一个迭代器，该迭代器生成给定的标记和ngrams。

MindSpore：TensorOp从一维字符串张量生成n-gram。

## 代码示例

```python
from mindspore.dataset import text
from torchtext.data.utils import ngrams_iterator

# In MindSpore, output numpy.ndarray type n-gram.

ngram_op = text.Ngram(3, separator="-")
output = ngram_op(["WildRose Country", "Canada's Ocean Playground", "Land of Living Skies"])
print(output)
# Out:
# ["WildRose Country-Canada's Ocean Playground-Land of Living Skies"]

# In torch, return an iterator that yields the given tokens and their ngrams.
token_list = ['here', 'we', 'are']
print(list(ngrams_iterator(token_list, 2)))
# Out:
# ['here', 'we', 'are', 'here we', 'we are']
```
