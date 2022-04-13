# Function differences with torchtext.data.utils.ngrams_iterator

<a href="https://gitee.com/mindspore/docs/blob/master/docs/mindspore/source_en/note/api_mapping/pytorch_diff/Ngram.md" target="_blank"><img src="https://mindspore-website.obs.cn-north-4.myhuaweicloud.com/website-images/master/resource/_static/logo_source_en.png"></a>

## torchtext.data.utils.ngrams_iterator

```python
torchtext.data.utils.ngrams_iterator(
    token_list,
    ngrams
)
```

For more information, see [torchtext.data.utils.ngrams_iterator](https://pytorch.org/text/0.10.0/data_utils.html#ngrams-iterator).

## mindspore.dataset.text.transforms.Ngram

```python
class mindspore.dataset.text.transforms.Ngram(
    n,
    left_pad=("", 0),
    right_pad=("", 0),
    separator=" "
)
```

For more information, see [mindspore.dataset.text.transforms.Ngram](https://mindspore.cn/docs/en/master/api_python/dataset_text/mindspore.dataset.text.transforms.Ngram.html#mindspore.dataset.text.transforms.Ngram).

## Differences

PyTorch: Returns an iterator that generates the given tokens and ngrams.

MindSpore: TensorOp generates n-grams from a one-dimensional string tensor.

## Code Example

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
