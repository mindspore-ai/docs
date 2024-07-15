# Differences with torchtext.data.utils.ngrams_iterator

[![View Source On Gitee](https://mindspore-website.obs.cn-north-4.myhuaweicloud.com/website-images/br_base/resource/_static/logo_source_en.svg)](https://gitee.com/mindspore/docs/blob/br_base/docs/mindspore/source_en/note/api_mapping/pytorch_diff/Ngram.md)

## torchtext.data.utils.ngrams_iterator

```python
torchtext.data.utils.ngrams_iterator(
    token_list,
    ngrams
)
```

For more information, see [torchtext.data.utils.ngrams_iterator](https://pytorch.org/text/0.9.0/data_utils.html#ngrams-iterator).

## mindspore.dataset.text.Ngram

```python
class mindspore.dataset.text.Ngram(
    n,
    left_pad=("", 0),
    right_pad=("", 0),
    separator=" "
)
```

For more information, see [mindspore.dataset.text.Ngram](https://mindspore.cn/docs/en/br_base/api_python/dataset_text/mindspore.dataset.text.Ngram.html#mindspore.dataset.text.Ngram).

## Differences

PyTorch: Generate n-gram from a 1-D string Tensor.

MindSpore: Generate n-gram from a 1-D string Tensor, string padding and connecting character are supported.

| Categories | Subcategories | PyTorch      | MindSpore     | Differences   |
| ---------- | ------------- | ------------ | ---------     | ------------- |
| Parameters       | Parameters 1       | token_list    | -         | A list of tokens, uasge see code example below |
|            | Parameters 2       | ngrams          | n          | n-gram number |
|            | Parameters 3       | -           | left_pad        | Strings to be paded left side |
|            | Parameters 4       | -     | right_pad          | Strings to be paded right side |
|            | Parameters 5       | -          | separator     | Symbol used to join strings together |

## Code Example

```python
# In torch, return an iterator that yields the given tokens and their ngrams.
from torchtext.data.utils import ngrams_iterator

token_list = ['here', 'we', 'are']
print(list(ngrams_iterator(token_list, 2)))
# Out:
# ['here', 'we', 'are', 'here we', 'we are']

# In MindSpore, output numpy.ndarray type n-gram.
from mindspore.dataset import text

ngram_op = text.Ngram([2, 1], separator=" ")
token_list = ['here', 'we', 'are']
output = ngram_op(token_list)
print(output)
# Out:
# ['here we' 'we are' 'here' 'we' 'are']
```
