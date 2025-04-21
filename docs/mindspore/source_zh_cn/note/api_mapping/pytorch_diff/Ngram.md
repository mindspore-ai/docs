# 比较与torchtext.data.utils.ngrams_iterator的差异

[![查看源文件](https://mindspore-website.obs.cn-north-4.myhuaweicloud.com/website-images/r2.6.0rc1/resource/_static/logo_source.svg)](https://gitee.com/mindspore/docs/blob/r2.6.0rc1/docs/mindspore/source_zh_cn/note/api_mapping/pytorch_diff/Ngram.md)

## torchtext.data.utils.ngrams_iterator

```python
torchtext.data.utils.ngrams_iterator(
    token_list,
    ngrams
)
```

更多内容详见[torchtext.data.utils.ngrams_iterator](https://pytorch.org/text/0.9.0/data_utils.html#ngrams-iterator)。

## mindspore.dataset.text.Ngram

```python
class mindspore.dataset.text.Ngram(
    n,
    left_pad=("", 0),
    right_pad=("", 0),
    separator=" "
)
```

更多内容详见[mindspore.dataset.text.Ngram](https://mindspore.cn/docs/zh-CN/r2.6.0rc1/api_python/dataset_text/mindspore.dataset.text.Ngram.html#mindspore.dataset.text.Ngram)。

## 使用方式

PyTorch：从1维的字符串生成N-gram

MindSpore：从1维的字符串生成N-gram，可以指定N-gram的连接符，或对序列进行额外的填充。

| 分类       | 子类         | PyTorch      | MindSpore      | 差异          |
| ---------- | ------------ | ------------ | ---------      | ------------- |
| 参数       | 参数 1       | token_list    | -         | 输入的分词列表，用法差别见代码示例 |
|            | 参数 2       | ngrams          | n          | n-gram的数量 |
|            | 参数 3       | -           | left_pad        | 指定序列的左侧填充 |
|            | 参数 4       | -     | right_pad          | 指定序列的右侧填充 |
|            | 参数 5       | -          | separator     | 指定用于将n-gram结果的连接符 |

## 代码示例

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
