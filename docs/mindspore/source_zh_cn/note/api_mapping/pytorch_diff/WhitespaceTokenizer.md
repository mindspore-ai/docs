# 比较与torchtext.data.functional.simple_space_split的差异

[![查看源文件](https://mindspore-website.obs.cn-north-4.myhuaweicloud.com/website-images/br_base/resource/_static/logo_source.svg)](https://gitee.com/mindspore/docs/blob/br_base/docs/mindspore/source_zh_cn/note/api_mapping/pytorch_diff/WhitespaceTokenizer.md)

## torchtext.data.functional.simple_space_split

```python
torchtext.data.functional.simple_space_split(iterator)
```

更多内容详见[torchtext.data.functional.simple_space_split](https://pytorch.org/text/0.9.0/data_functional.html#torchtext.data.functional.simple_space_split)。

## mindspore.dataset.text.WhitespaceTokenizer

```python
class mindspore.dataset.text.WhitespaceTokenizer(with_offsets=False)
```

更多内容详见[mindspore.dataset.text.WhitespaceTokenizer](https://www.mindspore.cn/docs/zh-CN/br_base/api_python/dataset_text/mindspore.dataset.text.WhitespaceTokenizer.html#mindspore.dataset.text.WhitespaceTokenizer)。

## 使用方式

PyTorch：基于空白字符对输入的字符串进行分词。

MindSpore：基于空白字符对输入的字符串进行分词。

| 分类 | 子类 |PyTorch | MindSpore | 差异 |
| --- | ---   | ---   | ---        |---  |
|参数 | 参数1 | -    | with_offsets     | 是否输出token的偏移量 |

## 代码示例

```python
# PyTorch
from torchtext.data.functional import simple_space_split

list_a = "sentencepiece encode as pieces"
result = simple_space_split([list_a])
print(list(result))
# Out: [['sentencepiece', 'encode', 'as', 'pieces']]

# MindSpore
import mindspore.dataset.text as text

result = text.WhitespaceTokenizer()(list_a)
print(list(result))
# Out: ['sentencepiece', 'encode', 'as', 'pieces']
```
