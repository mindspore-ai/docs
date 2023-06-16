# 比较与torchtext.data.functional.custom_replace的功能差异

<a href="https://gitee.com/mindspore/docs/blob/master/docs/mindspore/source_zh_cn/note/api_mapping/pytorch_diff/RegexReplace.md" target="_blank"><img src="https://mindspore-website.obs.cn-north-4.myhuaweicloud.com/website-images/master/resource/_static/logo_source.png"></a>

## torchtext.data.functional.custom_replace

```python
torchtext.data.functional.custom_replace(replace_pattern)
```

更多内容详见[torchtext.data.functional.custom_replace](https://pytorch.org/text/0.9.0/data_functional.html#load-sp-model)。

## mindspore.dataset.text.RegexReplace

```python
class mindspore.dataset.text.RegexReplace(pattern, replace, replace_all=True)
```

更多内容详见[mindspore.dataset.text.RegexReplace](https://www.mindspore.cn/docs/en/master/api_python/dataset_text/mindspore.dataset.text.RegexReplace.html#mindspore.dataset.text.RegexReplace)。

## 使用方式

PyTorch：根据正则表达式对字符串内容进行正则替换。

MindSpore：根据正则表达式对字符串内容进行正则替换。

| 分类 | 子类 |PyTorch | MindSpore | 差异 |
| --- | ---   | ---   | ---        |---  |
|参数 | 参数1 | replace_pattern   | -     | 以tuple形式表达的正则表达式的模式以及被替换的字符 |
|     | 参数2 | -    |pattern    | 正则表达式的模式 |
|     | 参数3 | - | replace   | 替换匹配元素的字符串 |
|     | 参数4 | - | replace_all    | 是否只替换第一个匹配的元素 |

## 代码示例

```python
list_a = ["Sentencepiece encode  aS  pieces"]

# PyTorch
from torchtext.data.functional import custom_replace

custom_replace_transform = custom_replace([(r'S', 's')])
print(list(custom_replace_transform(list_a)))
# Out: ['sentencepiece encode  as  pieces']

# MindSpore
import mindspore.dataset.text as text

transform = text.RegexReplace(pattern=r'S', replace='s')
print(transform(list_a))
# Out: ['sentencepiece encode  as  pieces']
```
