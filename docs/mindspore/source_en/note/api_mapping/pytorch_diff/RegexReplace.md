# Differences with torchtext.data.functional.custom_replace

[![View Source On Gitee](https://mindspore-website.obs.cn-north-4.myhuaweicloud.com/website-images/r2.1/resource/_static/logo_source_en.png)](https://gitee.com/mindspore/docs/blob/r2.1/docs/mindspore/source_en/note/api_mapping/pytorch_diff/RegexReplace.md)

## torchtext.data.functional.custom_replace

```python
torchtext.data.functional.custom_replace(replace_pattern)
```

For more information, see [torchtext.data.functional.custom_replace](https://pytorch.org/text/0.9.0/data_functional.html#load-sp-model).

## mindspore.dataset.text.RegexReplace

```python
class mindspore.dataset.text.RegexReplace(pattern, replace, replace_all=True)
```

For more information, see [mindspore.dataset.text.RegexReplace](https://www.mindspore.cn/docs/en/r2.1/api_python/dataset_text/mindspore.dataset.text.RegexReplace.html#mindspore.dataset.text.RegexReplace).

## Differences

PyTorch: Replace a part of string with given text according to regular expressions.

MindSpore: Replace a part of string with given text according to regular expressions.

| Categories | Subcategories |PyTorch | MindSpore | Difference |
| --- | ---   | ---   | ---        |---  |
|Parameter | Parameter1 | replace_pattern   | -     | Regex expression and string to replace stored in tuple |
|     | Parameter2 | -    |pattern    | Regex expression patterns |
|     | Parameter3 | - | replace   | String to replace matched element. |
|     | Parameter4 | - | replace_all    | If only replace first matched element |

## Code Example

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
