# Differences with torchtext.data.functional.simple_space_split

<a href="https://gitee.com/mindspore/docs/blob/r2.1/docs/mindspore/source_en/note/api_mapping/pytorch_diff/WhitespaceTokenizer.md" target="_blank"><img src="https://mindspore-website.obs.cn-north-4.myhuaweicloud.com/website-images/r2.1/resource/_static/logo_source_en.png"></a>

## torchtext.data.functional.simple_space_split

```python
torchtext.data.functional.simple_space_split(
    spm
)
```

For more information, see [torchtext.data.functional.simple_space_split](https://pytorch.org/text/0.9.0/data_functional.html#load-sp-model).

## mindspore.dataset.text.WhitespaceTokenizer

```python
class mindspore.dataset.text.WhitespaceTokenizer(with_offsets=False)
```

For more information, see [mindspore.dataset.text.WhitespaceTokenizer](https://www.mindspore.cn/docs/en/r2.1/api_python/dataset_text/mindspore.dataset.text.WhitespaceTokenizer.html#mindspore.dataset.text.WhitespaceTokenizer).

## Differences

PyTorch: Tokenize a string on with whitespaces.

MindSpore: Tokenize a string on with whitespaces.

| Categories | Subcategories |PyTorch | MindSpore | Difference |
| --- | ---   | ---   | ---        |---  |
|Parameter | Parameter1 | -    | with_offsets     | Whether or not output offsets of tokens |

## Code Example

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
