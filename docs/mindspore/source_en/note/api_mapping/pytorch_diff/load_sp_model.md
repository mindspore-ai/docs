# Differences with torchtext.data.functional.load_sp_model

[![View Source On Gitee](https://mindspore-website.obs.cn-north-4.myhuaweicloud.com/website-images/br_base/resource/_static/logo_source_en.svg)](https://gitee.com/mindspore/docs/blob/br_base/docs/mindspore/source_en/note/api_mapping/pytorch_diff/load_sp_model.md)

## torchtext.data.functional.load_sp_model

```python
torchtext.data.functional.load_sp_model(
    spm
)
```

For more information, see [torchtext.data.functional.load_sp_model](https://pytorch.org/text/0.9.0/data_functional.html#load-sp-model).

## mindspore.dataset.text.SentencePieceTokenizer

```python
class mindspore.dataset.text.SentencePieceTokenizer(mode, out_type)
```

For more information, see [mindspore.dataset.text.SentencePieceTokenizer](https://www.mindspore.cn/docs/en/br_base/api_python/dataset_text/mindspore.dataset.text.SentencePieceTokenizer.html#mindspore.dataset.text.SentencePieceTokenizer).

## Differences

PyTorch: Load a sentencepiece model.

MindSpore: Construct a SentencePiece tokenizer, including load a sentencepiece model.

| Categories | Subcategories |PyTorch | MindSpore | Difference |
| --- | ---   | ---   | ---        |---  |
|Parameter | Parameter1 | spm    | mode    | MindSpore support SentencePieceVocab object or path of SentencePiece model |
|     | Parameter2 | -    |out_type     | The output type of tokenizer  |

## Code Example

```python
from download import download

url = "https://mindspore-website.obs.cn-north-4.myhuaweicloud.com/notebook/datasets/sentencepiece.bpe.model"
download(url, './sentencepiece.bpe.model', replace=True)

# PyTorch
from torchtext.data.functional import load_sp_model
model = load_sp_model("sentencepiece.bpe.model")

# MindSpore
import mindspore.dataset.text as text
model = text.SentencePieceTokenizer("sentencepiece.bpe.model", out_type=text.SPieceTokenizerOutType.STRING)
```
