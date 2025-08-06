# Differences with torchtext.data.functional.sentencepiece_tokenizer

[![View Source On Gitee](https://mindspore-website.obs.cn-north-4.myhuaweicloud.com/website-images/br_base/resource/_static/logo_source_en.svg)](https://gitee.com/mindspore/docs/blob/br_base/docs/mindspore/source_en/note/api_mapping/pytorch_diff/SentencePieceTokenizer_Out_INT.md)

## torchtext.data.functional.sentencepiece_tokenizer

```python
torchtext.data.functional.sentencepiece_tokenizer(
    sp_model
)
```

For more information, see [torchtext.data.functional.sentencepiece_tokenizer](https://pytorch.org/text/0.9.0/data_functional.html#sentencepiece-tokenizer).

## mindspore.dataset.text.SentencePieceTokenizer

```python
class mindspore.dataset.text.SentencePieceTokenizer(
    mode,
    out_type
)
```

For more information, see [mindspore.dataset.text.SentencePieceTokenizer](https://mindspore.cn/docs/en/br_base/api_python/dataset_text/mindspore.dataset.text.SentencePieceTokenizer.html#mindspore.dataset.text.SentencePieceTokenizer).

## Differences

PyTorch: Returns a generator that converts text into string based on the input sentencepiece tokenizer model.

MindSpore: According to the incoming sentencepiece model, the input text is segmented and marked; the output type is string or int type.

| Categories | Subcategories |PyTorch | MindSpore | Difference |
| --- | ---   | ---   | ---        |---  |
|Parameter | Parameter1 | sp_model    | mode    | MindSpore support SentencePieceVocab object or path of  SentencePiece model |
|     | Parameter2 | -    |out_type     | The output type of tokenizer  |

## Code Example

```python
from download import download

url = "https://mindspore-website.obs.cn-north-4.myhuaweicloud.com/notebook/datasets/sentencepiece.bpe.model"
download(url, './sentencepiece.bpe.model', replace=True)

# PyTorch
from torchtext.data.functional import load_sp_model, sentencepiece_tokenizer

list_a = "sentencepiece encode as pieces"
model = load_sp_model("./sentencepiece.bpe.model")
sp_id_generator = sentencepiece_tokenizer(model)
print(list(sp_id_generator([list_a])))
# Out: [['▁sentence', 'piece', '▁en', 'code', '▁as', '▁pieces']]

# MindSpore
import mindspore.dataset.text as text

sp_id_generator = text.SentencePieceTokenizer("./sentencepiece.bpe.model", out_type=text.SPieceTokenizerOutType.STRING)
print(list(sp_id_generator(list_a)))
# Out: ['▁sentence', 'piece', '▁en', 'code', '▁as', '▁pieces']
```
