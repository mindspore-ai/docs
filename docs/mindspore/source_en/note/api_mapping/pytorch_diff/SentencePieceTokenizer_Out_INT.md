# Differences with torchtext.data.functional.sentencepiece_numericalizer

<a href="https://gitee.com/mindspore/docs/blob/master/docs/mindspore/source_en/note/api_mapping/pytorch_diff/SentencePieceTokenizer_Out_INT.md" target="_blank"><img src="https://mindspore-website.obs.cn-north-4.myhuaweicloud.com/website-images/master/resource/_static/logo_source_en.png"></a>

## torchtext.data.functional.sentencepiece_numericalizer

```python
torchtext.data.functional.sentencepiece_numericalizer(
    sp_model
)
```

For more information, see [torchtext.data.functional.sentencepiece_numericalizer](https://pytorch.org/text/0.9.0/data_functional.html#sentencepiece-numericalizer).

## mindspore.dataset.text.SentencePieceTokenizer

```python
class mindspore.dataset.text.SentencePieceTokenizer(
    mode,
    out_type
)
```

For more information, see [mindspore.dataset.text.SentencePieceTokenizer](https://mindspore.cn/docs/en/master/api_python/dataset_text/mindspore.dataset.text.SentencePieceTokenizer.html#mindspore.dataset.text.SentencePieceTokenizer).

## Differences

PyTorch: A sentencepiece model to numericalize a text sentence into a generator according to the ids.

MindSpore: According to the incoming sentencepiece model, the input text is segmented and marked; the output type is string or int type.

| Categories | Subcategories |PyTorch | MindSpore | Difference |
| --- | ---   | ---   | ---        |---  |
|Parameter | Parameter 1 | sp_model    | mode    | MindSpore support SentencePieceVocab object or path of SentencePiece model |
|     | Parameter 2 | -    |out_type     | The output type of tokenizer  |

## Code Example

```python
from download import download

url = "https://mindspore-website.obs.cn-north-4.myhuaweicloud.com/notebook/datasets/sentencepiece.bpe.model"
download(url, './sentencepiece.bpe.model', replace=True)

# PyTorch
from torchtext.data.functional import load_sp_model, sentencepiece_numericalizer

list_a = "sentencepiece encode as pieces"
model = load_sp_model("./sentencepiece.bpe.model")
sp_id_generator = sentencepiece_numericalizer(model)
print(list(sp_id_generator([list_a])))
# Out: [[149356, 152666, 21, 40898, 236, 126370]]

# MindSpore
import mindspore.dataset.text as text

sp_id_generator = text.SentencePieceTokenizer("./sentencepiece.bpe.model", out_type=text.SPieceTokenizerOutType.INT)
print(list(sp_id_generator(list_a)))
# Out: [149356, 152666, 21, 40898, 236, 126370]
```
