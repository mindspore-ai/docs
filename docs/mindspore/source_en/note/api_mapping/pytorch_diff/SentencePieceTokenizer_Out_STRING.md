# Function differences with torchtext.data.functional.sentencepiece_tokenizer

[![View Source On Gitee](https://mindspore-website.obs.cn-north-4.myhuaweicloud.com/website-images/r2.0/resource/_static/logo_source_en.png)](https://gitee.com/mindspore/docs/blob/r2.0/docs/mindspore/source_en/note/api_mapping/pytorch_diff/SentencePieceTokenizer_Out_INT.md)

## torchtext.data.functional.sentencepiece_tokenizer

```python
torchtext.data.functional.sentencepiece_tokenizer(
    sp_model
)
```

For more information, see [torchtext.data.functional.sentencepiece_tokenizer](https://pytorch.org/text/0.10.0/data_functional.html#sentencepiece-tokenizer).

## mindspore.dataset.text.SentencePieceTokenizer

```python
class mindspore.dataset.text.SentencePieceTokenizer(
    mode,
    out_type
)
```

For more information, see [mindspore.dataset.text.SentencePieceTokenizer](https://mindspore.cn/docs/en/r2.0/api_python/dataset_text/mindspore.dataset.text.SentencePieceTokenizer.html#mindspore.dataset.text.SentencePieceTokenizer).

## Differences

PyTorch: Returns a generator that converts text into string based on the input sentencepiece tokenizer model.

MindSpore: According to the incoming sentencepiece model, the input text is segmented and marked; the output type is string or int type.

## Code Example

```python
import mindspore.dataset as ds
from mindspore.dataset import text
from mindspore.dataset.text import SentencePieceModel, SPieceTokenizerOutType
from torchtext.data.functional import sentencepiece_tokenizer
from torchtext.data.functional import load_sp_model

# In MindSpore, Tokenize scalar token or 1-D tokens to tokens by sentencepiece.
sentence_piece_vocab_file = "/path/to/datasets/1.txt"

vocab = text.SentencePieceVocab.from_file([sentence_piece_vocab_file], 27, 0.9995,
                                          SentencePieceModel.CHAR, {})
tokenizer = text.SentencePieceTokenizer(vocab, out_type=SPieceTokenizerOutType.STRING)
data = 'sentencepiece encode as pieces", "examples to   try!'

print(list(tokenizer(data)))
# Out:
# ['▁', 's', 'e', 'n', 't', 'e', 'n', 'c', 'e', 'p', 'i', 'e', 'c', 'e', '▁', 'e', 'n', 'c', 'o', 'd', 'e', '▁', 'a', 's', '▁', 'p', 'i', 'e', 'c', 'e', 's', '"', ',', '▁', '"', 'e', 'x', 'a', 'm', 'p', 'l', 'e', 's', '▁', 't', 'o', '▁', 't', 'r', 'y', '!']

root = "/path/to/m_user.model"
sp_model = load_sp_model(root)
# In torch, output a generator with the input of text sentence and the output of the corresponding tokens based on SentencePiece model.
sp_tokens_generator = sentencepiece_tokenizer(sp_model)
list_a = ["sentencepiece encode as pieces", "examples to   try!"]
list(sp_tokens_generator(list_a))
```
