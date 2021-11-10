# Function differences with torchtext.data.functional.sentencepiece_tokenizer

<a href="https://gitee.com/mindspore/docs/blob/r1.5/docs/mindspore/migration_guide/source_en/api_mapping/pytorch_diff/SentencePieceTokenizer_Out_INT.md" target="_blank"><img src="https://gitee.com/mindspore/docs/raw/r1.5/resource/_static/logo_source_en.png"></a>

## torchtext.data.functional.sentencepiece_tokenizer

```python
torchtext.data.functional.sentencepiece_tokenizer(
    sp_model
)
```

For more information, see [torchtext.data.functional.sentencepiece_tokenizer](https://pytorch.org/text/0.10.0/data_functional.html#sentencepiece-tokenizer).

## mindspore.dataset.text.transforms.SentencePieceTokenizer

```python
class mindspore.dataset.text.transforms.SentencePieceTokenizer(
    mode,
    out_type
)
```

For more information, see [mindspore.dataset.text.transforms.SentencePieceTokenizer](https://mindspore.cn/docs/api/en/r1.5/api_python/dataset_text/mindspore.dataset.text.transforms.SentencePieceTokenizer.html#mindspore.dataset.text.transforms.SentencePieceTokenizer).

## Differences

PyTorch: Returns a generator that converts text into string based on the input sentencepiece tokenizer model.

MindSpore: According to the incoming sentencepiece model, the input text is segmented and marked; the output type is string or int type.

## Code Example

```python
import mindspore.dataset as ds
from mindspore.dataset import text
from mindspore.dataset.text import SentencePieceModel, SPieceTokenizerOutType
from torchtext.data.functional import sentencepiece_tokenizer

# In MindSpore, Tokenize scalar token or 1-D tokens to tokens by sentencepiece.
sentence_piece_vocab_file = "/path/to/datasets/1.txt"

vocab = text.SentencePieceVocab.from_file([sentence_piece_vocab_file], 5000, 0.9995,
                                          SentencePieceModel.CHAR, {})
tokenizer = text.SentencePieceTokenizer(vocab, out_type=SPieceTokenizerOutType.STRING)
data = 'sentencepiece encode as pieces", "examples to   try!'

print(list(tokenizer(data)))
# Out:
# ['▁', 's', 'e', 'n', 't', 'e', 'n', 'c', 'e', 'p', 'i', 'e', 'c', 'e', '▁', 'e', 'n', 'c', 'o', 'd', 'e', '▁', 'a', 's', '▁', 'p', 'i', 'e', 'c', 'e', 's', '"', ',', '▁', '"', 'e', 'x', 'a', 'm', 'p', 'l', 'e', 's', '▁', 't', 'o', '▁', 't', 'r', 'y', '!']

# In torch, output a generator with the input of text sentence and the output of the corresponding tokens based on SentencePiece model.
sp_tokens_generator = sentencepiece_tokenizer(sp_model)
list_a = ["sentencepiece encode as pieces", "examples to   try!"]
list(sp_tokens_generator(list_a))
```
