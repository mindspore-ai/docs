# 比较与torchtext.data.functional.sentencepiece_tokenizer的功能差异

[![查看源文件](https://mindspore-website.obs.cn-north-4.myhuaweicloud.com/website-images/r1.8/resource/_static/logo_source.png)](https://gitee.com/mindspore/docs/blob/r1.8/docs/mindspore/source_zh_cn/note/api_mapping/pytorch_diff/SentencePieceTokenizer_Out_STRING.md)

## torchtext.data.functional.sentencepiece_tokenizer

```python
torchtext.data.functional.sentencepiece_tokenizer(
    sp_model
)
```

更多内容详见[torchtext.data.functional.sentencepiece_tokenizer](https://pytorch.org/text/0.10.0/data_functional.html#sentencepiece-tokenizer)。

## mindspore.dataset.text.SentencePieceTokenizer

```python
class mindspore.dataset.text.SentencePieceTokenizer(
    mode,
    out_type
)
```

更多内容详见[mindspore.dataset.text.SentencePieceTokenizer](https://mindspore.cn/docs/zh-CN/r1.8/api_python/dataset_text/mindspore.dataset.text.SentencePieceTokenizer.html#mindspore.dataset.text.SentencePieceTokenizer)。

## 使用方式

PyTorch：依据传入的分词模型，返回将文本转换为字符串的生成器。

MindSpore：依据传入的分词模型，对输入的文本进行分词及标记；输出类型是string或int类型。

## 代码示例

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
