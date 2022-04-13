# 比较与torchtext.data.functional.sentencepiece_numericalizer的功能差异

<a href="https://gitee.com/mindspore/docs/blob/master/docs/mindspore/source_zh_cn/note/api_mapping/pytorch_diff/SentencePieceTokenizer_Out_INT.md" target="_blank"><img src="https://mindspore-website.obs.cn-north-4.myhuaweicloud.com/website-images/master/resource/_static/logo_source.png"></a>

## torchtext.data.functional.sentencepiece_numericalizer

```python
torchtext.data.functional.sentencepiece_numericalizer(
    sp_model
)
```

更多内容详见[torchtext.data.functional.sentencepiece_numericalizer](https://pytorch.org/text/0.10.0/data_functional.html#sentencepiece-numericalizer)。

## mindspore.dataset.text.transforms.SentencePieceTokenizer

```python
class mindspore.dataset.text.transforms.SentencePieceTokenizer(
    mode,
    out_type
)
```

更多内容详见[mindspore.dataset.text.transforms.SentencePieceTokenizer](https://mindspore.cn/docs/zh-CN/master/api_python/dataset_text/mindspore.dataset.text.transforms.SentencePieceTokenizer.html#mindspore.dataset.text.transforms.SentencePieceTokenizer)。

## 使用方式

PyTorch：依据传入的分词模型，返回将文本转换为id的生成器。

MindSpore：依据传入的分词模型，对输入的文本进行分词及标记；输出类型是string或int类型。

## 代码示例

```python
import mindspore.dataset as ds
from mindspore.dataset import text
from mindspore.dataset.text import SentencePieceModel, SPieceTokenizerOutType
from torchtext.data.functional import sentencepiece_numericalizer

# In MindSpore, return tokenizer from vocab object.
sentence_piece_vocab_file = "/path/to/datasets/1.txt"

vocab = text.SentencePieceVocab.from_file(
    [sentence_piece_vocab_file],
    5000,
    0.9995,
    SentencePieceModel.UNIGRAM,
    {})
tokenizer = text.SentencePieceTokenizer(vocab, out_type=SPieceTokenizerOutType.INT)
text_file_dataset_dir = "/path/to/datasets/2.txt"
text_file_dataset1 = ds.TextFileDataset(dataset_files=text_file_dataset_dir)
text_file_dataset = text_file_dataset1.map(operations=tokenizer)

for item in text_file_dataset:
    print(item[0])
    break
# Out:
# [ 165   28    8   11 4746 1430    4]

# In torch, return the sentencepiece model according to the input model path.
sp_id_generator = sentencepiece_numericalizer(sp_model)
list_a = ["sentencepiece encode as pieces", "examples to   try!"]
list(sp_id_generator(list_a))
```
