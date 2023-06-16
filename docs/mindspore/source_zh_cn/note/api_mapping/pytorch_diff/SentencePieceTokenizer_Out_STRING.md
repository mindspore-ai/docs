# 比较与torchtext.data.functional.sentencepiece_tokenizer的功能差异

<a href="https://gitee.com/mindspore/docs/blob/master/docs/mindspore/source_zh_cn/note/api_mapping/pytorch_diff/SentencePieceTokenizer_Out_STRING.md" target="_blank"><img src="https://mindspore-website.obs.cn-north-4.myhuaweicloud.com/website-images/master/resource/_static/logo_source.png"></a>

## torchtext.data.functional.sentencepiece_tokenizer

```python
torchtext.data.functional.sentencepiece_tokenizer(
    sp_model
)
```

更多内容详见[torchtext.data.functional.sentencepiece_tokenizer](https://pytorch.org/text/0.9.0/data_functional.html#sentencepiece-tokenizer)。

## mindspore.dataset.text.SentencePieceTokenizer

```python
class mindspore.dataset.text.SentencePieceTokenizer(
    mode,
    out_type
)
```

更多内容详见[mindspore.dataset.text.SentencePieceTokenizer](https://mindspore.cn/docs/zh-CN/master/api_python/dataset_text/mindspore.dataset.text.SentencePieceTokenizer.html#mindspore.dataset.text.SentencePieceTokenizer)。

## 使用方式

PyTorch：依据传入的分词模型，返回将文本转换为字符串的生成器。

MindSpore：依据传入的分词模型，对输入的文本进行分词及标记；输出类型是string或int类型。

| 分类 | 子类 |PyTorch | MindSpore | 差异 |
| --- | ---   | ---   | ---        |---  |
|参数 | 参数1 | sp_model    | mode    | MindSpore支持SentencePiece词汇表或分词模型地址 |
|     | 参数2 | -    |out_type     | 分词器输出的类型 |

## 代码示例

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
