# 比较与torchtext.data.functional.load_sp_model的差异

[![查看源文件](https://mindspore-website.obs.cn-north-4.myhuaweicloud.com/website-images/br_base/resource/_static/logo_source.svg)](https://gitee.com/mindspore/docs/blob/br_base/docs/mindspore/source_zh_cn/note/api_mapping/pytorch_diff/load_sp_model.md)

## torchtext.data.functional.load_sp_model

```python
torchtext.data.functional.load_sp_model(
    spm
)
```

更多内容详见[torchtext.data.functional.load_sp_model](https://pytorch.org/text/0.9.0/data_functional.html#load-sp-model)。

## mindspore.dataset.text.SentencePieceTokenizer

```python
class mindspore.dataset.text.SentencePieceTokenizer(mode, out_type)
```

更多内容详见[mindspore.dataset.text.SentencePieceTokenizer](https://www.mindspore.cn/docs/zh-CN/br_base/api_python/dataset_text/mindspore.dataset.text.SentencePieceTokenizer.html#mindspore.dataset.text.SentencePieceTokenizer)。

## 使用方式

PyTorch：加载SentencePiece分词模型。

MindSpore：构造一个SentencePiece分词器，包含加载SentencePiece模型功能。

| 分类 | 子类 |PyTorch | MindSpore | 差异 |
| --- | ---   | ---   | ---        |---  |
|参数 | 参数1 | spm    | mode    | MindSpore支持SentencePiece词汇表或SentencePiece模型地址 |
|     | 参数2 | -    |out_type     | 分词器输出的类型 |

## 代码示例

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
