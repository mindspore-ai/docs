# 比较与torchtext.data.functional.load_sp_model的功能差异

<a href="https://gitee.com/mindspore/docs/blob/master/docs/mindspore/source_zh_cn/note/api_mapping/pytorch_diff/SentencePieceTokenizer.md" target="_blank"><img src="https://mindspore-website.obs.cn-north-4.myhuaweicloud.com/website-images/master/resource/_static/logo_source.png"></a>

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

更多内容详见[mindspore.dataset.text.SentencePieceTokenizer](https://www.mindspore.cn/docs/zh-CN/master/api_python/dataset_text/mindspore.dataset.text.SentencePieceTokenizer.html#mindspore.dataset.text.SentencePieceTokenizer)。

## 使用方式

PyTorch：加载语句片段模型。

MindSpore：构造一个语句片段分词器，包含加载语句片段模型功能。

| 分类 | 子类 |PyTorch | MindSpore | 差异 |
| --- | ---   | ---   | ---        |---  |
|参数 | 参数1 | spm    | mode    | MindSpore支持SentencePiece词汇表或语句片段模型地址 |
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
