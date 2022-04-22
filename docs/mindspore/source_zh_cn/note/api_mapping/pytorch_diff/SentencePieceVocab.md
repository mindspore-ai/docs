# 比较与torchtext.data.functional.load_sp_model的功能差异

<a href="https://gitee.com/mindspore/docs/blob/master/docs/mindspore/source_zh_cn/note/api_mapping/pytorch_diff/SentencePieceVocab.md" target="_blank"><img src="https://mindspore-website.obs.cn-north-4.myhuaweicloud.com/website-images/master/resource/_static/logo_source.png"></a>

## torchtext.data.functional.load_sp_model

```python
torchtext.data.functional.load_sp_model(
    spm
)
```

更多内容详见[torchtext.data.functional.load_sp_model](https://pytorch.org/text/0.10.0/data_functional.html#load-sp-model)。

## mindspore.dataset.text.utils.SentencePieceVocab

```python
classmindspore.dataset.text.utils.SentencePieceVocab
```

更多内容详见[mindspore.dataset.text.utils.SentencePieceVocab](https://www.mindspore.cn/docs/zh-CN/master/api_python/dataset_text/mindspore.dataset.text.SentencePieceVocab.html#mindspore.dataset.text.utils.SentencePieceVocab)。

## 使用方式

PyTorch：为文件加载语句片段模型。输入句子片段模型的路径，输出句子片段模型。

MindSpore：构造用于单词分割的词汇表，输入可以是数据集对象或词汇表文件。

## 代码示例

```python
import mindspore.dataset as ds
from mindspore.dataset import text
from mindspore.dataset.text import SentencePieceModel, SPieceTokenizerOutType, to_str
from torchtext.data.functional import load_sp_model

# In MindSpore, return tokenizer from vocab object.
sentence_piece_vocab_file = "/path/to/test_sentencepiece/botchan.txt"

vocab = text.SentencePieceVocab.from_file([sentence_piece_vocab_file], 5000, 0.9995,
                                          SentencePieceModel.WORD, {})
tokenizer = text.SentencePieceTokenizer(vocab, out_type=SPieceTokenizerOutType.STRING)
text_file_dataset_dir = "/path/to/testTokenizerData/sentencepiece_tokenizer.txt"
text_file_dataset = ds.TextFileDataset(dataset_files=text_file_dataset_dir)
text_file_dataset = text_file_dataset.map(operations=tokenizer)

for i in text_file_dataset.create_dict_iterator(num_epochs=1, output_numpy=True):
    ret = to_str(i["text"])
    for key, value in enumerate(ret):
        print(value)
# Out:
# ▁I
# ▁saw
# ▁a
# ▁girl
# ▁with
# ▁a
# ▁telescope.

# In torch, return the sentencepiece model according to the input model path.
sp_model = load_sp_model("m_user.model")
sp_model = load_sp_model(open("m_user.model", 'rb'))
```
