# 分词器

<a href="https://gitee.com/mindspore/docs/blob/r1.1/docs/programming_guide/source_zh_cn/tokenizer.md" target="_blank"><img src="./_static/logo_source.png"></a>
&nbsp;&nbsp;
<a href="https://obs.dualstack.cn-north-4.myhuaweicloud.com/mindspore-website/notebook/r1.1/programming_guide/mindspore_tokenizer.ipynb"><img src="./_static/logo_notebook.png"></a>
&nbsp;&nbsp;
<a href="https://console.huaweicloud.com/modelarts/?region=cn-north-4#/notebook/loading?share-url-b64=aHR0cHM6Ly9vYnMuZHVhbHN0YWNrLmNuLW5vcnRoLTQubXlodWF3ZWljbG91ZC5jb20vbWluZHNwb3JlLXdlYnNpdGUvbm90ZWJvb2svbW9kZWxhcnRzL3Byb2dyYW1taW5nX2d1aWRlL21pbmRzcG9yZV90b2tlbml6ZXIuaXB5bmI=&image_id=65f636a0-56cf-49df-b941-7d2a07ba8c8c" target="_blank"><img src="./_static/logo_modelarts.png"></a>

## 概述

分词就是将连续的字序列按照一定的规范重新组合成词序列的过程，合理的进行分词有助于语义的理解。

MindSpore提供了多种用途的分词器（Tokenizer），能够帮助用户高性能地处理文本，用户可以构建自己的字典，使用适当的标记器将句子拆分为不同的标记，并通过查找操作获取字典中标记的索引。

MindSpore目前提供的分词器如下表所示。此外，用户也可以根据需要实现自定义的分词器。

| 分词器 | 分词器说明 |
| -- | -- |
| BasicTokenizer | 根据指定规则对标量文本数据进行分词。 |
| BertTokenizer | 用于处理Bert文本数据的分词器。 |
| JiebaTokenizer | 基于字典的中文字符串分词器。 |
| RegexTokenizer | 根据指定正则表达式对标量文本数据进行分词。 |
| SentencePieceTokenizer | 基于SentencePiece开源工具包进行分词。 |
| UnicodeCharTokenizer | 将标量文本数据分词为Unicode字符。 |
| UnicodeScriptTokenizer | 根据Unicode边界对标量文本数据进行分词。 |
| WhitespaceTokenizer | 根据空格符对标量文本数据进行分词。 |
| WordpieceTokenizer | 根据单词集对标量文本数据进行分词。 |

更多分词器的详细说明，可以参见[API文档](https://www.mindspore.cn/doc/api_python/zh-CN/r1.1/mindspore/mindspore.dataset.text.html)。

## MindSpore分词器

下面介绍几种常用分词器的使用方法。

### BertTokenizer

`BertTokenizer`是通过调用`BasicTokenizer`和`WordpieceTokenizer`来进行分词的。

下面的样例首先构建了一个文本数据集和字符串列表，然后通过`BertTokenizer`对数据集进行分词，并展示了分词前后的文本结果。

```python
import mindspore.dataset as ds
import mindspore.dataset.text as text

input_list = ["床前明月光", "疑是地上霜", "举头望明月", "低头思故乡", "I am making small mistakes during working hours",
                "😀嘿嘿😃哈哈😄大笑😁嘻嘻", "繁體字"]
dataset = ds.NumpySlicesDataset(input_list, column_names=["text"], shuffle=False)

print("------------------------before tokenization----------------------------")

for data in dataset.create_dict_iterator(output_numpy=True):
    print(text.to_str(data['text']))

vocab_list = [
  "床", "前", "明", "月", "光", "疑", "是", "地", "上", "霜", "举", "头", "望", "低", "思", "故", "乡",
  "繁", "體", "字", "嘿", "哈", "大", "笑", "嘻", "i", "am", "mak", "make", "small", "mistake",
  "##s", "during", "work", "##ing", "hour", "😀", "😃", "😄", "😁", "+", "/", "-", "=", "12",
  "28", "40", "16", " ", "I", "[CLS]", "[SEP]", "[UNK]", "[PAD]", "[MASK]", "[unused1]", "[unused10]"]

vocab = text.Vocab.from_list(vocab_list)
tokenizer_op = text.BertTokenizer(vocab=vocab)
dataset = dataset.map(operations=tokenizer_op)

print("------------------------after tokenization-----------------------------")

for i in dataset.create_dict_iterator(num_epochs=1, output_numpy=True):
    print(text.to_str(i['text']))
```

输出结果如下：

```text
------------------------before tokenization----------------------------
床前明月光
疑是地上霜
举头望明月
低头思故乡
I am making small mistakes during working hours
😀嘿嘿😃哈哈😄大笑😁嘻嘻
繁體字
------------------------after tokenization-----------------------------
['床' '前' '明' '月' '光']
['疑' '是' '地' '上' '霜']
['举' '头' '望' '明' '月']
['低' '头' '思' '故' '乡']
['I' 'am' 'mak' '##ing' 'small' 'mistake' '##s' 'during' 'work' '##ing'
 'hour' '##s']
['😀' '嘿' '嘿' '😃' '哈' '哈' '😄' '大' '笑' '😁' '嘻' '嘻']
['繁' '體' '字']
```

### JiebaTokenizer

`JiebaTokenizer`是基于jieba的中文分词。

下面的样例首先构建了一个文本数据集，然后使用HMM与MP字典文件创建`JiebaTokenizer`对象，并对数据集进行分词，最后展示了分词前后的文本结果。

```python
import mindspore.dataset as ds
import mindspore.dataset.text as text

input_list = ["今天天气太好了我们一起去外面玩吧"]
dataset = ds.NumpySlicesDataset(input_list, column_names=["text"], shuffle=False)

print("------------------------before tokenization----------------------------")

for data in dataset.create_dict_iterator(output_numpy=True):
    print(text.to_str(data['text']))

# files from open source repository https://github.com/yanyiwu/cppjieba/tree/master/dict
HMM_FILE = "hmm_model.utf8"
MP_FILE = "jieba.dict.utf8"
jieba_op = text.JiebaTokenizer(HMM_FILE, MP_FILE)
dataset = dataset.map(operations=jieba_op, input_columns=["text"], num_parallel_workers=1)

print("------------------------after tokenization-----------------------------")

for i in dataset.create_dict_iterator(num_epochs=1, output_numpy=True):
    print(text.to_str(i['text']))
```

输出结果如下：

```text
------------------------before tokenization----------------------------
今天天气太好了我们一起去外面玩吧
------------------------after tokenization-----------------------------
['今天天气' '太好了' '我们' '一起' '去' '外面' '玩吧']
```

### SentencePieceTokenizer

`SentencePieceTokenizer`是基于[SentencePiece](https://github.com/google/sentencepiece)这个开源的自然语言处理工具包。

下面的样例首先构建了一个文本数据集，然后从`vocab_file`文件中构建一个`vocab`对象，再通过`SentencePieceTokenizer`对数据集进行分词，并展示了分词前后的文本结果。

```python
import mindspore.dataset as ds
import mindspore.dataset.text as text
from mindspore.dataset.text import SentencePieceModel, SPieceTokenizerOutType

input_list = ["I saw a girl with a telescope."]
dataset = ds.NumpySlicesDataset(input_list, column_names=["text"], shuffle=False)

print("------------------------before tokenization----------------------------")

for data in dataset.create_dict_iterator(output_numpy=True):
    print(text.to_str(data['text']))

# file from MindSpore repository https://gitee.com/mindspore/mindspore/blob/r1.1/tests/ut/data/dataset/test_sentencepiece/botchan.txt
vocab_file = "botchan.txt"
vocab = text.SentencePieceVocab.from_file([vocab_file], 5000, 0.9995, SentencePieceModel.UNIGRAM, {})
tokenizer_op = text.SentencePieceTokenizer(vocab, out_type=SPieceTokenizerOutType.STRING)
dataset = dataset.map(operations=tokenizer_op)

print("------------------------after tokenization-----------------------------")

for i in dataset.create_dict_iterator(num_epochs=1, output_numpy=True):
    print(text.to_str(i['text']))
```

输出结果如下：

```text
------------------------before tokenization----------------------------
I saw a girl with a telescope.
------------------------after tokenization-----------------------------
['▁I' '▁sa' 'w' '▁a' '▁girl' '▁with' '▁a' '▁te' 'les' 'co' 'pe' '.']
```

### UnicodeCharTokenizer

`UnicodeCharTokenizer`是根据Unicode字符集来分词的。

下面的样例首先构建了一个文本数据集，然后通过`UnicodeCharTokenizer`对数据集进行分词，并展示了分词前后的文本结果。

```python
import mindspore.dataset as ds
import mindspore.dataset.text as text

input_list = ["Welcome to Beijing!", "北京欢迎您！", "我喜欢English!"]
dataset = ds.NumpySlicesDataset(input_list, column_names=["text"], shuffle=False)

print("------------------------before tokenization----------------------------")

for data in dataset.create_dict_iterator(output_numpy=True):
    print(text.to_str(data['text']))

tokenizer_op = text.UnicodeCharTokenizer()
dataset = dataset.map(operations=tokenizer_op)

print("------------------------after tokenization-----------------------------")

for i in dataset.create_dict_iterator(num_epochs=1, output_numpy=True):
    print(text.to_str(i['text']).tolist())
```

输出结果如下：

```text
------------------------before tokenization----------------------------
Welcome to Beijing!
北京欢迎您！
我喜欢English!
------------------------after tokenization-----------------------------
['W', 'e', 'l', 'c', 'o', 'm', 'e', ' ', 't', 'o', ' ', 'B', 'e', 'i', 'j', 'i', 'n', 'g', '!']
['北', '京', '欢', '迎', '您', '！']
['我', '喜', '欢', 'E', 'n', 'g', 'l', 'i', 's', 'h', '!']
```

### WhitespaceTokenizer

`WhitespaceTokenizer`是根据空格来进行分词的。

下面的样例首先构建了一个文本数据集，然后通过`WhitespaceTokenizer`对数据集进行分词，并展示了分词前后的文本结果。

```python
import mindspore.dataset as ds
import mindspore.dataset.text as text

input_list = ["Welcome to Beijing!", "北京欢迎您！", "我喜欢English!"]
dataset = ds.NumpySlicesDataset(input_list, column_names=["text"], shuffle=False)

print("------------------------before tokenization----------------------------")

for data in dataset.create_dict_iterator(output_numpy=True):
    print(text.to_str(data['text']))

tokenizer_op = text.WhitespaceTokenizer()
dataset = dataset.map(operations=tokenizer_op)

print("------------------------after tokenization-----------------------------")

for i in dataset.create_dict_iterator(num_epochs=1, output_numpy=True):
    print(text.to_str(i['text']).tolist())
```

输出结果如下：

```text
------------------------before tokenization----------------------------
Welcome to Beijing!
北京欢迎您！
我喜欢English!
------------------------after tokenization-----------------------------
['Welcome', 'to', 'Beijing!']
['北京欢迎您！']
['我喜欢English!']
```

### WordpieceTokenizer

`WordpieceTokenizer`是基于单词集来进行划分的，划分依据可以是单词集中的单个单词，或者多个单词的组合形式。

下面的样例首先构建了一个文本数据集，然后从单词列表中构建`vocab`对象，通过`WordpieceTokenizer`对数据集进行分词，并展示了分词前后的文本结果。

```python
import mindspore.dataset as ds
import mindspore.dataset.text as text

input_list = ["my", "favorite", "book", "is", "love", "during", "the", "cholera", "era", "what",
    "我", "最", "喜", "欢", "的", "书", "是", "霍", "乱", "时", "期", "的", "爱", "情", "您"]
vocab_english = ["book", "cholera", "era", "favor", "##ite", "my", "is", "love", "dur", "##ing", "the"]
vocab_chinese = ["我", '最', '喜', '欢', '的', '书', '是', '霍', '乱', '时', '期', '爱', '情']

dataset = ds.NumpySlicesDataset(input_list, column_names=["text"], shuffle=False)

print("------------------------before tokenization----------------------------")

for data in dataset.create_dict_iterator(output_numpy=True):
    print(text.to_str(data['text']))

vocab = text.Vocab.from_list(vocab_english+vocab_chinese)
tokenizer_op = text.WordpieceTokenizer(vocab=vocab)
dataset = dataset.map(operations=tokenizer_op)

print("------------------------after tokenization-----------------------------")

for i in dataset.create_dict_iterator(num_epochs=1, output_numpy=True):
    print(text.to_str(i['text']))
```

输出结果如下：

```text
------------------------before tokenization----------------------------
my
favorite
book
is
love
during
the
cholera
era
what
我
最
喜
欢
的
书
是
霍
乱
时
期
的
爱
情
您
------------------------after tokenization-----------------------------
['my']
['favor' '##ite']
['book']
['is']
['love']
['dur' '##ing']
['the']
['cholera']
['era']
['[UNK]']
['我']
['最']
['喜']
['欢']
['的']
['书']
['是']
['霍']
['乱']
['时']
['期']
['的']
['爱']
['情']
['[UNK]']
```
