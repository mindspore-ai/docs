# 分词器

<!-- TOC depthFrom:1 depthTo:6 withLinks:1 updateOnSave:1 orderedList:0 -->

- [分词器](#分词器)
    - [概述](#概述)
    - [MindSpore分词器](#mindspore分词器)

<!-- /TOC -->

<a href="https://gitee.com/mindspore/docs/blob/master/api/source_zh_cn/programming_guide/tokenizer.md" target="_blank"><img src="../_static/logo_source.png"></a>

## 概述

分词就是将连续的字序列按照一定的规范重新组合成词序列的过程，合理的进行分词有助于语义的理解。

MindSpore提供了多种用途的分词器，能够帮助用户高性能地处理文本，用户可以构建自己的字典，使用适当的标记器将句子拆分为不同的标记，并通过查找操作获取字典中标记的索引。

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

更多分词器的详细说明，可以参见[API文档](https://www.mindspore.cn/api/zh-CN/master/api/python/mindspore/mindspore.dataset.text.html)。

## MindSpore分词器

### BasicTokenizer

`BasicTokenizer`是通过大小写折叠、编码统一、去除重音符，按照正则匹配模式来分词的。

```python
import mindspore.dataset as ds
import mindspore.dataset.text as text

# 构建输入的数据列表
input_list = ["Welcome to Beijing北京欢迎您", "長風破浪會有時，直掛雲帆濟滄海","😀嘿嘿😃哈哈😄大笑😁嘻嘻",
                "明朝（1368—1644年）和清朝（1644—1911年），是中国封建王朝史上最后两个朝代",
                "明代（1368-1644）と清代（1644-1911）は、中国の封建王朝の歴史における最後の2つの王朝でした",
                "명나라 (1368-1644)와 청나라 (1644-1911)는 중국 봉건 왕조의 역사에서 마지막 두 왕조였다"]

dataset = ds.NumpySlicesDataset(input_list, column_names=["text"], shuffle=False)

print("------------------------before tokenize----------------------------")

# 输出分词之前的数据
for data in dataset.create_dict_iterator():
    print(text.to_str(data['text']))

#打印分词后的数据输出
print("------------------------after tokenize-----------------------------")

# 输出分词之后的数据
# BasicTokenizer为分词的函数
basic_tokenizer = text.BasicTokenizer()

dataset = dataset.map(operations=basic_tokenizer)

for i in dataset.create_dict_iterator(num_epochs=1):
    token = text.to_str(i['text'])
    print(token)
```

```
------------------------before tokenize----------------------------
Welcome to Beijing北京欢迎您
長風破浪會有時，直掛雲帆濟滄海
😀嘿嘿😃哈哈😄大笑😁嘻嘻
明朝（1368—1644年）和清朝（1644—1911年），是中国封建王朝史上最后两个朝代
明代（1368-1644）と清代（1644-1911）は、中国の封建王朝の歴史における最後の2つの王朝でした
명나라 (1368-1644)와 청나라 (1644-1911)는 중국 봉건 왕조의 역사에서 마지막 두 왕조였다
------------------------after tokenize-----------------------------
['Welcome' 'to' 'Beijing' '北' '京' '欢' '迎' '您']
['長' '風' '破' '浪' '會' '有' '時' '，' '直' '掛' '雲' '帆' '濟' '滄' '海']
['😀' '嘿' '嘿' '😃' '哈' '哈' '😄' '大' '笑' '😁' '嘻' '嘻']
['明' '朝' '（' '1368' '—' '1644' '年' '）' '和' '清' '朝' '（' '1644' '—' '1911' '年' '）' '，' '是' '中' '国' '封' '建' '王' '朝' '史' '上' '最' '后' '两' '个' '朝' '代']
['明' '代' '（' '1368' '-' '1644' '）' 'と' '清' '代' '（' '1644' '-' '1911' '）' 'は' '、' '中' '国' 'の' '封' '建' '王' '朝' 'の' '歴' '史' 'における' '最' '後' 'の2つの' '王' '朝' 'でした']
['명나라' '(' '1368' '-' '1644' ')' '와' '청나라' '(' '1644' '-' '1911' ')' '는' '중국' '봉건' '왕조의' '역사에서' '마지막' '두' '왕조였다']
```

### BertTokenizer

`BertTokenizer`是通过调用`BasicTokenizer`和`WordpieceTokenizer`来进行分词的。

```python
import mindspore.dataset as ds
import mindspore.dataset.text as text

# 构建输入的数据列表
input_list = ["床前明月光", "疑是地上霜", "举头望明月", "低头思故乡", "I am making small mistakes during working hours",
                "😀嘿嘿😃哈哈😄大笑😁嘻嘻", "繁體字"]

dataset = ds.NumpySlicesDataset(input_list, column_names=["text"], shuffle=False)

print("------------------------before tokenize----------------------------")

# 输出分词之前的数据
for data in dataset.create_dict_iterator():
    print(text.to_str(data['text']))

# 字符串列表，其中每个元素都是字符串类型的单词。
vocab_list = [
  "床", "前", "明", "月", "光", "疑", "是", "地", "上", "霜", "举", "头", "望", "低", "思", "故", "乡",
  "繁", "體", "字", "嘿", "哈", "大", "笑", "嘻", "i", "am", "mak", "make", "small", "mistake",
  "##s", "during", "work", "##ing", "hour", "😀", "😃", "😄", "😁", "+", "/", "-", "=", "12",
  "28", "40", "16", " ", "I", "[CLS]", "[SEP]", "[UNK]", "[PAD]", "[MASK]", "[unused1]", "[unused10]"]

# 从单词列表中构建一个vocab对象
vocab = text.Vocab.from_list(vocab_list)

# 输出分词之后的数据
# BertTokenizer为分词的函数
tokenizer_op = text.BertTokenizer(vocab=vocab)

#打印分词后的数据输出
print("------------------------after tokenize-----------------------------")

dataset = dataset.map(operations=tokenizer_op)

for i in dataset.create_dict_iterator(num_epochs=1):
    token = text.to_str(i['text'])
    print(token)
```

```
------------------------before tokenize----------------------------
床前明月光
疑是地上霜
举头望明月
低头思故乡
I am making small mistakes during working hours
😀嘿嘿😃哈哈😄大笑😁嘻嘻
繁體字
------------------------after tokenize-----------------------------
['床' '前' '明' '月' '光']
['疑' '是' '地' '上' '霜']
['举' '头' '望' '明' '月']
['低' '头' '思' '故' '乡']
['i' 'am' 'mak' '##ing' 'small' 'mistake' '##s' 'during' 'work' '##ing' 'hour' '##s']
['😀' '嘿' '嘿' '😃' '哈' '哈' '😄' '大' '笑' '😁' '嘻' '嘻']
['繁' '體' '字']
```

### JiebaTokenizer

`JiebaTokenizer`是基于jieba的中文分词。

```python
import mindspore.dataset as ds
import mindspore.dataset.text as text

# 构建输入的数据列表
input_list = ["床前明月光", "疑是地上霜", "举头望明月", "低头思故乡", "I am making small mistakes during working hours",
                "😀嘿嘿😃哈哈😄大笑😁嘻嘻", "繁體字"]

# 字典文件由HMMSegment算法和MPSegment算法使用，该字典可在cppjieba的官方网站上获得。
HMM_FILE = "hmm_model.utf8"
MP_FILE = "jieba.dict.utf8"

dataset = ds.NumpySlicesDataset(input_list, column_names=["text"], shuffle=False)

print("------------------------before tokenize----------------------------")

# 输出分词之前的数据
for data in dataset.create_dict_iterator():
    print(text.to_str(data['text']))

tokenizer_op = text.JiebaTokenizer(HMM_FILE, MP_FILE)

#打印分词后的数据输出
print("------------------------after tokenize-----------------------------")

dataset = dataset.map(operations=jieba_op, input_columns=["text"], num_parallel_workers=1)

for i in dataset.create_dict_iterator(num_epochs=1):
    token = text.to_str(i['text'])
    print(token)
```

```
------------------------before tokenize----------------------------
今天天气太好了我们一起去外面玩吧
------------------------after tokenize-----------------------------
['今天天气' '太好了' '我们' '一起' '去' '外面' '玩吧']
```

### RegexTokenizer

`RegexTokenizer`是通正则表达式匹配模式来进行分词的。

```python
import mindspore.dataset as ds
import mindspore.dataset.text as text

# 构建输入的数据列表
input_list = ["Welcome to Shenzhen!"]

# 原始字符串将由匹配的元素分隔。
delim_pattern = "\\s+"

dataset = ds.NumpySlicesDataset(input_list, column_names=["text"], shuffle=False)

print("------------------------before tokenize----------------------------")

# 输出分词之前的数据
for data in dataset.create_dict_iterator():
    print(text.to_str(data['text']))

tokenizer_op = text.RegexTokenizer(delim_pattern)

#打印分词后的数据输出
print("------------------------after tokenize-----------------------------")

dataset = dataset.map(operations=tokenizer_op)

for i in dataset.create_dict_iterator(num_epochs=1):
    token = text.to_str(i['text']).tolist()
    print(token)
```

```
------------------------before tokenize----------------------------
Welcome to Shenzhen!
------------------------after tokenize-----------------------------
['Welcome', 'to', 'Shenzhen!']
```

### SentencePieceTokenizer

`SentencePieceTokenizer`是基于SentencePiece这个开源的自然语言处理工具包。

```python
import mindspore.dataset as ds
import mindspore.dataset.text as text

# 构建输入的数据列表
input_list = ["I saw a girl with a telescope."]

dataset = ds.NumpySlicesDataset(input_list, column_names=["text"], shuffle=False)

print("------------------------before tokenize----------------------------")

# 输出分词之前的数据
for data in dataset.create_dict_iterator():
    print(text.to_str(data['text']))

# 从文件数据中构建一个vocab对象
vocab = text.SentencePieceVocab.from_file([VOCAB_FILE], 5000, 0.9995, SentencePieceModel.UNIGRAM, {})
tokenizer_op = text.SentencePieceTokenizer(vocab, out_type=SPieceTokenizerOutType.STRING)

#打印分词后的数据输出
print("------------------------after tokenize-----------------------------")

dataset = dataset.map(operations=tokenizer_op)

for i in dataset.create_dict_iterator(num_epochs=1):
    token = text.to_str(i['text'])
    print(token)
```

```
------------------------before tokenize----------------------------
I saw a girl with a telescope.
------------------------after tokenize-----------------------------
['▁I' '▁sa' 'w' '▁a' '▁girl' '▁with' '▁a' '▁te' 'les' 'co' 'pe' '.']
```

### UnicodeCharTokenizer

`UnicodeCharTokenizer`是根据Unicode字符集来分词的。

```python
import mindspore.dataset as ds
import mindspore.dataset.text as text

# 构建输入的数据列表
input_list = ["Welcome to Beijing!", "北京欢迎您！", "我喜欢English!"]

dataset = ds.NumpySlicesDataset(input_list, column_names=["text"], shuffle=False)

print("------------------------before tokenize----------------------------")

# 输出分词之前的数据
for data in dataset.create_dict_iterator():
    print(text.to_str(data['text']))

tokenizer_op = text.UnicodeCharTokenizer()

#打印分词后的数据输出
print("------------------------after tokenize-----------------------------")

dataset = dataset.map(operations=tokenizer_op)

for i in dataset.create_dict_iterator(num_epochs=1):
    token = text.to_str(i['text']).tolist()
    print(token)
```

```
------------------------before tokenize----------------------------
Welcome to Beijing!
北京欢迎您！
我喜欢English!
------------------------after tokenize-----------------------------
['W', 'e', 'l', 'c', 'o', 'm', 'e', ' ', 't', 'o', ' ', 'B', 'e', 'i', 'j', 'i', 'n', 'g', '!']
['北', '京', '欢', '迎', '您', '！']
['我', '喜', '欢', 'E', 'n', 'g', 'l', 'i', 's', 'h', '!']
```

### UnicodeScriptTokenizer

`UnicodeScriptTokenizer`是根据不同的Unicode的边界来进行分词的。

```python
import mindspore.dataset as ds
import mindspore.dataset.text as text

# 构建输入的数据列表
input_list = ["Welcome to Beijing!", "北京欢迎您！", "我喜欢English!"]

dataset = ds.NumpySlicesDataset(input_list, column_names=["text"], shuffle=False)

print("------------------------before tokenize----------------------------")

# 输出分词之前的数据
for data in dataset.create_dict_iterator():
    print(text.to_str(data['text']))

tokenizer_op = text.UnicodeScriptTokenizer()

#打印分词后的数据输出
print("------------------------after tokenize-----------------------------")

dataset = dataset.map(operations=tokenizer_op)

for i in dataset.create_dict_iterator(num_epochs=1):
    token = text.to_str(i['text']).tolist()
    print(token)      
```

```
------------------------before tokenize----------------------------
Welcome to Beijing!
北京欢迎您！
我喜欢English!
------------------------after tokenize-----------------------------
['Welcome', 'to', 'Beijing', '!']
['北京欢迎您', '！']
['我喜欢', 'English', '!']
```

### WhitespaceTokenizer

`WhitespaceTokenizer`是根据空格来进行分词的。

```python
import mindspore.dataset as ds
import mindspore.dataset.text as text

# 构建输入的数据列表
input_list = ["Welcome to Beijing!", "北京欢迎您！", "我喜欢English!"]

dataset = ds.NumpySlicesDataset(input_list, column_names=["text"], shuffle=False)

print("------------------------before tokenize----------------------------")

# 输出分词之前的数据
for data in dataset.create_dict_iterator():
    print(text.to_str(data['text']))

tokenizer_op = text.WhitespaceTokenizer()

#打印分词后的数据输出
print("------------------------after tokenize-----------------------------")

dataset = dataset.map(operations=tokenizer_op)

for i in dataset.create_dict_iterator(num_epochs=1):
    token = text.to_str(i['text']).tolist()
    print(token)
```

```
>> Tokenize Result
------------------------before tokenize----------------------------
Welcome to Beijing!
北京欢迎您！
我喜欢English!
------------------------after tokenize-----------------------------
['Welcome', 'to', 'Beijing!']
['北京欢迎您！']
['我喜欢English!']
```

### WordpieceTokenizer

`WordpieceTokenizer`是基于单词集来划分的，单词集里没有的，但是有组合的也会划分出来。

```python
import mindspore.dataset as ds
import mindspore.dataset.text as text

# 构建输入的数据列表
input_list = ["my", "favorite", "book", "is", "love", "during", "the", "cholera", "era", "what", "我", "最", "喜", "欢", "的", "书", "是", "霍", "乱", "时", "期", "的", "爱", "情", "您"]

dataset = ds.NumpySlicesDataset(input_list, column_names=["text"], shuffle=False)

print("------------------------before tokenize----------------------------")

# 输出分词之前的数据
for data in dataset.create_dict_iterator():
    print(text.to_str(data['text']))

#打印分词后的数据输出
print("------------------------after tokenize-----------------------------")

# 从单词列表中构建一个vocab对象
vocab = text.Vocab.from_list(vocab_list)

# 输出分词之后的数据
# BasicTokenizer为分词的函数
tokenizer_op = text.WordpieceTokenizer(vocab=vocab)

dataset = dataset.map(operations=tokenizer_op)

for i in dataset.create_dict_iterator(num_epochs=1):
    token = text.to_str(i['text'])
    print(token)
```

```
------------------------before tokenize----------------------------
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
------------------------after tokenize-----------------------------
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
