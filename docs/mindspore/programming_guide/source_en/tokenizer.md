# Text Data Processing and Enhancement

`Ascend` `GPU` `CPU` `Data Preparation`

<!-- TOC -->

- [Text Data Processing and Enhancement](#text-data-processing-and-enhancement)
    - [Overview](#overview)
    - [Vocab build and usage](#vocab-build-and-usage)
    - [MindSpore Tokenizers](#mindspore-tokenizers)
        - [BertTokenizer](#berttokenizer)
        - [JiebaTokenizer](#jiebatokenizer)
        - [SentencePieceTokenizer](#sentencepiecetokenizer)
        - [UnicodeCharTokenizer](#unicodechartokenizer)
        - [WhitespaceTokenizer](#whitespacetokenizer)
        - [WordpieceTokenizer](#wordpiecetokenizer)

<!-- /TOC -->

<a href="https://gitee.com/mindspore/docs/blob/master/docs/mindspore/programming_guide/source_en/tokenizer.md" target="_blank"><img src="https://gitee.com/mindspore/docs/raw/master/resource/_static/logo_source_en.png"></a>

## Overview

With the gradual increase of available text data, it is more urgent to preprocess the text data in order to obtain clean data that can be used for network training. The preprocessing of text data usually includes two parts: loading text data and data enhancement.

There are several ways to load text data:

- Load text data with specific dataset API, such as [ClueDataset](https://www.mindspore.cn/docs/api/en/master/api_python/dataset/mindspore.dataset.CLUEDataset.html#mindspore.dataset.CLUEDataset), [TextFileDataset](https://www.mindspore.cn/docs/api/en/master/api_python/dataset/mindspore.dataset.TextFileDataset.html#mindspore.dataset.TextFileDataset).
- Convert the data into a standard format (such as MindRecord format), and then load it with corresponding interface (such as [MindDataset](https://www.mindspore.cn/docs/api/en/master/api_python/dataset/mindspore.dataset.MindDataset.html#mindspore.dataset.MindDataset)).
- Use GeneratorDataset interface to load data, here the user-defined dataset loading function is required as input. For detail, please refer to chapter of [loading-user-defined-dataset](https://www.mindspore.cn/docs/programming_guide/en/master/dataset_loading.html#loading-user-defined-dataset).

As for text data enhancement, common operations include text segmentation, vocabulary lookup, etc.

- After loading the text data, word segmentation is usually required, that is, the original sentences is continuously divided into multiple basic words.
- Furthermore, it is necessary to build a vocabulary, in order to find the id corresponding to each word after segmentation, and pass the id contained in the sentence into the network for training.

Following is an introduction to the word segmentation function and vocabulary lookup function used in the data enhancement process. For the usage of the text processing API, please refer to the [API document](https://www.mindspore.cn/docs/api/en/master/api_python/mindspore.dataset.text.html).

## Vocab build and usage

The vocab provides the mapping relationship between words and id. With the vocab, the corresponding word id can be found by inputting the word, and vice versa, the corresponding word can also be obtained according to the word id.

MindSpore provides several methods for constructing a vocabulary, which can obtain raw data from dictionaries, files, lists, and Dataset objects for constructing a vocabulary. The corresponding interface is: [from_dict](https://www.mindspore.cn/docs/api/en/master/api_python/dataset_text/mindspore.dataset.text.Vocab.html#mindspore.dataset.text.Vocab.from_dict), [from_file](https://www.mindspore.cn/docs/api/en/master/api_python/dataset_text/mindspore.dataset.text.Vocab.html#mindspore.dataset.text.Vocab.from_file), [from_list](https://www.mindspore.cn/docs/api/en/master/api_python/dataset_text/mindspore.dataset.text.Vocab.html#mindspore.dataset.text.Vocab.from_list), [from_dataset](https://www.mindspore.cn/docs/api/en/master/api_python/dataset_text/mindspore.dataset.text.Vocab.html#mindspore.dataset.text.Vocab.from_dataset).

Taking `from_dict` as an example, the way to construct Vocab is as follows. The incoming dict contains multiple sets of words and id pairs.

```python
from mindspore.dataset import text

vocab = text.Vocab.from_dict({"home": 3, "behind": 2, "the": 4, "world": 5, "<unk>": 6})
```

Vocab provides methods for querying each other between words and ids, namely: [tokens_to_ids]() and [ids_to_tokens] methods, the usage is as follows:

```python
from mindspore.dataset import text

vocab = text.Vocab.from_dict({"home": 3, "behind": 2, "the": 4, "world": 5, "<unk>": 6})

ids = vocab.tokens_to_ids(["home", "world"])
print("ids: ", ids)

tokens = vocab.ids_to_tokens([2, 5])
print("tokens: ", tokens)
```

The output is as follows:

```text
ids:  [3, 5]
tokens:  ['behind', 'world']
```

In addition, Vocab is also a necessary input parameter for various tokenizers (such as WordpieceTokenizer). During word segmentation, the word that exists in the vocabulary in the sentence will be divided into a single word, and then the corresponding word id can be obtained by looking up the vocabulary.

## MindSpore Tokenizers

Tokenization is a process of re-combining continuous character sequences into word sequences according to certain specifications. Reasonable tokenization is helpful for semantic comprehension.

MindSpore provides a tokenizer for multiple purposes to help you process text with high performance. You can build your own dictionaries, use appropriate tokenizers to split sentences into different tokens, and search for indexes of the tokens in the dictionaries.

MindSpore provides the following tokenizers. In addition, you can customize tokenizers as required.

| Tokenizer | Description |
| -- | -- |
| BasicTokenizer | Performs tokenization on scalar text data based on specified rules.  |
| BertTokenizer | Processes BERT text data.  |
| JiebaTokenizer | Dictionary-based Chinese character string tokenizer.  |
| RegexTokenizer | Performs tokenization on scalar text data based on a specified regular expression.  |
| SentencePieceTokenizer | Performs tokenization based on the open-source tool package SentencePiece.  |
| UnicodeCharTokenizer | Tokenizes scalar text data into Unicode characters.  |
| UnicodeScriptTokenizer | Performs tokenization on scalar text data based on Unicode boundaries.  |
| WhitespaceTokenizer | Performs tokenization on scalar text data based on spaces.  |
| WordpieceTokenizer | Performs tokenization on scalar text data based on the word set.  |

For details about tokenizers, see [MindSpore API](https://www.mindspore.cn/docs/api/en/master/api_python/mindspore.dataset.text.html).

The following describes how to use common tokenizers.

### BertTokenizer

`BertTokenizer` performs tokenization by calling `BasicTokenizer` and `WordpieceTokenizer`.

The following example builds a text dataset and a character string list, uses `BertTokenizer` to perform tokenization on the dataset, and displays the text results before and after tokenization.

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

The output is as follows:

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

`JiebaTokenizer` performs Chinese tokenization based on Jieba.

Download the dictionary files `hmm_model.utf8` and `jieba.dict.utf8` and put them in the specified location.

```python
import os
import requests

def download_dataset(dataset_url, path):
    filename = dataset_url.split("/")[-1]
    save_path = os.path.join(path, filename)
    if os.path.exists(save_path):
        return
    if not os.path.exists(path):
        os.makedirs(path)
    res = requests.get(dataset_url, stream=True, verify=False)
    with open(save_path, "wb") as f:
        for chunk in res.iter_content(chunk_size=512):
            if chunk:
                f.write(chunk)

download_dataset("https://obs.dualstack.cn-north-4.myhuaweicloud.com/mindspore-website/notebook/datasets/hmm_model.utf8", "./datasets/tokenizer/")
download_dataset("https://obs.dualstack.cn-north-4.myhuaweicloud.com/mindspore-website/notebook/datasets/jieba.dict.utf8", "./datasets/tokenizer/")
```

```text
./datasets/tokenizer/
├── hmm_model.utf8
└── jieba.dict.utf8

0 directories, 2 files
```

The following example builds a text dataset, uses the HMM and MP dictionary files to create a `JiebaTokenizer` object, performs tokenization on the dataset, and displays the text results before and after tokenization.

```python
import mindspore.dataset as ds
import mindspore.dataset.text as text

input_list = ["今天天气太好了我们一起去外面玩吧"]
dataset = ds.NumpySlicesDataset(input_list, column_names=["text"], shuffle=False)

print("------------------------before tokenization----------------------------")

for data in dataset.create_dict_iterator(output_numpy=True):
    print(text.to_str(data['text']))

# files from open source repository https://github.com/yanyiwu/cppjieba/tree/master/dict
HMM_FILE = "./datasets/tokenizer/hmm_model.utf8"
MP_FILE = "./datasets/tokenizer/jieba.dict.utf8"
jieba_op = text.JiebaTokenizer(HMM_FILE, MP_FILE)
dataset = dataset.map(operations=jieba_op, input_columns=["text"], num_parallel_workers=1)

print("------------------------after tokenization-----------------------------")

for i in dataset.create_dict_iterator(num_epochs=1, output_numpy=True):
    print(text.to_str(i['text']))
```

The output is as follows:

```text
------------------------before tokenization----------------------------
今天天气太好了我们一起去外面玩吧
------------------------after tokenization-----------------------------
['今天天气' '太好了' '我们' '一起' '去' '外面' '玩吧']
```

### SentencePieceTokenizer

`SentencePieceTokenizer` performs tokenization based on an open-source natural language processing tool package [SentencePiece](https://github.com/google/sentencepiece).

Download the text dataset file `botchan.txt` and place it in the specified location.

```python
download_dataset("https://obs.dualstack.cn-north-4.myhuaweicloud.com/mindspore-website/notebook/datasets/botchan.txt", "./datasets/tokenizer/")
```

```text
./datasets/tokenizer/
└── botchan.txt

0 directories, 1 files
```

The following example builds a text dataset, creates a `vocab` object from the `vocab_file` file, uses `SentencePieceTokenizer` to perform tokenization on the dataset, and displays the text results before and after tokenization.

```python
import mindspore.dataset as ds
import mindspore.dataset.text as text
from mindspore.dataset.text import SentencePieceModel, SPieceTokenizerOutType

input_list = ["I saw a girl with a telescope."]
dataset = ds.NumpySlicesDataset(input_list, column_names=["text"], shuffle=False)

print("------------------------before tokenization----------------------------")

for data in dataset.create_dict_iterator(output_numpy=True):
    print(text.to_str(data['text']))

# file from MindSpore repository https://gitee.com/mindspore/mindspore/blob/master/tests/ut/data/dataset/test_sentencepiece/botchan.txt
vocab_file = "./datasets/tokenizer/botchan.txt"
vocab = text.SentencePieceVocab.from_file([vocab_file], 5000, 0.9995, SentencePieceModel.UNIGRAM, {})
tokenizer_op = text.SentencePieceTokenizer(vocab, out_type=SPieceTokenizerOutType.STRING)
dataset = dataset.map(operations=tokenizer_op)

print("------------------------after tokenization-----------------------------")

for i in dataset.create_dict_iterator(num_epochs=1, output_numpy=True):
    print(text.to_str(i['text']))
```

The output is as follows:

```text
------------------------before tokenization----------------------------
I saw a girl with a telescope.
------------------------after tokenization-----------------------------
['▁I' '▁sa' 'w' '▁a' '▁girl' '▁with' '▁a' '▁te' 'les' 'co' 'pe' '.']
```

### UnicodeCharTokenizer

`UnicodeCharTokenizer` performs tokenization based on the Unicode character set.

The following example builds a text dataset, uses `UnicodeCharTokenizer` to perform tokenization on the dataset, and displays the text results before and after tokenization.

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

The output is as follows:

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

`WhitespaceTokenizer` performs tokenization based on spaces.

The following example builds a text dataset, uses `WhitespaceTokenizer` to perform tokenization on the dataset, and displays the text results before and after tokenization.

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

The output is as follows:

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

`WordpieceTokenizer` performs tokenization based on the word set. A token can be a single word in the word set or a combination of words.

The following example builds a text dataset, creates a `vocab` object from the word list, uses `WordpieceTokenizer` to perform tokenization on the dataset, and displays the text results before and after tokenization.

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

The output is as follows:

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
