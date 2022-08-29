# Text Data Loading and Augmentation

<a href="https://gitee.com/mindspore/docs/blob/master/tutorials/source_en/advanced/dataset/augment_text_data.md" target="_blank"><img src="https://mindspore-website.obs.cn-north-4.myhuaweicloud.com/website-images/master/resource/_static/logo_source_en.png"></a>

As available text data increases, it is urgent to preprocess the text data to obtain clean data required for network training. Text dataset preprocessing usually includes text dataset loading and data augmentation.

Text data can be loaded in the following ways:

1. Use Dataset interfaces such as [ClueDataset](https://www.mindspore.cn/docs/en/master/api_python/dataset/mindspore.dataset.CLUEDataset.html#mindspore.dataset.CLUEDataset) and [TextFileDataset](https://www.mindspore.cn/docs/en/master/api_python/dataset/mindspore.dataset.TextFileDataset.html#mindspore.dataset.TextFileDataset) to read text.
2. Convert the dataset into a standard format (for example, MindRecord) and read the dataset through the corresponding API (for example, [MindDataset](https://www.mindspore.cn/docs/en/master/api_python/dataset/mindspore.dataset.MindDataset.html#mindspore.dataset.MindDataset)).
3. Use the GeneratorDataset interface to receive customized dataset loading functions and load data. For details, see [Custom Dataset Loading](https://www.mindspore.cn/tutorials/en/master/advanced/dataset/custom.html).

## Loading Text Data

The following uses `TextFileDataset` to read data from a TXT file. For more information about loading text datasets, see [API](https://www.mindspore.cn/docs/en/master/api_python/mindspore.dataset.html).

1. Prepare text data. The content is as follows:

    ```text
    Welcome to Beijing
    北京欢迎您！
    我喜欢China!
    ```

2. Create a `tokenizer.txt` file, copy text data to the file, and save the file in ./datasets. Run the following code:

    ```python
    import os

    if not os.path.exists('./datasets'):
        os.mkdir('./datasets')

    # Write the preceding text data to the tokenizer.txt file.
    file_handle = open('./datasets/tokenizer.txt', mode='w')
    file_handle.write('Welcome to Beijing \n北京欢迎您！ \n我喜欢China! \n')
    file_handle.close()
    ```

    After the preceding code is executed, the dataset structure is as follows:

    ```text
    ./datasets
    └── tokenizer.txt
    ```

3. Load the dataset from the TXT file and print it. The code is as follows:

    ```python
    import mindspore.dataset as ds
    import mindspore.dataset.text as text

    # Define the loading path of the text dataset.
    DATA_FILE = './datasets/tokenizer.txt'

    # Load the dataset from the tokenizer.txt file.
    dataset = ds.TextFileDataset(DATA_FILE, shuffle=False)

    for data in dataset.create_dict_iterator(output_numpy=True):
        print(data['text'])
    ```

    ```text
        Welcome to Beijing
        北京欢迎您!
        我喜欢China!
    ```

## Text Data Augmentation

For text data augmentation, common operations include tokenization and vocabulary search.

- Tokenization: Tokenize a long sentence into multiple words.
- Vocabulary search: Search for IDs of tokenized words, and form the IDs into a word vector, and pass the word vector to a network for training.

The following describes the tokenization and vocabulary search functions used in data augmentation. For details about how to use text processing APIs, see [API](https://www.mindspore.cn/docs/en/master/api_python/mindspore.dataset.text.html).

### Building and Using a Vocabulary (Vocab)

A vocab provides the mapping between words and IDs. You can enter a word in the vocab to find its ID or enter an ID to obtain a word.

MindSpore provides multiple methods for building a vocab. You can use the corresponding APIs to obtain original data from dictionaries ([from\_dict](https://www.mindspore.cn/docs/en/master/api_python/dataset_text/mindspore.dataset.text.Vocab.html#mindspore.dataset.text.Vocab.from_dict)), files ([from\_file](https://www.mindspore.cn/docs/en/master/api_python/dataset_text/mindspore.dataset.text.Vocab.html#mindspore.dataset.text.Vocab.from_file)), lists ([from\_list](https://www.mindspore.cn/docs/en/master/api_python/dataset_text/mindspore.dataset.text.Vocab.html#mindspore.dataset.text.Vocab.from_list)), and Dataset objects ([from\_dataset](https://www.mindspore.cn/docs/en/master/api_python/dataset_text/mindspore.dataset.text.Vocab.html#mindspore.dataset.text.Vocab.from_dataset)) to build a vocab.

For example, to use from\_dict to build a vocab, you need to input a dict containing multiple pairs of words and IDs.

```python
from mindspore.dataset import text

# Build a vocab.
vocab = text.Vocab.from_dict({"home": 3, "behind": 2, "the": 4, "world": 5, "<unk>": 6})
```

The vocab provides the [tokens\_to\_ids](https://www.mindspore.cn/docs/en/master/api_python/dataset_text/mindspore.dataset.text.Vocab.html#mindspore.dataset.text.Vocab.tokens_to_ids) and [ids\_to\_tokens](https://www.mindspore.cn/docs/en/master/api_python/dataset_text/mindspore.dataset.text.Vocab.html#mindspore.dataset.text.Vocab.ids_to_tokens) methods for querying words and IDs as follows:

```python
# Search for IDs by word.
ids = vocab.tokens_to_ids(["home", "world"])
print("ids: ", ids)

# Search for words by ID.
tokens = vocab.ids_to_tokens([2, 5])
print("tokens: ", tokens)
```

```text
    ids:  [3, 5]
    tokens:  ['behind', 'world']
```

The preceding results show that:

- The IDs of the words `"home"` and `"world"` are `3` and `5` respectively.
- The word with ID `2` is `"behind"`, and the word with ID is `5` is `"world"`.

This result is also consistent with the vocab. In addition, the vocab is a input parameter required by multiple tokenizers (such as WordpieceTokenizer). During tokenization, if words in a sentence exist in the vocab, the sentence is tokenized into independent words. Then, the corresponding word ID can be obtained by searching the vocab.

### Tokenizers

Tokenization is a process of dividing continuous character sequences into word sequences according to certain specifications. Proper tokenization is helpful for semantic comprehension.

MindSpore provides multiple tokenizers for different purposes, such as BasicTokenizer, BertTokenizer, and JiebaTokenizer, to help users process texts efficiently. Users can build their own dictionaries, use appropriate tokenizers to divide sentences into different tokens, and search for indexes of tokens from the dictionaries. In addition, users can customize tokenizers as required.

> The following describes how to use several common tokenizers. For more information about tokenizers, see [API](https://mindspore.cn/docs/en/master/api_python/mindspore.dataset.text.html).

#### BertTokenizer

`BertTokenizer` calls `BasicTokenizer` and `WordpieceTokenizer` to perform tokenization.

The following example builds a text dataset and a character string list, uses `BertTokenizer` to perform tokenization on the dataset, and displays the text results before and after tokenization.

```python
import mindspore.dataset as ds
import mindspore.dataset.text as text

# Build data to be tokenized.
input_list = ["床前明月光", "疑是地上霜", "举头望明月", "低头思故乡", "I am making small mistakes during working hours",
              "😀嘿嘿😃哈哈😄大笑😁嘻嘻", "繁體字"]

# Load the text dataset.
dataset = ds.NumpySlicesDataset(input_list, column_names=["text"], shuffle=False)

print("------------------------before tokenization----------------------------")
for data in dataset.create_dict_iterator(output_numpy=True):
    print(data['text'])
```

```text
    ------------------------before tokenization----------------------------
    床前明月光
    疑是地上霜
    举头望明月
    低头思故乡
    I am making small mistakes during working hours
    😀嘿嘿😃哈哈😄大笑😁嘻嘻
    繁體字
```

The preceding example shows the data before tokenization. The following uses the tokenizer `BertTokenizer` to tokenize the dataset.

```python
# Build a vocab.
vocab_list = [
    "床", "前", "明", "月", "光", "疑", "是", "地", "上", "霜", "举", "头", "望", "低", "思", "故", "乡",
    "繁", "體", "字", "嘿", "哈", "大", "笑", "嘻", "i", "am", "mak", "make", "small", "mistake",
    "##s", "during", "work", "##ing", "hour", "😀", "😃", "😄", "😁", "+", "/", "-", "=", "12",
    "28", "40", "16", " ", "I", "[CLS]", "[SEP]", "[UNK]", "[PAD]", "[MASK]", "[unused1]", "[unused10]"]

# Load the vocab.
vocab = text.Vocab.from_list(vocab_list)

# Use BertTokenizer to tokenize the text dataset.
tokenizer_op = text.BertTokenizer(vocab=vocab)
dataset = dataset.map(operations=tokenizer_op)

print("------------------------after tokenization-----------------------------")
for i in dataset.create_dict_iterator(num_epochs=1, output_numpy=True):
    print(i['text'])
```

```text
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

According to the preceding two results,  `BertTokenizer` tokenizes sentences, words, and emojis in the dataset based on words in the vocab as the minimum unit. "故乡" is divided into "故" and "乡". "明月" is divided into "明" and "月". It should be noted that " mistakes" is divided into "mistake" and "##s".

#### JiebaTokenizer

`JiebaTokenizer` performs Chinese tokenization based on Jieba.

The following sample code is used to download and save the dictionary files `hmm_model.utf8` and `jieba.dict.utf8` to the specified location.

```python
from mindvision.dataset import DownLoad

# Directory for storing the dictionary file
dl_path = "./dictionary"

# Obtains the dictionary file source.
dl_url_hmm = "https://obs.dualstack.cn-north-4.myhuaweicloud.com/mindspore-website/notebook/datasets/hmm_model.utf8"
dl_url_jieba = "https://obs.dualstack.cn-north-4.myhuaweicloud.com/mindspore-website/notebook/datasets/jieba.dict.utf8"

# Download the dictionary file.
dl = DownLoad()
dl.download_url(url=dl_url_hmm, path=dl_path)
dl.download_url(url=dl_url_jieba, path=dl_path)
```

The directory structure for storing downloaded files is as follows:

```text
./dictionary/
├── hmm_model.utf8
└── jieba.dict.utf8
```

The following example builds a text dataset, uses the HMM and MP dictionary files to create a `JiebaTokenizer` object, performs tokenization on the dataset, and displays the text results before and after tokenization.

```python
import mindspore.dataset as ds
import mindspore.dataset.text as text

# Build data to be tokenized.
input_list = ["明天天气太好了我们一起去外面玩吧"]

# Load the dataset.
dataset = ds.NumpySlicesDataset(input_list, column_names=["text"], shuffle=False)

print("------------------------before tokenization----------------------------")
for data in dataset.create_dict_iterator(output_numpy=True):
    print(data['text'])
```

```text
    ------------------------before tokenization----------------------------
    明天天气太好了我们一起去外面玩吧
```

The preceding example shows the data before tokenization. The following uses the tokenizer `JiebaTokenizer` to tokenize the dataset.

```python
HMM_FILE = "./dictionary/hmm_model.utf8"
MP_FILE = "./dictionary/jieba.dict.utf8"

# Use the JiebaTokenizer to tokenize the dataset.
jieba_op = text.JiebaTokenizer(HMM_FILE, MP_FILE)
dataset = dataset.map(operations=jieba_op, input_columns=["text"], num_parallel_workers=1)

print("------------------------after tokenization-----------------------------")
for data in dataset.create_dict_iterator(num_epochs=1, output_numpy=True):
    print(data['text'])
```

```text
    ------------------------after tokenization-----------------------------
    ['明天' '天气' '太好了' '我们' '一起' '去' '外面' '玩吧']
```

According to the preceding two results, `JiebaTokenizer` tokenizes sentences in the dataset based on words as the minimum unit.

#### SentencePieceTokenizer

`SentencePieceTokenizer` is encapsulated based on the open-source NLP toolkit [SentencePiece](https://github.com/google/sentencepiece).

The following sample code is used to download and save the text dataset file `botchan.txt` to the specified location.

```python
# Dataset storage location
dl_path = "./datasets"

# Obtain the corpus data source.
dl_url_botchan = "https://obs.dualstack.cn-north-4.myhuaweicloud.com/mindspore-website/notebook/datasets/botchan.txt"

# Download corpus data.
dl.download_url(url=dl_url_botchan, path=dl_path)
```

The directory structure for storing downloaded files is as follows:

```text
./datasets/
└── botchan.txt
```

The following example builds a text dataset, creates a `vocab` object from the `vocab_file` file, uses `SentencePieceTokenizer` to perform tokenization on the dataset, and displays the text results before and after tokenization.

```python
import mindspore.dataset as ds
import mindspore.dataset.text as text
from mindspore.dataset.text import SentencePieceModel, SPieceTokenizerOutType

# Build the data to be tokenized.
input_list = ["Nothing in the world is difficult for one who sets his mind on it."]

# Load the dataset.
dataset = ds.NumpySlicesDataset(input_list, column_names=["text"], shuffle=False)

print("------------------------before tokenization----------------------------")
for data in dataset.create_dict_iterator(output_numpy=True):
    print(data['text'])
```

```text
    ------------------------before tokenization----------------------------
    Nothing in the world is difficult for one who sets his mind on it.
```

The preceding example shows the data before tokenization. The following uses the tokenizer `SentencePieceTokenizer` to tokenize the dataset.

```python
# Path for storing corpus data files
vocab_file = "./datasets/botchan.txt"

# Learn to build vocab from corpus data.
vocab = text.SentencePieceVocab.from_file([vocab_file], 5000, 0.9995, SentencePieceModel.WORD, {})

# Use SentencePieceTokenizer to tokenize the dataset.
tokenizer_op = text.SentencePieceTokenizer(vocab, out_type=SPieceTokenizerOutType.STRING)
dataset = dataset.map(operations=tokenizer_op)

print("------------------------after tokenization-----------------------------")
for i in dataset.create_dict_iterator(num_epochs=1, output_numpy=True):
    print(i['text'])
```

```text
    ------------------------after tokenization-----------------------------
    ['▁Nothing' '▁in' '▁the' '▁world' '▁is' '▁difficult' '▁for' '▁one' '▁who'
     '▁sets' '▁his' '▁mind' '▁on' '▁it.']
```

According to the preceding two results, `SentencePieceTokenizer` tokenizes sentences in the dataset based on words as the minimum unit. During tokenization, `SentencePieceTokenizer` processes spaces as common symbols and uses underscores to replace spaces.

#### UnicodeCharTokenizer

`UnicodeCharTokenizer` performs tokenization based on the Unicode character set.

The following example builds a text dataset, uses `UnicodeCharTokenizer` to perform tokenization on the dataset, and displays the text results before and after tokenization.

```python
import mindspore.dataset as ds
import mindspore.dataset.text as text

# Build the data to be tokenized.
input_list = ["Welcome to Beijing!", "北京欢迎您！", "我喜欢China!"]

# Load the dataset.
dataset = ds.NumpySlicesDataset(input_list, column_names=["text"], shuffle=False)

print("------------------------before tokenization----------------------------")
for data in dataset.create_dict_iterator(output_numpy=True):
    print(data['text'])
```

```text
    ------------------------before tokenization----------------------------
    Welcome to Beijing!
    北京欢迎您！
    我喜欢China!
```

The preceding example shows the data before tokenization. The following uses the tokenizer `UnicodeCharTokenizer` to tokenize the dataset.

```python
# Use UnicodeCharTokenizer to tokenize the dataset.
tokenizer_op = text.UnicodeCharTokenizer()
dataset = dataset.map(operations=tokenizer_op)

print("------------------------after tokenization-----------------------------")
for data in dataset.create_dict_iterator(num_epochs=1, output_numpy=True):
    print(data['text'].tolist())
```

```text
    ------------------------after tokenization-----------------------------
    ['W', 'e', 'l', 'c', 'o', 'm', 'e', ' ', 't', 'o', ' ', 'B', 'e', 'i', 'j', 'i', 'n', 'g', '!']
    ['北', '京', '欢', '迎', '您', '！']
    ['我', '喜', '欢', 'C', 'h', 'i', 'n', 'a', '!']
```

According to the preceding two results, `UnicodeCharTokenizer` tokenizes sentences in the dataset based on Chinese characters or letters as the minimum unit.

#### WhitespaceTokenizer

`WhitespaceTokenizer` performs tokenization based on spaces.

The following example builds a text dataset, uses `WhitespaceTokenizer` to perform tokenization on the dataset, and displays the text results before and after tokenization.

```python
import mindspore.dataset as ds
import mindspore.dataset.text as text

# Build the data to be tokenized.
input_list = ["Welcome to Beijing!", "北京欢迎您！", "我喜欢China!", "床前明月光，疑是地上霜。"]

# Load the dataset.
dataset = ds.NumpySlicesDataset(input_list, column_names=["text"], shuffle=False)

print("------------------------before tokenization----------------------------")
for data in dataset.create_dict_iterator(output_numpy=True):
    print(data['text'])
```

```text
    ------------------------before tokenization----------------------------
    Welcome to Beijing!
    北京欢迎您！
    我喜欢China!
    床前明月光，疑是地上霜。
```

The preceding example shows the data before tokenization. The following uses the tokenizer `WhitespaceTokenizer` to tokenize the dataset.

```python
# Use WhitespaceTokenizer to tokenize the dataset.
tokenizer_op = text.WhitespaceTokenizer()
dataset = dataset.map(operations=tokenizer_op)

print("------------------------after tokenization-----------------------------")
for i in dataset.create_dict_iterator(num_epochs=1, output_numpy=True):
    print(i['text'].tolist())
```

```text
    ------------------------after tokenization-----------------------------
    ['Welcome', 'to', 'Beijing!']
    ['北京欢迎您！']
    ['我喜欢China!']
    ['床前明月光，疑是地上霜。']
```

According to the preceding two results, `WhitespaceTokenizer` tokenizes sentences in the dataset based on spaces.

#### WordpieceTokenizer

`WordpieceTokenizer` performs tokenization based on a vocab which contains words and morphemes.

The following example builds a text dataset, creates a `vocab` object from the word list, uses `WordpieceTokenizer` to perform tokenization on the dataset, and displays the text results before and after tokenization.

```python
import mindspore.dataset as ds
import mindspore.dataset.text as text

# Build the data to be tokenized.
input_list = ["My", "favorite", "book", "is", "love", "during", "the", "cholera", "era", ".", "what",
              "我", "最", "喜", "欢", "的", "书", "是", "霍", "乱", "时", "期", "的", "爱", "情", "。", "好"]

# Build an English vocab.
vocab_english = ["book", "cholera", "era", "favor", "##ite", "My", "is", "love", "dur", "##ing", "the", "."]

# Build a Chinese vocab.
vocab_chinese = ['我', '最', '喜', '欢', '的', '书', '是', '霍', '乱', '时', '期', '爱', '情', '。']

# Load the dataset.
dataset = ds.NumpySlicesDataset(input_list, column_names=["text"], shuffle=False)

print("------------------------before tokenization----------------------------")
for data in dataset.create_dict_iterator(output_numpy=True):
    print(data['text'])
```

```text
    ------------------------before tokenization----------------------------
    My
    favorite
    book
    is
    love
    during
    the
    cholera
    era
    .
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
    。
    好
```

The preceding result shows data before tokenization. The words "what" and "好" that are not listed in the vocab. The following uses the tokenizer `WordpieceTokenizer` to tokenize the dataset.

```python
# Use WordpieceTokenizer to tokenize the dataset.
vocab = text.Vocab.from_list(vocab_english+vocab_chinese)
tokenizer_op = text.WordpieceTokenizer(vocab=vocab)
dataset = dataset.map(operations=tokenizer_op)

print("------------------------after tokenization-----------------------------")
for i in dataset.create_dict_iterator(num_epochs=1, output_numpy=True):
    print(i['text'])
```

```text
    ------------------------after tokenization-----------------------------
    ['My']
    ['favor' '##ite']
    ['book']
    ['is']
    ['love']
    ['dur' '##ing']
    ['the']
    ['cholera']
    ['era']
    ['.']
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
    ['。']
    ['[UNK]']
```

According to the preceding two results, `WordpieceTokenizer` tokenizes words in the dataset based on the vocab. "My" and "love" remain the same. It should be noted that "favorite" is tokenized into "favor" and "##ite". Because "word" and "good" are not listed in the vocab, they are displayed as \[UNK].
