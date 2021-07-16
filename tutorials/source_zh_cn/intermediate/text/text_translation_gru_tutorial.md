# 序列到序列（seq2seq）模型实现文本翻译

<a href="https://gitee.com/mindspore/docs/tree/master/tutorials/source_zh_cn/intermediate/text/text_translation_gru_tutorial.md" target="_blank"><img src="https://gitee.com/mindspore/docs/raw/master/resource/_static/logo_source.png"></a>

本教程通过著名数据集Multi30k来训练序列到序列（seq2seq）模型，以此实现将德语句子翻译成英语的功能。该教程讲解如下知识：

* 如何将句子预处理为向量用于NLP建模。
* 了解并搭建一个序列到序列（seq2seq）网络。
* 通过`MindDataset`读取数据并进行模型训练。

通过该教程将学会MindSpore搭建nlp模型所需要的全部流程，下面将逐步解释实现过程。

## 数据处理

本教程采用Multi30k数据集，该数据集由大约30000个对应的英语、德语和法语句子组成，每个句子包含约12个单词。详细内容可参考[WMT16官网](http://www.statmt.org/wmt16/multimodal-task.html)。实验过程中用其来生成.tok文件和两个vocab文件，并将其分别命名为vocab.de和vocab.en方便辨识。

运行本教程所需的[脚本](https://mindspore-website.obs.cn-north-4.myhuaweicloud.com/notebook/source-codes/intermediate/train.py)及[数据集](https://mindspore-website.obs.cn-north-4.myhuaweicloud.com/notebook/datasets/intermediate/translation.rar)，在教程的同级目录下新建 `nltk_predata` 文件夹，将下载好的数据集按照文件结构放到相应目录下。

```text
project
│ text_translation_gru_tutorial.md
| text_translation_gru_tutorial.py
└─nltk_predata
│   train.de
│   test.de
│   val.de
│   train.en
│   test.en
│   val.en
```

`MindSpore` 拥有的工具函数可以创建轻松迭代的数据集，以此训练语言翻译模型。在此教程中，我们将展示如何对原始文本句子进行标记，构建词汇表以及将标记数字化为向量。

注意：本教程中的分词需要自然语言工具包 `nltk` ，我们使用 `nltk` 是因为它为英语以外的其他语言的分词提供了强大的支持。 `nltk` 提供了 `english` 标记器，并支持其他英语标记器（例如 `german`

），所以对于需要多种语言的翻译情景，nltk是您的最佳选择之一。

要运行本教程，请先使用 `pip` 或 `conda` 安装 `nltk` 。接下来，下载英语和德语nltk分词器的原始数据：

```python
pip install nltk
import nltk
nltk.download()
```

> 如果 `nltk` 下载失败，[点击链接](https://github.com/nltk/nltk_data) 手动下载。

在数据处理前，首先引入处理数据所需环境：

```python
import os
import argparse
from collections import Counter

import nltk
from nltk.tokenize import word_tokenize
```

### 创建标记句子文件方法

为了之后更好地对句子进行区分和翻译，我们定义 `create_tokenized_sentences` 函数用来去除文本中的标志符，并将文本进行统一小写处理，最后将处理后的句子分词后拼接到新的文件中。

```python
def create_tokenized_sentences(input_files, language)
    """ 创建带有标记的句子文件 """
    sentence = []
    total_lines = open(input_files, "r").read().splitlines()

    # 逐句处理
    for line in total_lines:
        line = line.strip('\r\n ')
        line = line.lower()
        # 根据语言进行分词
        tokenize_sentence = word_tokenize(line, language)
        str_sentence = " ".join(tokenize_sentence)
        sentence.append(str_sentence)

    # 写入到.tok文件中
    tokenize_file = input_files + ".tok"
    f = open(tokenize_file, "w")
    for line in sentence:
        f.write(line)
        f.write("\n")
    f.close()
```

### 创建定义词频表方法

定义 `get_dataset_vocab` 函数， `text_file` 为需要统计词频的文件， `vacab_file` 为生成词汇表的名称，该函数主要实现了如下功能：

* 统计每个句子中词语的频率。
* 将标志符写入词频表。
* 按照词语频率将词语写入词汇表。

```python
def get_dataset_vocab(text_file, vocab_file):
    """ 创建词汇表 """
    counter = Counter()
    text_lines = open(text_file, "r").read().splitlines()

    # 统计每个词的出现频率
    for line in text_lines:
        for word in line.strip('\r\n ').split(' '):
            if word:
                counter[word] += 1
        vocab = open(vocab_file, "w")
        # 将标志符写入词频表中
        basic_label = ["<unk>", "<pad>", "<sos>", "<eos>"]

    # 在每行后加入换行符
    for label in basic_label:
        vocab.write(label + "\n")

    # 按照词的频率排序后写入词频表文件
    for key, f in sorted(counter.items(), key=lambda x: x[1], reverse=True):
        if f < 2:
            continue
        vocab.write(key + "\n")
    vocab.close()
```

### 定义合并数据集方法保留全部数据

Multi30k数据集不存在总数据集，为了以后拆分数据和数据安全，通过定义 `merge_text` 函数将全部数据以新数据 `all_de.tok` ， `all_en.tok` 的形式处理保存。

```python
def merge_text(root_dir, file_list, output_file):
    """ 合并文本文件 """
    output_file = os.path.join(root_dir, output_file)
    f_output = open(output_file, "w")

    # 写入tok文件
    for file_name in file_list:
        text_path = os.path.join(root_dir, file_name) + ".tok"
        f = open(text_path)
        f_output.write(f.read() + "\n")
    f_output.close()
```

通过遍历将所需文件进行所有文本文件的预处理，合并 `merge_text` 来生成全部文本集，并通过 `get_dataset_vocab` 来创建两个所需文件的词汇表。

```python
dataset_path = "./nltk_predata"
src_file_list = ["train.de", "test.de", "val.de"]
dst_file_list = ["train.en", "test.en", "val.en"]

# 生成预处理文件
for file in src_file_list:
    file_path = os.path.join(dataset_path, file)
    create_tokenized_sentences(file_path, "english")
for file in dst_file_list:
    file_path = os.path.join(dataset_path, file)
    create_tokenized_sentences(file_path, "german")

src_all_file = "all.de.tok"
dst_all_file = "all.en.tok"
# 保留全部文本的数据集
merge_text(dataset_path, src_file_list, src_all_file)
merge_text(dataset_path, dst_file_list, dst_all_file)
src_vocab = os.path.join(dataset_path, "vocab.de")
dst_vocab = os.path.join(dataset_path, "vocab.en")
# 生成词频表
get_dataset_vocab(os.path.join(dataset_path, src_all_file), src_vocab)
get_dataset_vocab(os.path.join(dataset_path, dst_all_file), dst_vocab)
```

### 生成MindRecord数据

至此我们得到了所需的所有数据，现在我们开始讲解如何利用 `MindSpore` 将数据进行预处理和转换MindRecord格式数据。

首先引入后续所要用的环境变量。

```python
import time
import ast
import collections
import logging
import math
import sys
import unicodedata

import mindspore.dataset as ds
from mindspore import Tensor, Parameter
from mindspore.context import ParallelMode
from mindspore import context
from mindspore import set_seed
from mindspore.communication import init
from mindspore import dtype as mstype

import numpy as np
```

在运行实验前定义用到的模型参数与环境参数，本实验仅能在Ascend环境中运行。

```python
set_seed(1)
context.set_context(mode=context.GRAPH_MODE, device_target="Ascend", device_id=0, save_graphs=False)
```

### 创建文本打印方法

为防止数据读入中因为不同类型的数据而导致的报错，定义 `convert_to_unicode` 方法，在数据读进去前判断数据类型和Python版本，如果是字符串则正常读入，如果不是则以无视错误的 `utf-8` 方式解码。

```python
def convert_to_unicode(text):
    """
    将文本转换为Unicode的代码格式
    """
    # 判断python版本
    if sys.version_info[0] == 3:
        # 判断文本类型
        if isinstance(text, str):
            return text
        if isinstance(text, bytes):
            return text.decode("utf-8", "ignore")
        raise ValueError("Only support type `str` or `bytes`, while text type is `%s`" % (type(text)))

    if sys.version_info[0] == 2:
        # 判断文本类型
        if isinstance(text, str):
            return text.decode("utf-8", "ignore")
        if isinstance(text, unicode):
            return text
        raise ValueError("Only support type `str` or `unicode`, while text type is `%s`" % (type(text)))
    raise ValueError("Only supported when running on Python2 or Python3.")
```

### 读取vocab文件并转换为字典

定义 `laod_vocab_file` 函数来读取输入参数 `vocab_file` 词频表文件，将其转换为 `{token:id}` 字典格式。

```python
def load_vocab_file(vocab_file):
    # 定义有序字典
    vocab_dict = collections.OrderedDict()
    index = 0

    # 写入字典
    with open(vocab_file, "r") as vocab:
        while True:
            token = convert_to_unicode(vocab.readline())
            if not token:
                break
            token = token.strip()
            vocab_dict[token] = index
            index += 1
    return vocab_dict
```

### 将vocab字典转换为序列

词频表通过 `load_vocab_file` 处理后 `{token:id}` 格式，将通过 `conver_by_vocab_dict` 函数，再次转换为 `[tokens|ids]` 格式。

对于频率过低而不存在于词频表中的词，统一用标志位 `<unk>` 来标记。

```python
def convert_by_vocab_dict(vocab_dict, items):
    UNK = "<unk>"
    output = []

    for item in items:
        if item in vocab_dict:
            output.append(vocab_dict[item])
        else:
            output.append(vocab_dict[UNK])
    return output
```

## 定义空格分词器

通过定义 `WhiteSpaceTokenizer` 类来封装成一个利用空格实现分词的分词器。为了更有效的对文本进行切分，我们需要对文本中的控制符也当做空格来处理，该类包括五个大的部分： `_is_whitespace_char` ， `_is_control_char` ， `_clean_text` ， `_whitespace_tokenize` ， `tokenize` 。分别负责的功能如下：

* `_is_whitespace_char` ：提供控制符判定为空格功能。

* `_is_control_char` ：提供判断是否为控制符功能。

* `_clean_text` ：提供清理无效字符并清理空格功能。

* `_whitespace_tokenize` ：提供空格切分功能。

* `tokenize` ：通过拼合上述函数来实现分词pipeline。

```python
class WhiteSpaceTokenizer():
    """
    空格分词器
    """

    def __init__(self, vocab_file):
        self.vocab_dict = load_vocab_file(vocab_file)
        self.inv_vocab_dict = {index: token for token, index in self.vocab_dict.items()}

    def _is_whitespace_char(self, char):
        """
        检查是否是空格(替换 "\t", "\n", "\r" 当做空格).
        """
        if char in (" ", "\t", "\n", "\r"):
            return True

        uni = unicodedata.category(char)

        if uni == "Zs":
            return True
        return False

    def _is_control_char(self, char):
        """
        检查是否为控制符
        """
        if char in ("\t", "\n", "\r"):
            return False

        uni = unicodedata.category(char)

        if uni in ("Cc", "Cf"):
            return True
        return False

    def _clean_text(self, text):
        """
        删除错误字符并删除多余空格
        """
        output = []

        for char in text:
            cp = ord(char)
            if cp == 0 or cp == 0xfffd or self._is_control_char(char):
                continue
            if self._is_whitespace_char(char):
                output.append(" ")
            else:
                output.append(char)
        return "".join(output)

    def _whitespace_tokenize(self, text):
        """
        按照空格切分字符并删除做操作的空格
        """
        text = text.strip()
        text = text.lower()

        if text.endswith("."):
            text = text.replace(".", " .")
        if not text:
            tokens = []
        else:
            tokens = text.split()
        return tokens

    def tokenize(self, text):
        """
        切分文本pipline
        """
        text = convert_to_unicode(text)
        text = self._clean_text(text)
        tokens = self._whitespace_tokenize(text)
        return tokens

    def convert_tokens_to_ids(self, tokens):
        return convert_by_vocab_dict(self.vocab_dict, tokens)

    def convert_ids_to_tokens(self, ids):
        return convert_by_vocab_dict(self.inv_vocab_dict, ids)
```

## 创建单个实例

`WhiteSpaceTokenizer` 让我们拥有了一个完整的分词器，现在定义 `SampleInstance` 方法来验证我们是否完成了上述功能的搭建。

```python
class SampleInstance():
    """单个句子对示例"""

    def __init__(self, source_tokens, target_tokens):
        self.source_tokens = source_tokens
        self.target_tokens = target_tokens

    def __str__(self):
        s = ""
        # 拼接分词
        s += "source_tokens: %s\n" % (" ".join(
            [tokenization.convert_to_unicode(x) for x in self.source_tokens]))
        s += "target tokens: %s\n" % (" ".join(
            [tokenization.convert_to_unicode(x) for x in self.target_tokens]))
        s += "\n"
        return s

    def __repr__(self):
        return self.__str__()
```

### 定义训练实例方法和提取训练特征方法

现在可以通过定义一个 `create_training_instance` 方法为语言文本增加开始和结束符，并通过 `SampleInstance` 类来创建一个训练的实例，该方法需要传递四个参数：

* source_words：原始语言语句。
* target_words：翻译目标语言。
* max_seq_length：最大序列长度。
* clip_to_max_len：确认计算是否合理。

```python
def create_training_instance(source_words, target_words, max_seq_length, clip_to_max_len):
    """为句子创建句子对示例."""
    EOS = "<eos>"
    SOS = "<sos>"
    # 判断输入原始文本是否超过最长字节
    if len(source_words) >= max_seq_length - 1 or len(target_words) >= max_seq_length - 1:
        if clip_to_max_len:
            source_words = source_words[:min([len(source_words, max_seq_length - 2)])]
            target_words = target_words[:min([len(target_words, max_seq_length - 2)])]
        else:
            return None
    # 为句子增加开始符和结束符
    source_tokens = [SOS] + source_words + [EOS]
    target_tokens = [SOS] + target_words + [EOS]
    # 实例化实例并返回输出
    instance = SampleInstance(
        source_tokens=source_tokens,
        target_tokens=target_tokens)
    return instance
```

现在来定义 `get_instance_features` 方法来提取实例语句中的特征，该方法需要提供如下参数：

* instance：实例对象。
* tokenizer_src/tokenizer_trg：源/目的语言标记文件。
* max_seq_length：最大序列长度。
* bucket：分块数。

```python
def get_instance_features(instance, tokenizer_src, tokenizer_trg, max_seq_length, bucket):
    """获取实例特征"""

    def _find_bucket_length(source_tokens, target_tokens):
        # 为文本寻找适合长度的bucket
        source_ids = tokenizer_src.convert_tokens_to_ids(source_tokens)
        target_ids = tokenizer_trg.convert_tokens_to_ids(target_tokens)
        num = max(len(source_ids), len(target_ids))
        assert num <= bucket[-1]
        for index in range(1, len(bucket)):
            if bucket[index - 1] < num <= bucket[index]:
                return bucket[index]
        return bucket[0]

    # 转换为词频表
    def _convert_ids_and_mask(tokenizer, input_tokens, seq_max_bucket_length):
        input_ids = tokenizer.convert_tokens_to_ids(input_tokens)
        input_mask = [1] * len(input_ids)
        assert len(input_ids) <= max_seq_length

        while len(input_ids) < seq_max_bucket_length:
            input_ids.append(1)
            input_mask.append(0)

        assert len(input_ids) == seq_max_bucket_length
        assert len(input_mask) == seq_max_bucket_length

        return input_ids, input_mask

    # 匹配文本长度，并将原始文本和目的文本分词
    seq_max_bucket_length = _find_bucket_length(instance.source_tokens, instance.target_tokens)
    source_ids, source_mask = _convert_ids_and_mask(tokenizer_src, instance.source_tokens, seq_max_bucket_length)
    target_ids, target_mask = _convert_ids_and_mask(tokenizer_trg, instance.target_tokens, seq_max_bucket_length)

    # 建立特征有序字典保存处理后的特征
    features = collections.OrderedDict()
    features["source_ids"] = np.asarray(source_ids)
    features["source_mask"] = np.asarray(source_mask)
    features["target_ids"] = np.asarray(target_ids)
    features["target_mask"] = np.asarray(target_mask)

    return features, seq_max_bucket_length
```

## 配置路径文件并生产可转换数据

上述已经定义完毕所有处理文件所必须的函数，接下来我们配置所需处理的文件路径，路径参数如下：

* input_file：数据集路径。
* output_file：保存MindRecord文件路径。
* src_file/trg_file：源/目的语言对照词汇表。

```python
input_file = "nltk_predata/"
output_file = "nltk_mindrecord/mindrecord"
src_file = "nltk_predata/vocab.de"
trg_file = "nltk_predata/vocab.en"
outputs_dir = './output_data/'
ckpt_path = 'outputs/'
```

现在通过读取路径内容，通过输出示例来确认输出是否为我们之后需要处理的内容。

```python
from mindspore.mindrecord import FileWriter

logging.basicConfig(level=logging.INFO)
# 读取标记文件
tokenizer_src = WhiteSpaceTokenizer(src_file)
tokenizer_trg = WhiteSpaceTokenizer(trg_file)
input_files = ["nltk_predata/train.de.tok", "nltk_predata/train.en.tok"]
logging.info("*** Write to output files ***")
logging.info("  %s", output_file)
# 用来记录总数据和记录数量
total_written = 0
total_read = 0
feature_dict = {}

for i in ast.literal_eval:
    feature_dict[i] = []

for input_file in input_files:
    # 开始读取文件
    logging.info("*** Reading from   %s ***", input_file)
    with open(input_file, "r") as reader:
        while True:
            line = convert_to_unicode(reader.readline())
            if not line:
                break
            total_read += 1
            if total_read % 100000 == 0:
                logging.info("Read %d ...", total_read)
            if line.strip() == "":
                continue
            source_tokens = tokenizer_src.tokenize(line)
            target_tokens = tokenizer_trg.tokenize(line)

            # 判断词条长度，如果过长报错
            if len(source_tokens) >= 32 or len(target_tokens) >= 32:
                print(len(target_tokens), 32)
                logging.info("ignore long sentence!")
                continue
            # 创建样例
            instance = create_training_instance(source_tokens, target_tokens, 32,
                                                clip_to_max_len=ast.literal_eval)
            if instance is None:
                continue
            features, seq_max_bucket_length = get_instance_features(instance, tokenizer_src, tokenizer_trg,
                                                                    32, ast.literal_eval)
            # 如果样例满足长度便放入输出样例列表
            for key in feature_dict:
                if key == seq_max_bucket_length:
                    feature_dict[key].append(features)
            # 输出10个实例
            if total_read <= 10:
                logging.info("*** Example ***")
                logging.info("source tokens: %s", " ".join(
                    [convert_to_unicode(x) for x in instance.source_tokens]))
                logging.info("target tokens: %s", " ".join(
                    [convert_to_unicode(x) for x in instance.target_tokens]))
                for feature_name in features.keys():
                    feature = features[feature_name]
                    logging.info("%s: %s", feature_name, feature)
```

输出：

```python
INFO: root: ** *Write
to
output
files ** *
INFO: root: nltk_mindrecord / mindrecord
INFO: root: ** *Reading
from nltk_predata / train.de.tok ** *
INFO: root: ** *Example ** *
INFO: root:source
tokens: zwei
junge
weiße
männer
sind
im
freien in der
nähe
vieler
büsche.
INFO: root:target
tokens: zwei
junge
weiße
männer
sind
im
freien in der
nähe
vieler
büsche.
INFO: root:source_ids: [2 18 26 247 30 85 20 84 7 15 115 5633 3245 4
                        3 1 1 1 1 1 1 1 1 1 1 1 1 1
                        1 1 1 1]
INFO: root:source_mask: [1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0]
INFO: root:target_ids: [2 0 0 0 0 0 0 0 6 0 0 0 0 5 3 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1]
INFO: root:target_mask: [1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0]
INFO: root: ** *Example ** *
INFO: root:source
tokens: mehrere
männer
mit
schutzhelmen
bedienen
ein
antriebsradsystem.
INFO: root:target
tokens: mehrere
männer
mit
schutzhelmen
bedienen
ein
antriebsradsystem.
INFO: root:source_ids: [2 76 30 11 859 2163 5 0 4 3 1 1 1 1
                        1 1 1 1 1 1 1 1 1 1 1 1 1 1
                        1 1 1 1]
INFO: root:source_mask: [1 1 1 1 1 1 1 1 1 1 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0]
INFO: root:target_ids: [2 0 0 0 0 0 0 0 5 3 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1]
INFO: root:target_mask: [1 1 1 1 1 1 1 1 1 1 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0]
INFO: root: ** *Example ** *
INFO: root:source
tokens: ein
kleines
mädchen
klettert in ein
spielhaus
aus
holz.
INFO: root:target
tokens: ein
kleines
mädchen
klettert in ein
spielhaus
aus
holz.
INFO: root:source_ids: [2 5 66 25 218 7 5 5634 54 515 4 3 1 1
                        1 1 1 1 1 1 1 1 1 1 1 1 1 1
                        1 1 1 1]
INFO: root:source_mask: [1 1 1 1 1 1 1 1 1 1 1 1 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0]
INFO: root:target_ids: [2 0 0 0 0 6 0 0 0 0 5 3 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1]
INFO: root:target_mask: [1 1 1 1 1 1 1 1 1 1 1 1 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0]
```

在本案例中，将非标准的数据集和常用的数据集转换为MindSpore数据格式，即MindRecord，从而方便地加载到MindSpore中进行训练。

通过上述输出可以看到，内容正是本教程所需要的, 接下来要做的便是将数据存储为方便操作的MindRecord文件。循环存储不同数据长度的 `bucket` , 拼接对应的MindRecord文件名，将数据按照需要类型写入到文件中。

```python
# 定义保存MindRecord文件格式
num_splits = 1

for i in ast.literal_eval:
    if num_splits == 1:
        output_file_name = output_file + '_' + str(i)
    else:
        output_file_name = output_file + '_' + str(i) + '_'
    writer = FileWriter(output_file_name, num_splits)
    # 定义写入格式
    data_schema = {"source_ids": {"type": "int64", "shape": [-1]},
                   "source_mask": {"type": "int64", "shape": [-1]},
                   "target_ids": {"type": "int64", "shape": [-1]},
                   "target_mask": {"type": "int64", "shape": [-1]}
                   }
    writer.add_schema(data_schema, "gru")
    features_ = feature_dict[i]
    # 写入
    logging.info("Bucket length %d has %d samples, start writing...", i, len(features_))
    for item in features_:
        writer.write_raw_data([item])
        total_written += 1
    writer.commit()
logging.info("Wrote %d total instances", total_written)
```

## 数据加载

### Teacher Forcing 机制

所谓Teacher Forcing，就是在学习时跟着老师(ground truth)走。
这是一种网络训练方法，对于开发用于机器翻译，文本摘要，图像字幕的深度学习语言模型以及许多其余应用程序相当重要。训练过程中的每个时刻，有一定概率使用上一时刻的输出作为输入，也有一定概率使用正确的 target 作为输入。

通过定义 `random_teacher_force` 函数来为我们的数据集添加Teacher Forcing机制。

```python
def random_teacher_force(source_ids, target_ids, target_mask):
    teacher_force = np.random.random() < 0.5
    teacher_force_array = np.array([teacher_force], dtype=bool)
    return source_ids, target_ids, teacher_force_array
```

接下来使用 `MindDataset` 来为MindSpore加载 `MindRecord` 数据集。

```python
import mindspore.dataset.transforms.c_transforms as deC

# 将数据转成合理数据集
def create_gru_dataset(epoch_count=1, batch_size=1, rank_size=1, rank_id=0, do_shuffle=True, dataset_path=None,
                       is_training=True):
    """创建数据集"""
    dataset = ds.MindDataset(dataset_path,
                             columns_list=["source_ids", "target_ids",
                                           "target_mask"],
                             shuffle=do_shuffle, num_parallel_workers=10, num_shards=rank_size, shard_id=rank_id)
    # 实例化Teacher Forcing
    operations = random_teacher_force
    dataset = dataset.map(operations=operations, input_columns=["source_ids", "target_ids", "target_mask"],
                          output_columns=["source_ids", "target_ids", "teacher_force"],
                          column_order=["source_ids", "target_ids", "teacher_force"])
    type_cast_op = deC.TypeCast(mstype.int32)
    type_cast_op_bool = deC.TypeCast(mstype.bool_)
    # 生成数据集
    dataset = dataset.map(operations=type_cast_op, input_columns="source_ids")
    dataset = dataset.map(operations=type_cast_op, input_columns="target_ids")
    dataset = dataset.map(operations=type_cast_op_bool, input_columns="teacher_force")
    dataset = dataset.batch(batch_size, drop_remainder=True)
    dataset = dataset.repeat(1)
    return dataset
```

### 读取数据

通过实例化 `create_gru_dataset` 来读取生成的MindRecord数据，可以打印出来确认数据集的大小。

```python
# 创建数据集并输出数据大小
rank = 0
device_num = 1
mindrecord_file = "nltk_mindrecord/mindrecord_32"
# 读取mindrecord转换成dataset
dataset = create_gru_dataset(epoch_count=9, batch_size=16,
                             dataset_path=mindrecord_file, rank_size=device_num,
                             rank_id=rank)
# 输出数据集大小
dataset_size = dataset.get_dataset_size()
print("dataset size is {}".format(dataset_size))
```

输出：

```python
dataset
size is 3617
```

## 网络定义

接下来的部分以 `MindDatasets` 转换的数据集构建了数据集并定义了迭代器，本教程的其余部分仅将模型定义为 `nn. Cell` 以及 `Optimizer` ，然后对其进行训练。

如果需要了解关于该模型的更多细节，可以通过官网[GRU模型](https://gitee.com/mindspore/mindspore/tree/r1.3/model_zoo/official/nlp/gru) 查看。

### 定义GRU网络和全链接层的权重

在定义网络前我们需要对全链接层和GRU网络中的权重进行定义。

```python
def gru_default_state(batch_size, input_size, hidden_size, num_layers=1, bidirectional=False):
    '''初始化GRU网络权重'''
    stdv = 1 / math.sqrt(hidden_size)
    # 初始化输入和隐藏层的权重
    weight_i = Parameter(Tensor(
        np.random.uniform(-stdv, stdv, (input_size, 3 * hidden_size)).astype(np.float32)), name='weight_i')
    weight_h = Parameter(Tensor(
        np.random.uniform(-stdv, stdv, (hidden_size, 3 * hidden_size)).astype(np.float32)), name='weight_h')
    bias_i = Parameter(Tensor(
        np.random.uniform(-stdv, stdv, (3 * hidden_size)).astype(np.float32)), name='bias_i')
    bias_h = Parameter(Tensor(
        np.random.uniform(-stdv, stdv, (3 * hidden_size)).astype(np.float32)), name='bias_h')
    init_h = Tensor(np.zeros((batch_size, hidden_size)).astype(np.float16))
    return weight_i, weight_h, bias_i, bias_h, init_h

def dense_default_state(in_channel, out_channel):
    '''初始化全链接层权重'''
    stdv = 1 / math.sqrt(in_channel)
    weight = Tensor(np.random.uniform(-stdv, stdv, (out_channel, in_channel)).astype(np.float32))
    bias = Tensor(np.random.uniform(-stdv, stdv, (out_channel)).astype(np.float32))
    return weight, bias
```

### 定义Encoder层和Decoder网络

因为Seq2Seq的网络采用了Encoder-Decoder架构，所以需要为Encoder和Decoder层分别定义一个网络。

#### Encoder层的BidirectionGRU网络

本教程采用采用双向门结构的 `BidirectionGRU` 网络作为序列到序列（seq2seq）网络的Encoder层。

```python
import mindspore.ops as ops
import mindspore.nn as nn
from mindspore import dtype as mstype

# 定义bidirectionGRU
class BidirectionGRU(nn.Cell):
    '''
    定义bidirectionGRU网络
    '''

    def __init__(self, is_training=True):
        super(BidirectionGRU, self).__init__()
        if is_training:
            self.batch_size = 16
        else:
            self.batch_size = 1
        self.embeddingzeBidirectionGRU = 256
        self.hidden_size = 512
        self.weight_i, self.weight_h, self.bias_i, self.bias_h, self.init_h = gru_default_state(self.batch_size,
                                                                                                self.embedding_size,
                                                                                                self.hidden_size)
        self.weight_bw_i, self.weight_bw_h, self.bias_bw_i, self.bias_bw_h, self.init_bw_h =
        gru_default_state(self.batch_size, self.embedding_size, self.hidden_size)

    self.reverse = ops.ReverseV2(axis=[1])
    self.concat = ops.Concat(axis=2)
    self.squeeze = ops.Squeeze(axis=0)
    self.rnn = ops.DynamicGRUV2()
    self.text_len = 32
    self.cast = ops.Cast()

    def construct(self, x):
        '''
        BidirectionGRU结构构造
        '''
        # 设置为粗精度方便计算
        x = self.cast(x, mstype.float16)
        # 定义卷积网络
        y1, _, _, _, _, _ = self.rnn(x, self.weight_i, self.weight_h, self.bias_i, self.bias_h, None, self.init_h)
        bw_x = self.reverse(x)
        y1_bw, _, _, _, _, _ = self.rnn(bw_x, self.weight_bw_i,
                                        self.weight_bw_h, self.bias_bw_i, self.bias_bw_h, None, self.init_bw_h)
        y1_bw = self.reverse(y1_bw)
        # 拼接网络
        output = self.concat((y1, y1_bw))
        hidden = self.concat((y1[self.text_len - 1:self.text_len:1, ::, ::],
                              y1_bw[self.text_len - 1:self.text_len:1, ::, ::]))
        hidden = self.squeeze(hidden)
        return output, hidden
```

GRU 背后的原理与LSTM非常相似，即用门控机制控制输入、记忆等信息做出预测。本教程利用 `GRU` 这种特征，通过继承 `nn. Cell` 来定义 `GRU` 作为seq2seq模型的Decoder层。

```python
# 定义GRU的decoder网络
class GRU(nn.Cell):

    def __init__(self, is_training=True):
        super(GRU, self).__init__()
        if is_training:
            self.batch_size = 16
        else:
            self.batch_size = 1
        self.embedding_size = 256
        self.hidden_size = 512
        self.weight_i, self.weight_h, self.bias_i, self.bias_h, self.init_h =
        gru_default_state(self.batch_size, self.embedding_size + self.hidden_size * 2, self.hidden_size)

    self.rnn = ops.DynamicGRUV2()
    self.cast = ops.Cast()

    def construct(self, x):
        x = self.cast(x, mstype.float16)
        y1, h1, _, _, _, _ = self.rnn(x, self.weight_i, self.weight_h, self.bias_i, self.bias_h, None, self.init_h)
        return y1, h1
```

### 定义Attention机制

引入Attention注意力机制是为了解决由长序列到定长向量转化而造成的信息损失瓶颈。Attention机制跟人类翻译文章的时候思路有些相似：将注意力关注于翻译部分相对应的上下文。当翻译当前词语时，会寻找源语句中相对应的几个词语，并结合之前的已经翻译的部分做出相应的翻译。

为了解决相同的问题，本教程也引入 `Attention` 机制。

```python
class Attention(nn.Cell):
    """
    定义Attention机制
    """

    def __init__(self):
        super(Attention, self).__init__()
        self.text_len = 32
        self.attn = nn.Dense(in_channels=512 * 3,
                             out_channels=512).to_float(mstype.float16)
        self.fc = nn.Dense(512, 1, has_bias=False).to_float(mstype.float16)
        self.expandims = ops.ExpandDims()
        self.tanh = ops.Tanh()
        self.softmax = ops.Softmax()
        self.tile = ops.Tile()
        self.transpose = ops.Transpose()
        self.concat = ops.Concat(axis=2)
        self.squeeze = ops.Squeeze(axis=2)
        self.cast = ops.Cast()

    def construct(self, hidden, encoder_outputs):
        hidden = self.expandims(hidden, 1)
        hidden = self.tile(hidden, (1, self.text_len, 1))
        encoder_outputs = self.transpose(encoder_outputs, (1, 0, 2))
        out = self.concat((hidden, encoder_outputs))
        out = self.attn(out)
        energy = self.tanh(out)
        attention = self.fc(energy)
        attention = self.squeeze(attention)
        attention = self.cast(attention, mstype.float32)
        attention = self.softmax(attention)
        attention = self.cast(attention, mstype.float16)
        return attention
```

### 定义Encoder架构部分

模型构建主要包括Encoder层与Decoder层。在Encoder层，首先需要定义输入的 `tensor` ，同时要对词句进行 `Embedding` ，再输入到RNN层。

现在我们来定义 `Encoder` 部分，该部分将前面定义的 `BidirectionGRU` 网络激活以此来完成encode任务。

```python
class Encoder(nn.Cell):
    """
    定义Encoder层
    """

    def __init__(self, is_training=True):
        super(Encoder, self).__init__()
        self.hidden_size = 512
        self.vocab_size = 8154
        self.embedding_size = 256
        self.embedding = nn.Embedding(self.vocab_size, self.embedding_size)
        # 引入BidirctionGRU网络
        self.rnn = BidirectionGRU(is_training=is_training).to_float(mstype.float16)
        self.fc = nn.Dense(2 * self.hidden_size, self.hidden_size).to_float(mstype.float16)
        self.shape = ops.Shape()
        self.transpose = ops.Transpose()
        self.p = ops.Print()
        self.cast = ops.Cast()
        self.text_len = 32
        self.squeeze = ops.Squeeze(axis=0)
        self.tanh = ops.Tanh()

    def construct(self, src):
        # 定义Encoder结构
        embedded = self.embedding(src)
        embedded = self.transpose(embedded, (1, 0, 2))
        embedded = self.cast(embedded, mstype.float16)
        output, hidden = self.rnn(embedded)
        hidden = self.fc(hidden)
        hidden = self.tanh(hidden)
        return output, hidden
```

### 定义Decoder架构部分

在Decoder端，我们主要要完成以下几件事情：

* 对target数据进行处理

* Embedding操作

* 构造Decoder层

现在我们来定义 `Decoder` 部分，该部分将前面定义的 `GRU` 网络激活并将 `attention` 模块引入以此来完成Decode任务。

```python
class Decoder(nn.Cell):
    """
    定义Decoder层
    """

    def __init__(self, is_training=True):
        super(Decoder, self).__init__()
        self.hidden_size = 512
        self.vocab_size = 6113
        self.embedding_size = 256
        self.embedding = nn.Embedding(self.vocab_size, self.embedding_size)
        # 初始化GRU网络
        self.rnn = GRU(is_training=is_training).to_float(mstype.float16)
        self.text_len = 32
        self.shape = ops.Shape()
        self.transpose = ops.Transpose()
        self.p = ops.Print()
        self.cast = ops.Cast()
        self.concat = ops.Concat(axis=2)
        self.squeeze = ops.Squeeze(axis=0)
        self.expandims = ops.ExpandDims()
        self.log_softmax = ops.LogSoftmax(axis=1)
        weight, bias = dense_default_state(self.embedding_size + self.hidden_size * 3, self.vocab_size)
        self.fc = nn.Dense(self.embedding_size + self.hidden_size * 3, self.vocab_size,
                           weight_init=weight, bias_init=bias).to_float(mstype.float16)
        # 引入Attention模块
        self.attention = Attention()
        self.bmm = ops.BatchMatMul()
        self.dropout = nn.Dropout(0.7)
        self.expandims = ops.ExpandDims()

    def construct(self, inputs, hidden, encoder_outputs):
        # 定义Decoder网络结构
        embedded = self.embedding(inputs)
        embedded = self.transpose(embedded, (1, 0, 2))
        embedded = self.cast(embedded, mstype.float16)
        attn = self.attention(hidden, encoder_outputs)
        attn = self.expandims(attn, 1)
        encoder_outputs = self.transpose(encoder_outputs, (1, 0, 2))
        weight = self.bmm(attn, encoder_outputs)
        weight = self.transpose(weight, (1, 0, 2))
        emd_con = self.concat((embedded, weight))
        output, hidden = self.rnn(emd_con)
        out = self.concat((embedded, output, weight))
        out = self.squeeze(out)
        hidden = self.squeeze(hidden)
        prediction = self.fc(out)
        prediction = self.dropout(prediction)
        prediction = self.cast(prediction, mstype.float32)
        prediction = self.cast(prediction, mstype.float32)
        pred_prob = self.log_softmax(prediction)
        pred_prob = self.expandims(pred_prob, 0)
        return pred_prob, hidden
```

### 定义seq2seq网络

通过拥有 `BidirectionGRU` 的encode模块和 `GRU` 的decode模块，终于达成了我们所需的所有序列到序列网络的所有架构模块，目前我们要做的就是将它们放入我们的最后一个网络：序列到序列网络。

```python
class Seq2Seq(nn.Cell):

    def __init__(self, is_training=True):
        super(Seq2Seq, self).__init__()
        if is_training:
            self.batch_size = 16
        else:
            self.batch_size = 1
        # 将encoder和decoder带入到网络中
        self.encoder = Encoder(is_training=is_training)
        self.decoder = Decoder(is_training=is_training)
        self.expandims = ops.ExpandDims()
        self.dropout = nn.Dropout()
        self.shape = ops.Shape()
        self.concat = ops.Concat(axis=0)
        self.argmax = ops.ArgMaxWithValue(axis=1, keep_dims=True)
        self.squeeze = ops.Squeeze(axis=0)
        self.sos = Tensor(np.ones((self.batch_size, 1)) * 2, mstype.int32)
        self.select = ops.Select()
        self.text_len = 32

    def construct(self, encoder_inputs, decoder_inputs, teacher_force):
        decoder_input = self.sos
        # 通过encoder得到hidden
        encoder_output, hidden = self.encoder(encoder_inputs)
        decoder_hidden = hidden
        decoder_outputs = ()
        for i in range(1, self.text_len):
            # 传递hidden给decoder层
            decoder_output, decoder_hidden = self.decoder(decoder_input, decoder_hidden, encoder_output)
            decoder_outputs += (decoder_output,)
            if self.training:
                decoder_input_force = decoder_inputs[::, i:i + 1]
                decoder_input_top1, _ = self.argmax(self.squeeze(decoder_output))
                decoder_input = self.select(teacher_force, decoder_input_force, decoder_input_top1)
            else:
                decoder_input, _ = self.argmax(self.squeeze(decoder_output))
        outputs = self.concat(decoder_outputs)
        return outputs
```

最后，来创建序列到序列的网络实例。

```python
network = Seq2Seq()
```

对语言翻译模型的表现进行评分时，我们必须告诉 `nn.Loss` 函数我们需要定如何计算loss。

```python
from mindspore import nn
from mindspore.ops import functional as F

class NLLLoss(nn.Loss):
    def __init__(self, reduction='mean'):
        super(NLLLoss, self).__init__(reduction)
        self.one_hot = ops.OneHot()
        self.reduce_sum = ops.ReduceSum()

    def construct(self, logits, label):
        label_one_hot = self.one_hot(label, ops.Shape(logits)[-1], ops.scalar_to_array(1.0), ops.scalar_to_array(0.0))
        loss = self.reduce_sum(-1.0 * logits * label_one_hot, (1,))
        return self.get_loss(loss)
```

## 定义带有Loss值的GRU计算方法

通过定义 `GRUWithLossCell` 方法类来将 `loss` 类和 `Seq2Seq` 方法合并，定义为新的 `cell` 方法。该方法需要提供网络实例作为输入参数。

```python
class GRUWithLossCell(nn.Cell):
    """
    配有loss计算的GRU网络
    """

    def __init__(self, network):
        super(GRUWithLossCell, self).__init__()
        self.network = network
        self.loss = NLLLoss()
        self.logits_shape = (-1, 8154)
        self.reshape = ops.Reshape()
        self.cast = ops.Cast()
        self.mean = ops.ReduceMean()
        self.text_len = 32
        self.split = ops.Split(axis=0, output_num=32 - 1)
        self.squeeze = ops.Squeeze()
        self.add = ops.AddN()
        self.transpose = ops.Transpose()
        self.shape = ops.Shape()

    def construct(self, encoder_inputs, decoder_inputs, teacher_force):
        logits = self.network(encoder_inputs, decoder_inputs, teacher_force)
        logits = self.cast(logits, mstype.float32)
        loss_total = ()
        decoder_targets = decoder_inputs
        decoder_output = logits

        # 遍历文本来计算loss值
        for i in range(1, self.text_len):
            loss = self.loss(self.squeeze(decoder_output[i - 1:i:1, ::, ::]), decoder_targets[:, i])
            loss_total += (loss,)
        loss = self.add(loss_total) / self.text_len
        return loss
```

将网络实例更新为带有loss值的 `GRUWithLossCell` 网络。

```python
network = GRUWithLossCell(network)
```

## 定义动态学习率生成器

这边需要定义三个学习率方法函数：

* 线性变化学习率方法`linear_warmup_learning_rate`

* 基于cos曲线的学习率变换方法`a_cosine_learning_rate`

* 学习率变化器`dynamic_lr`

```python
import math

def linear_warmup_learning_rate(current_step, warmup_steps, base_lr, init_lr):
    """线性学习率变化"""
    lr_inc = (float(base_lr) - float(init_lr)) / float(warmup_steps)
    learning_rate = float(init_lr) + lr_inc * current_step
    return learning_rate

def a_cosine_learning_rate(current_step, base_lr, warmup_steps, total_steps):
    """cos曲线学习率变化"""
    decay_steps = total_steps - warmup_steps
    linear_decay = (total_steps - current_step) / decay_steps
    cosine_decay = 0.5 * (1 + math.cos(math.pi * 2 * 0.47 * current_step / decay_steps))
    decayed = linear_decay * cosine_decay + 0.00001
    learning_rate = decayed * base_lr
    return learning_rate

def dynamic_lr(base_step):
    """学习率变化生成器"""
    base_lr = 0.001
    total_steps = int(base_step * 30)
    warmup_steps = int(300)
    lr = []
    for i in range(total_steps):
        if i < warmup_steps:
            lr.append(linear_warmup_learning_rate(i, warmup_steps, base_lr, base_lr * 300))
        else:
            lr.append(a_cosine_learning_rate(i, base_lr, warmup_steps, total_steps))
    return lr
```

实例化学习器

```python
lr = dynamic_lr(dataset_size)
```

## 定义梯度修剪方法

这里通过 `ClipGradients` 定义训练过程中的梯度修剪方法。

```python
from mindspore import Tensor, Parameter, ParameterTuple, context
import mindspore.nn as nn
import mindspore.ops as ops
from mindspore.context import ParallelMode
from mindspore.nn import DistributedGradReducer
from mindspore.parallel._utils import _get_device_num, _get_parallel_mode, _get_gradients_mean

grad_scale = ops.MultitypeFuncGraph("grad_scale")
reciprocal = ops.Reciprocal()
GRADIENT_CLIP_TYPE = 1
GRADIENT_CLIP_VALUE = 1.0

class ClipGradients(nn.Cell):
    """
    修剪梯度
    """

    def __init__(self):
        super(ClipGradients, self).__init__()
        self.clip_by_norm = nn.ClipByNorm()
        self.cast = ops.Cast()
        self.dtype = ops.DType()

    def construct(self, grads, clip_type, clip_value):
        if clip_type not in (0, 1):
            return grads
        new_grads = ()
        # 循环梯度元祖列表
        for grad in grads:
            dt = self.dtype(grad)
            if clip_type == 0:
                t = ops.clip_by_value(grad, self.cast(ops.tuple_to_array((-clip_value,)), dt),
                                      self.cast(ops.tuple_to_array((clip_value,)), dt))
            else:
                t = self.clip_by_norm(grad, self.cast(ops.tuple_to_array((clip_value,)), dt))
            t = self.cast(t, dt)
            new_grads = new_grads + (t,)
        return new_grads
```

因为梯度每次需要传入的所需数据格式不同，采用修饰器方法 `grad_scale` 和 `grad_overflow` 重新定义输入格式，分别为：

* grad_scale：scale定义为张量， grad定义为张量。
* grad_overflow：grad定义为张量。

定义后方便我们后续的修剪梯度过程。

```python
@grad_scale.register("Tensor", "Tensor")
def tensor_grad_scale(scale, grad):
    return grad * ops.Cast(reciprocal(scale), ops.Dtype(grad))

_grad_overflow = ops.MultitypeFuncGraph("_grad_overflow")
grad_overflow = ops.FloatStatus()

@_grad_overflow.register("Tensor")
def _tensor_grad_overflow(grad):
    return grad_overflow(grad)
```

## 定义单步训练pipline

当所有的准备完整后，我们开始正式定义我们一次训练所需要的pipline，通过 `GRUTrainOneStepWithLossScaleCell` 类用来封装我们对于GRU的训练过程。

在训练网络中添加优化，然后通过调用construct函数来创建反向图定义执行计算。

```python
class GRUTrainOneStepWithLossScaleCell(nn.Cell):
    """
    GRU网络训练
    """

    def __init__(self, network, optimizer, scale_update_cell=None):
        super(GRUTrainOneStepWithLossScaleCell, self).__init__(auto_prefix=False)
        self.network = network
        self.network.set_grad()
        self.network.add_flags(defer_inline=True)
        self.weights = ParameterTuple(network.trainable_params())
        self.optimizer = optimizer
        self.grad = ops.GradOperation(get_by_list=True,
                                      sens_param=True)
        self.reducer_flag = False
        self.allreduce = ops.AllReduce()

        self.parallel_mode = _get_parallel_mode()
        if self.parallel_mode not in ParallelMode.MODE_LIST:
            raise ValueError("Parallel mode does not support: ", self.parallel_mode)
        if self.parallel_mode in [ParallelMode.DATA_PARALLEL, ParallelMode.HYBRID_PARALLEL]:
            self.reducer_flag = True
        self.grad_reducer = None
        if self.reducer_flag:
            mean = _get_gradients_mean()
            degree = _get_device_num()
            self.grad_reducer = DistributedGradReducer(optimizer.parameters, mean, degree)
        self.is_distributed = (self.parallel_mode != ParallelMode.STAND_ALONE)
        self.clip_gradients = ClipGradients()
        self.cast = ops.Cast()
        if context.get_context("device_target") == "GPU":
            self.gpu_target = True
            self.float_status = ops.FloatStatus()
            self.addn = ops.AddN()
            self.reshape = ops.Reshape()
        self.reduce_sum = ops.ReduceSum(keep_dims=False)
        self.base = Tensor(1, mstype.float32)
        self.less_equal = ops.LessEqual()
        self.hyper_map = ops.HyperMap()

        self.loss_scale = None
        self.loss_scaling_manager = scale_update_cell
        if scale_update_cell:
            self.loss_scale = Parameter(Tensor(scale_update_cell.get_loss_scale(), dtype=mstype.float32))

    @ops.add_flags(has_effect=True)
    def construct(self,
                  encoder_inputs,
                  decoder_inputs,
                  teacher_force,
                  sens=None):
        """定义执行计算方法"""
        # 初始化网络和权重
        weights = self.weights
        loss = self.network(encoder_inputs,
                            decoder_inputs,
                            teacher_force)
        init = False
        if sens is None:
            scaling_sens = self.loss_scale
        else:
            scaling_sens = sens
        # 定义梯度就散
        grads = self.grad(self.network, weights)(encoder_inputs,
                                                 decoder_inputs,
                                                 teacher_force,
                                                 self.cast(scaling_sens,
                                                           mstype.float32))

        grads = self.hyper_map(ops.partial(grad_scale, scaling_sens), grads)
        grads = self.clip_gradients(grads, GRADIENT_CLIP_TYPE, GRADIENT_CLIP_VALUE)
        if self.reducer_flag:
            # 梯度下降
            grads = self.grad_reducer(grads)
        if not self.gpu_target:
            self.get_status(init)
            # 计算溢出缓冲区元素合，0为未溢出，大于0为溢出状态
            flag_sum = self.reduce_sum(init, (0,))
        else:
            flag_sum = self.hyper_map(ops.partial(_grad_overflow), grads)
            flag_sum = self.addn(flag_sum)
            # 将溢出和转换为标量
            # convert flag_sum to scalar
            flag_sum = self.reshape(flag_sum, (()))

        if self.is_distributed:
            flag_reduce = self.allreduce(flag_sum)
            cond = self.less_equal(self.base, flag_reduce)
        else:
            cond = self.less_equal(self.base, flag_sum)
        overflow = cond
        # 检测溢出状态
        if sens is None:
            overflow = self.loss_scaling_manager(self.loss_scale, cond)
        if overflow:
            succ = False
        else:
            succ = self.optimizer(grads)
        ret = (loss, cond, scaling_sens)
        return ops.Depend(ret, succ)
```

现在，来给我们的训练模式定义优化器 `opt` 和标量混合精度策略机制 `scale_manager` ，这两者和网络作为参数变量放入到我们的训练中，并通过DynamicLossScaleManager来监控动态损失。

```python
from mindspore.nn import Adam
from mindspore import DynamicLossScaleManager

# 激活优化器
opt = Adam(network.trainable_params(), learning_rate=lr)
from mindspore.train.loss_scale_manager import DynamicLossScaleManager

# 创建动态损失管理器
scale_manager = DynamicLossScaleManager(init_loss_scale=1024,
                                        scale_factor=2,
                                        scale_window=2000)
update_cell = scale_manager.get_update_cell()
netwithgrads = GRUTrainOneStepWithLossScaleCell(network, opt, update_cell)
```

### 定义训练监控函数

为了确认该模型的有效性和收敛性，我们需要定义监控函数来实时对模型的loss值进行观察，如果中间loss计算错误（NAN或者INF）便停止模型的训练，如果预打印次数做空则也报错误并停止运算。

```python
from mindspore.train.callback import Callback, CheckpointConfig, ModelCheckpoint, TimeMonitor

class LossCallBack(Callback):
    """
    监视训练中的损失
    """

    def __init__(self, per_print_times=1, rank_id=0):
        super(LossCallBack, self).__init__()
        if not isinstance(per_print_times, int) or per_print_times < 0:
            raise ValueError("print_step must be int and >= 0.")
        self._per_print_times = per_print_times
        self.rank_id = rank_id
        global time_stamp_init, time_stamp_first
        if not time_stamp_init:
            time_stamp_first = get_ms_timestamp()
            time_stamp_init = True

    def step_end(self, run_context):
        """
        记录到loss_rank_id.log文档loss记录
        """
        global time_stamp_first
        time_stamp_current = get_ms_timestamp()
        cb_params = run_context.original_args()
        print("time: {}, epoch: {}, step: {}, outputs are {}, loss: {}".format(time_stamp_current - time_stamp_first,
                                                                               cb_params.cur_epoch_num,
                                                                               cb_params.cur_step_num,
                                                                               str(cb_params.net_outputs),
                                                                               str(cb_params.net_outputs[0].asnumpy())))
        # 记录log日志
        with open("./loss_{}.log".format(self.rank_id), "a+") as f:
            f.write("time: {}, epoch: {}, step: {}, loss: {}, overflow: {}, loss_scale: {}".format(
                time_stamp_current - time_stamp_first,
                cb_params.cur_epoch_num,
                cb_params.cur_step_num,
                str(cb_params.net_outputs[0].asnumpy()),
                str(cb_params.net_outputs[1].asnumpy()),
                str(cb_params.net_outputs[2].asnumpy())))
            f.write('\n')
```

定义时间记录和ckpt文件保存方法。

```python
def get_ms_timestamp():
    """
    设置计时器
    """
    t = time.time()
    return int(round(t * 1000))

# 设置时间监控和损失回调
time_stamp_init = False
time_stamp_first = 0
time_cb = TimeMonitor(data_size=dataset_size)
loss_cb = LossCallBack(rank_id=rank)
cb = [time_cb, loss_cb]

# 保存ckpt文件
ckpt_config = CheckpointConfig(save_checkpoint_steps=10 * dataset_size,
                               keep_checkpoint_max=30)
save_ckpt_path = os.path.join(outputs_dir, 'ckpt_' + str(0) + '/')
ckpt_cb = ModelCheckpoint(config=ckpt_config,
                          directory=save_ckpt_path,
                          prefix='{}'.format(0))
cb += [ckpt_cb]
```

至此，我们实现了一个基本的序列到序列模型，Encoder通过对输入序列的学习，将学习到的信息转化为一个状态向量传递给Decoder，Decoder再基于这个输入得到输出。 最后，我们可以训练和评估该模型：

```python
from mindspore.train import Model

netwithgrads.set_train(True)
model = Model(netwithgrads)
model.train(9, dataset, callbacks=cb, dataset_sink_mode=True)
```

结果如下：

```py
time: 295568, epoch: 1, step: 3617, loss: 44.72535, overflow: False, loss_scale: 2048.0
time: 433144, epoch: 2, step: 7234, loss: 9.692089, overflow: False, loss_scale: 8192.0
time: 570487, epoch: 3, step: 10851, loss: 7.9861655, overflow: False, loss_scale: 32768.0
time: 707772, epoch: 4, step: 14468, loss: 7.3073406, overflow: False, loss_scale: 131072.0
time: 845222, epoch: 5, step: 18085, loss: 10.868818, overflow: False, loss_scale: 524288.0
time: 982747, epoch: 6, step: 21702, loss: 8.94546, overflow: False, loss_scale: 1048576.0
time: 1120184, epoch: 7, step: 25319, loss: 5.6173162, overflow: False, loss_scale: 4194304.0
time: 1257522, epoch: 8, step: 28936, loss: 4.893917, overflow: False, loss_scale: 16777216.0
time: 1394913, epoch: 9, step: 32553, loss: 4.632412, overflow: False, loss_scale: 67108864.0
```
