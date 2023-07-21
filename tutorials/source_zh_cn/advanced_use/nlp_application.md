# 自然语言处理应用

[![查看源文件](../_static/logo_source.png)](https://gitee.com/mindspore/docs/blob/r0.3/tutorials/source_zh_cn/advanced_use/nlp_application.md)

## 概述

情感分类是自然语言处理中文本分类问题的子集，属于自然语言处理最基础的应用。它是对带有感情色彩的主观性文本进行分析和推理的过程，即分析说话人的态度，是倾向正面还是反面。

> 通常情况下，我们会把情感类别分为正面、反面和中性三类。虽然“面无表情”的评论也有不少；不过，大部分时候会只采用正面和反面的案例进行训练，下面这个数据集就是很好的例子。

传统的文本主题分类问题的典型参考数据集为[20 Newsgroups](http://qwone.com/~jason/20Newsgroups/)，该数据集由20组新闻数据组成，包含约20000个新闻文档。
其主题列表中有些类别的数据比较相似，例如comp.sys.ibm.pc.hardware和comp.sys.mac.hardware都是和电脑系统硬件相关的题目，相似度比较高。而有些主题类别的数据相对来说就毫无关联，例如misc.forsale和soc.religion.christian。

就网络本身而言，文本主题分类的网络结构和情感分类的网络结构大致相似。在掌握了情感分类网络如何构造之后，很容易可以构造一个类似的网络，稍作调参即可用于文本主题分类任务。

但在业务上下文侧，文本主题分类是分析文本讨论的客观内容，而情感分类是要从文本中得到它是否支持某种观点的信息。比如，“《阿甘正传》真是好看极了，影片主题明确，节奏流畅。”这句话，在文本主题分类是要将其归为类别为“电影”主题，而情感分类则要挖掘出这一影评的态度是正面还是负面。

相对于传统的文本主题分类，情感分类较为简单，实用性也较强。常见的购物网站、电影网站都可以采集到相对高质量的数据集，也很容易给业务领域带来收益。例如，可以结合领域上下文，自动分析特定类型客户对当前产品的意见，可以分主题分用户类型对情感进行分析，以作针对性的处理，甚至基于此进一步推荐产品，提高转化率，带来更高的商业收益。

特殊领域中，某些非极性词也充分表达了用户的情感倾向，比如下载使用APP时，“卡死了”、“下载太慢了”就表达了用户的负面情感倾向；股票领域中，“看涨”、“牛市”表达的就是用户的正面情感倾向。所以，本质上，我们希望模型能够在垂直领域中，挖掘出一些特殊的表达，作为极性词给情感分类系统使用：

$垂直极性词 = 通用极性词 + 领域特有极性词$

按照处理文本的粒度不同，情感分析可分为词语级、短语级、句子级、段落级以及篇章级等几个研究层次。这里以“段落级”为例，输入为一个段落，输出为影评是正面还是负面的信息。

## 准备及设计
### 下载数据集

采用IMDB影评数据集作为实验数据。
> 数据集下载地址：<http://ai.stanford.edu/~amaas/data/sentiment/>

以下是负面影评（Negative）和正面影评（Positive）的案例。

| Review  | Label  |
|---|---|
| "Quitting" may be as much about exiting a pre-ordained identity as about drug withdrawal. As a rural guy coming to Beijing, class and success must have struck this young artist face on as an appeal to separate from his roots and far surpass his peasant parents' acting success. Troubles arise, however, when the new man is too new, when it demands too big a departure from family, history, nature, and personal identity. The ensuing splits, and confusion between the imaginary and the real and the dissonance between the ordinary and the heroic are the stuff of a gut check on the one hand or a complete escape from self on the other.  |  Negative |
| This movie is amazing because the fact that the real people portray themselves and their real life experience and do such a good job it's like they're almost living the past over again. Jia Hongsheng plays himself an actor who quit everything except music and drugs struggling with depression and searching for the meaning of life while being angry at everyone especially the people who care for him most.  | Positive  |

同时，我们要下载GloVe文件，并在文件开头处添加新的一行，意思是总共读取400000个单词，每个单词用300纬度的词向量表示。
```
400000 300
```
GloVe文件下载地址：<http://nlp.stanford.edu/data/glove.6B.zip>。

### 确定评价标准

作为典型的分类问题，情感分类的评价标准可以比照普通的分类问题处理。常见的精度（Accuracy）、精准度（Precision）、召回率（Recall）和F_beta分数都可以作为参考。

$精度（Accuracy）= 分类正确的样本数目 / 总样本数目$

$精准度（Precision）= 真阳性样本数目 / 所有预测类别为阳性的样本数目$

$召回率（Recall）= 真阳性样本数目 / 所有真实类别为阳性的样本数目$

$F1分数 = (2 * Precision * Recall) / (Precision + Recall)$

在IMDB这个数据集中，正负样本数差别不大，可以简单地用精度（accuracy）作为分类器的衡量标准。


### 确定网络及流程

我们使用LSTM网络进行自然语言处理。
1. 加载使用的数据集，并进行必要的数据处理。
2. 使用LSTM网络训练数据，生成模型。
    > LSTM（Long short-term memory，长短期记忆）网络是一种时间循环神经网络，适合于处理和预测时间序列中间隔和延迟非常长的重要事件。具体介绍可参考网上资料，在此不再赘述。
3. 得到模型之后，使用验证数据集，查看模型精度情况。

> 本例面向GPU硬件平台，你可以在这里下载完整的样例代码：<https://gitee.com/mindspore/docs/tree/r0.3/tutorials/tutorial_code/lstm>
> - main.py：代码文件，包括数据预处理、网络定义、模型训练等代码。
> - config.py：网络中的一些配置，包括batch size、进行几次epoch训练等。

## 实现阶段
### 导入需要的库文件
下列是我们所需要的公共模块及MindSpore的模块及库文件。
```python
import os
import shutil
import math
import argparse
import json
from itertools import chain
import numpy as np
from config import lstm_cfg as cfg
# Install gensim with 'pip install gensim'
import gensim

import mindspore.nn as nn
import mindspore.context as context
import mindspore.dataset as ds
from mindspore.ops import operations as P
from mindspore import Tensor
from mindspore.common.initializer import initializer
from mindspore.common.parameter import Parameter
from mindspore.mindrecord import FileWriter
from mindspore.train import Model
from mindspore.nn.metrics import Accuracy
from mindspore.train.serialization import load_checkpoint, load_param_into_net
from mindspore.train.callback import ModelCheckpoint, CheckpointConfig, LossMonitor
```

### 配置环境信息

1. 使用`parser`模块，传入运行必要的信息，如数据集存放路径，GloVe存放路径，这样的好处是，对于经常变化的配置，可以在运行代码时输入，使用更加灵活。
    ```python
    parser = argparse.ArgumentParser(description='MindSpore LSTM Example')
    parser.add_argument('--preprocess', type=str, default='false', choices=['true', 'false'],
                        help='Whether to perform data preprocessing')
    parser.add_argument('--mode', type=str, default="train", choices=['train', 'test'],
                        help='implement phase, set to train or test')
    # Download dataset from 'http://ai.stanford.edu/~amaas/data/sentiment/aclImdb_v1.tar.gz' and extract to 'aclimdb_path'
    parser.add_argument('--aclimdb_path', type=str, default="./aclImdb",
                        help='path where the dataset is store')
    # Download GloVe from 'http://nlp.stanford.edu/data/glove.6B.zip' and extract to 'glove_path'
    # Add a new line '400000 300' at the beginning of 'glove.6B.300d.txt' with '40000' for total words and '300' for vector length
    parser.add_argument('--glove_path', type=str, default="./glove",
                        help='path where the GloVe is store')
    # Specify the path to save preprocessed data
    parser.add_argument('--preprocess_path', type=str, default="./preprocess",
                        help='path where the pre-process data is store')
    # Specify the path to save the CheckPoint file
    parser.add_argument('--ckpt_path', type=str, default="./ckpt", help='if mode is test, must provide\
                        path where the trained ckpt file')
    args = parser.parse_args()
    ```

2. 实现代码前，需要配置必要的信息，包括环境信息、执行的模式、后端信息及硬件信息。

    ```python
    context.set_context(
        mode=context.GRAPH_MODE,
        save_graphs=False,
        device_target="GPU")
    ```
    详细的接口配置信息，请参见`context.set_context`接口说明。

### 预处理数据集

1. 对文本数据集进行处理，包括编码、分词、对齐、处理GloVe原始数据，使之能够适应网络结构。

    ```python
    # Read data from the specified directory
    def read_imdb(path, seg='train'):
        """ read imdb dataset """
        pos_or_neg = ['pos', 'neg']
        data = []
        for label in pos_or_neg:
            files = os.listdir(os.path.join(path, seg, label))
            for file in files:
                with open(os.path.join(path, seg, label, file), 'r', encoding='utf8') as rf:
                    review = rf.read().replace('\n', '')
                    if label == 'pos':
                        data.append([review, 1])
                    elif label == 'neg':
                        data.append([review, 0])
        return data

    # Split records into words with spaces as separators
    def tokenizer(text):
        return [tok.lower() for tok in text.split(' ')]

    # Encode words into vectors
    def encode_samples(tokenized_samples, word_to_idx):
        """ encode word to index """
        features = []
        for sample in tokenized_samples:
            feature = []
            for token in sample:
                if token in word_to_idx:
                    feature.append(word_to_idx[token])
                else:
                    feature.append(0)
            features.append(feature)
        return features

    # Align the number of words in each record to 500 words
    def pad_samples(features, maxlen=500, pad=0):
        """ pad all features to the same length """
        padded_features = []
        for feature in features:
            if len(feature) >= maxlen:
                padded_feature = feature[:maxlen]
            else:
                padded_feature = feature
                while len(padded_feature) < maxlen:
                    padded_feature.append(pad)
            padded_features.append(padded_feature)
        return padded_features

    # Crop GloVe raw data into a table which contains about 25,000 word vectors
    def collect_weight(glove_path, vocab, word_to_idx, embed_size):
        """ collect weight """
        vocab_size = len(vocab)
        wvmodel = gensim.models.KeyedVectors.load_word2vec_format(os.path.join(glove_path, 'glove.6B.300d.txt'),
                                                                binary=False, encoding='utf-8')
        weight_np = np.zeros((vocab_size+1, embed_size)).astype(np.float32)

        idx_to_word = {i+1: word for i, word in enumerate(vocab)}
        idx_to_word[0] = '<unk>'

        for i in range(len(wvmodel.index2word)):
            try:
                index = word_to_idx[wvmodel.index2word[i]]
            except KeyError:
                continue
            weight_np[index, :] = wvmodel.get_vector(
                idx_to_word[word_to_idx[wvmodel.index2word[i]]])
        return weight_np

    def preprocess(aclimdb_path, glove_path, embed_size):
        """ preprocess the train and test data """
        train_data = read_imdb(aclimdb_path, 'train')
        test_data = read_imdb(aclimdb_path, 'test')

        train_tokenized = []
        test_tokenized = []
        for review, _ in train_data:
            train_tokenized.append(tokenizer(review))
        for review, _ in test_data:
            test_tokenized.append(tokenizer(review))

        vocab = set(chain(*train_tokenized))
        vocab_size = len(vocab)
        print("vocab_size: ", vocab_size)

        word_to_idx = {word: i+1 for i, word in enumerate(vocab)}
        word_to_idx['<unk>'] = 0

        train_features = np.array(pad_samples(encode_samples(train_tokenized, word_to_idx))).astype(np.int32)
        train_labels = np.array([score for _, score in train_data]).astype(np.int32)
        test_features = np.array(pad_samples(encode_samples(test_tokenized, word_to_idx))).astype(np.int32)
        test_labels = np.array([score for _, score in test_data]).astype(np.int32)

        weight_np = collect_weight(glove_path, vocab, word_to_idx, embed_size)
        return train_features, train_labels, test_features, test_labels, weight_np, vocab_size

    ```

2. 将数据集格式转化为`mindrecord`格式，便于MindSpore读取。

    ```python
    def get_imdb_data(labels_data, features_data):
        data_list = []
        for i, (label, feature) in enumerate(zip(labels_data, features_data)):
            data_json = {"id": i,
                        "label": int(label),
                        "feature": feature.reshape(-1)}
            data_list.append(data_json)
        return data_list

    # Convert the dataset to mindrecord dateset which is supported by MindSpore
    def convert_to_mindrecord(embed_size, aclimdb_path, proprocess_path, glove_path):
        """ convert imdb dataset to mindrecord """
        num_shard = 4
        train_features, train_labels, test_features, test_labels, weight_np, _ = \
            preprocess(aclimdb_path, glove_path, embed_size)
        np.savetxt(os.path.join(proprocess_path, 'weight.txt'), weight_np)

        # write mindrecord
        schema_json = {"id": {"type": "int32"},
                    "label": {"type": "int32"},
                    "feature": {"type": "int32", "shape": [-1]}}

        writer = FileWriter(os.path.join(proprocess_path, 'aclImdb_train.mindrecord'), num_shard)
        data = get_imdb_data(train_labels, train_features)
        writer.add_schema(schema_json, "nlp_schema")
        writer.add_index(["id", "label"])
        writer.write_raw_data(data)
        writer.commit()

        writer = FileWriter(os.path.join(proprocess_path, 'aclImdb_test.mindrecord'), num_shard)
        data = get_imdb_data(test_labels, test_features)
        writer.add_schema(schema_json, "nlp_schema")
        writer.add_index(["id", "label"])
        writer.write_raw_data(data)
        writer.commit()

    print("============== Starting Data Pre-processing ==============")
    shutil.rmtree(args.preprocess_path)
    os.mkdir(args.preprocess_path)
    convert_to_mindrecord(cfg.embed_size, args.aclimdb_path, args.preprocess_path, args.glove_path)
    ```

3. 创建数据集。
    ```python
    def create_dataset(base_path, batch_size, num_epochs, is_train):
        """Create dataset for training."""
        columns_list = ["feature", "label"]
        num_consumer = 4

        if is_train:
            path = os.path.join(base_path, 'aclImdb_train.mindrecord0')
        else:
            path = os.path.join(base_path, 'aclImdb_test.mindrecord0')

        dtrain = ds.MindDataset(path, columns_list, num_consumer)
        dtrain = dtrain.shuffle(buffer_size=dtrain.get_dataset_size())
        dtrain = dtrain.batch(batch_size, drop_remainder=True)
        dtrain = dtrain.repeat(count=num_epochs)

        return dtrain

    ds_train = create_dataset(args.preprocess_path, cfg.batch_size, cfg.num_epochs, True)
    ```

### 定义网络

1. 初始化网络参数及网络状态。

    ```python
    def init_lstm_weight(
            input_size,
            hidden_size,
            num_layers,
            bidirectional,
            has_bias=True):
        """Initialize lstm weight."""
        num_directions = 1
        if bidirectional:
            num_directions = 2

        weight_size = 0
        gate_size = 4 * hidden_size
        for layer in range(num_layers):
            for _ in range(num_directions):
                input_layer_size = input_size if layer == 0 else hidden_size * num_directions
                weight_size += gate_size * input_layer_size
                weight_size += gate_size * hidden_size
                if has_bias:
                    weight_size += 2 * gate_size

        stdv = 1 / math.sqrt(hidden_size)
        w_np = np.random.uniform(-stdv, stdv, (weight_size,
                                            1, 1)).astype(np.float32)
        w = Parameter(
            initializer(
                Tensor(w_np), [
                    weight_size, 1, 1]), name='weight')

        return w

    # Initialize short-term memory (h) and long-term memory (c) to 0
    def lstm_default_state(batch_size, hidden_size, num_layers, bidirectional):
        """init default input."""
        num_directions = 1
        if bidirectional:
            num_directions = 2

        h = Tensor(
            np.zeros((num_layers * num_directions, batch_size, hidden_size)).astype(np.float32))
        c = Tensor(
            np.zeros((num_layers * num_directions, batch_size, hidden_size)).astype(np.float32))
        return h, c
    ```

2. 使用`cell`方法，定义网络结构。
    ```python
    class SentimentNet(nn.Cell):
        """Sentiment network structure."""
        def __init__(self,
                    vocab_size,
                    embed_size,
                    num_hiddens,
                    num_layers,
                    bidirectional,
                    num_classes,
                    weight,
                    batch_size):
            super(SentimentNet, self).__init__()
            # Mapp words to vectors
            self.embedding = nn.Embedding(vocab_size,
                                        embed_size,
                                        embedding_table=weight)
            self.embedding.embedding_table.requires_grad = False
            self.trans = P.Transpose()
            self.perm = (1, 0, 2)
            self.encoder = nn.LSTM(input_size=embed_size,
                                hidden_size=num_hiddens,
                                num_layers=num_layers,
                                has_bias=True,
                                bidirectional=bidirectional,
                                dropout=0.0)
            w_init = init_lstm_weight(
                embed_size,
                num_hiddens,
                num_layers,
                bidirectional)
            self.encoder.weight = w_init
            self.h, self.c = lstm_default_state(batch_size, num_hiddens, num_layers, bidirectional)

            self.concat = P.Concat(1)
            if bidirectional:
                self.decoder = nn.Dense(num_hiddens * 4, num_classes)
            else:
                self.decoder = nn.Dense(num_hiddens * 2, num_classes)

        def construct(self, inputs):
            # input：(64,500,300)
            embeddings = self.embedding(inputs)
            embeddings = self.trans(embeddings, self.perm)
            output, _ = self.encoder(embeddings, (self.h, self.c))
            # states[i] size(64,200)  -> encoding.size(64,400)
            encoding = self.concat((output[0], output[1]))
            outputs = self.decoder(encoding)
            return outputs


    embedding_table = np.loadtxt(os.path.join(args.preprocess_path, "weight.txt")).astype(np.float32)
    network = SentimentNet(vocab_size=embedding_table.shape[0],
                    embed_size=cfg.embed_size,
                    num_hiddens=cfg.num_hiddens,
                    num_layers=cfg.num_layers,
                    bidirectional=cfg.bidirectional,
                    num_classes=cfg.num_classes,
                    weight=Tensor(embedding_table),
                    batch_size=cfg.batch_size)
    ```

### 定义优化器及损失函数

定义优化器及损失函数的示例代码如下：

```python
loss = nn.SoftmaxCrossEntropyWithLogits(is_grad=False, sparse=True)
opt = nn.Momentum(network.trainable_params(), cfg.learning_rate, cfg.momentum)
loss_cb = LossMonitor()
```

### 训练并保存模型

加载对应数据集并配置好CheckPoint生成信息，然后使用`model.train`接口，进行模型训练。

```python
model = Model(network, loss, opt, {'acc': Accuracy()})

print("============== Starting Training ==============")
ds_train = create_dataset(args.preprocess_path, cfg.batch_size, cfg.num_epochs, True)
config_ck = CheckpointConfig(save_checkpoint_steps=cfg.save_checkpoint_steps,
                                keep_checkpoint_max=cfg.keep_checkpoint_max)
ckpoint_cb = ModelCheckpoint(prefix="lstm", directory=args.ckpt_path, config=config_ck)
model.train(cfg.num_epochs, ds_train, callbacks=[ckpoint_cb, loss_cb])
```

### 模型验证

加载验证数据集及保存的CheckPoint文件，进行验证，查看模型质量。

```python
print("============== Starting Testing ==============")
ds_eval = create_dataset(args.preprocess_path, cfg.batch_size, 1, False)
param_dict = load_checkpoint(args.ckpt_path)
load_param_into_net(network, param_dict)
acc = model.eval(ds_eval)
print("============== Accuracy:{} ==============".format(acc))
```

## 实验结果
在经历了10轮epoch之后，在训练集上的精度收敛到约85%，在测试集上的精度约为86%。

**执行训练**
1. 运行训练代码，查看运行结果。
    ```shell
    $ python main.py --preprocess=true --mode=train --ckpt_path=./ckpt
    ```

    输出如下，可以看到loss值随着训练逐步降低，最后达到0.249左右，即经过10个epoch的训练，对当前文本分析的结果正确率在85%左右：

    ```shell
    ============== Starting Data Pre-processing ==============
    vocab_size:  252192
    ============== Starting Training ==============
    [INFO] ME(15831:140036161672960,MainProcess):2020-03-09-16:29:02.785.484 [mindspore/train/serialization.py:118] Execute save checkpoint process.
    [INFO] ME(15831:140036161672960,MainProcess):2020-03-09-16:29:03.658.733 [mindspore/train/serialization.py:143] Save checkpoint process finish.
    epoch: 1 step: 390 , loss is 0.6926409
    ...
    [INFO] ME(15831:140036161672960,MainProcess):2020-03-09-16:32:18.598.405 [mindspore/train/serialization.py:118] Execute save checkpoint process.
    [INFO] ME(15831:140036161672960,MainProcess):2020-03-09-16:32:19.561.926 [mindspore/train/serialization.py:143] Save checkpoint process finish.
    epoch: 6 step: 390 , loss is 0.222701
    ...
    epoch: 10 step: 390 , loss is 0.22616856
    epoch: 10 step: 390 , loss is 0.24914627
    ```

2. 查看保存的CheckPoint文件。

   训练过程中保存了CheckPoint文件，即模型文件，我们可以查看文件保存的路径下的所有保存文件。

    ```shell
    $ ls ckpt/
    ```

    输出如下：

    ```shell
    lstm-10_390.ckpt  lstm-1_390.ckpt  lstm-2_390.ckpt  lstm-3_390.ckpt  lstm-4_390.ckpt  lstm-5_390.ckpt  lstm-6_390.ckpt  lstm-7_390.ckpt  lstm-8_390.ckpt  lstm-9_390.ckpt
    ```

**验证模型**

使用最后保存的CheckPoint文件，加载验证数据集，进行验证。

```shell
$ python main.py --mode=test --ckpt_path=./ckpt/lstm-10_390.ckpt
```

输出如下，可以看到使用验证的数据集，对文本的情感分析正确率在86%左右，达到一个基本满意的结果。

```shell
RegisterOperatorCreator:OperatorCreators init
============== Starting Testing ==============
[INFO] ME(29963:140462460516096,MainProcess):2020-03-09-16:37:19.333.605 [mindspore/train/serialization.py:169] Execute load checkpoint process.
[INFO] ME(29963:140462460516096,MainProcess):2020-03-09-16:37:20.434.749 [mindspore/train/serialization.py:200] Load checkpoint process finish.
[INFO] ME(29963:140462460516096,MainProcess):2020-03-09-16:37:20.467.336 [mindspore/train/serialization.py:233] Execute parameter into net process.
[INFO] ME(29963:140462460516096,MainProcess):2020-03-09-16:37:20.467.649 [mindspore/train/serialization.py:268] Load parameter into net process finish.
============== Accuracy:{'acc': 0.8599358974358975} ==============
```



