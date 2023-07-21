# Natural Language Processing (NLP) Application

[![View Source On Gitee](../_static/logo_source.png)](https://gitee.com/mindspore/docs/blob/r0.5/tutorials/source_en/advanced_use/nlp_application.md)

## Overview

Sentiment classification is a subset of text classification in NLP, and is the most basic application of NLP. It is a process of analyzing and inferencing affective states and subjective information, that is, analyzing whether a person's sentiment is positive or negative.

> Generally, sentiments are classified into three categories: positive, negative, and neutral. In most cases, only positive and negative sentiments are used for training regardless of the neutral sentiments. The following dataset is a good example.

[20 Newsgroups](http://qwone.com/~jason/20Newsgroups/) is a typical reference dataset for traditional text classification. It is a collection of approximately 20,000 news documents partitioned across 20 different newsgroups.
Some of the newsgroups are very closely related to each other (such as comp.sys.ibm.pc.hardware and comp.sys.mac.hardware), while others are highly unrelated (such as misc.forsale and soc.religion.christian).

As far as the network itself is concerned, the network structure of text classification is roughly similar to that of sentiment classification. After mastering how to construct the sentiment classification network, it is easy to construct a similar network which can be used in a text classification task with some parameter adjustments.

In the service context, text classification is to analyze the objective content in the text discussion, but sentiment classification is to find a viewpoint from the text. For example, "Forrest Gump has a clear theme and smooth pacing, which is excellent." In the text classification, this sentence is classified into a "movie" theme, but in the sentiment classification, this movie review is used to explore whether the sentiment is positive or negative.

Compared with traditional text classification, sentiment classification is simpler and more practical. High-quality datasets can be collected on common shopping websites and movie websites to benefit the business domains. For example, based on the domain context, the system can automatically analyze opinions of specific types of customers on the current product, analyze sentiments by subject and user type, and recommend products based on the analysis result, improving the conversion rate and bringing more business benefits.

In special fields, some non-polar words also fully express a sentimental tendency of a user. For example, when an app is downloaded and used, "the app is stuck" and "the download speed is so slow" express users' negative sentiments. In the stock market, "bullish" and "bull market" express users' positive sentiments. Therefore, in essence, we hope that the model can be used to mine special expressions in the vertical field as polarity words for the sentiment classification system.

Vertical polarity word = General polarity word + Domain-specific polarity word

According to the text processing granularity, sentiment analysis can be divided into word, phrase, sentence, paragraph, and chapter levels. A sentiment analysis at paragraph level is used as an example. The input is a paragraph, and the output is information about whether the movie review is positive or negative.

## Preparation and Design
### Downloading the Dataset

The IMDb movie review dataset is used as experimental data.
> Dataset download address: <http://ai.stanford.edu/~amaas/data/sentiment/>

The following are cases of negative and positive reviews.

| Review  | Label  | 
|---|---|
| "Quitting" may be as much about exiting a pre-ordained identity as about drug withdrawal. As a rural guy coming to Beijing, class and success must have struck this young artist face on as an appeal to separate from his roots and far surpass his peasant parents' acting success. Troubles arise, however, when the new man is too new, when it demands too big a departure from family, history, nature, and personal identity. The ensuing splits, and confusion between the imaginary and the real and the dissonance between the ordinary and the heroic are the stuff of a gut check on the one hand or a complete escape from self on the other.  |  Negative |  
| This movie is amazing because the fact that the real people portray themselves and their real life experience and do such a good job it's like they're almost living the past over again. Jia Hongsheng plays himself an actor who quit everything except music and drugs struggling with depression and searching for the meaning of life while being angry at everyone especially the people who care for him most.  | Positive  |

Download the GloVe file and add the following line at the beginning of the file, which means that a total of 400,000 words are read, and each word is represented by a word vector of 300 latitudes.
```
400000 300
```
GloVe file download address: <http://nlp.stanford.edu/data/glove.6B.zip>

### Determining Evaluation Criteria

As a typical classification, the evaluation criteria of sentiment classification can be determined by referring to that of the common classification. For example, accuracy, precision, recall, and F_beta scores can be used as references.

Accuracy = Number of accurately classified samples/Total number of samples

Precision = True positives/(True positives + False positives)

Recall = True positives/(True positives + False negatives)

F1 score = (2 x Precision x Recall)/(Precision + Recall)

In the IMDb dataset, the number of positive and negative samples does not vary greatly. Accuracy can be used as the evaluation criterion of the classification system.


### Determining the Network and Process

Currently, MindSpore GPU and CPU supports SentimentNet network based on the long short-term memory (LSTM) network for NLP.
1. Load the dataset in use and process data.
2. Use the SentimentNet network based on LSTM training data to generate a model.
    Long short-term memory (LSTM) is an artificial recurrent neural network (RNN) architecture used for processing and predicting an important event with a long interval and delay in a time sequence. For details, refer to online documentation.
3. After the model is obtained, use the validation dataset to check the accuracy of model.

> The current sample is for the Ascend 910 AI processor. You can find the complete executable sample code at：<https://gitee.com/mindspore/docs/tree/r0.5/tutorials/tutorial_code/lstm>
> - `main.py`: code file, including code for data preprocessing, network definition, and model training.
> - `config.py`: some configurations on the network, including the `batch size` and number of training epochs.


## Implementation
### Importing Library Files
The following are the required public modules and MindSpore modules and library files.
```python
import os
import shutil
import math
import argparse
import json
from itertools import chain
import numpy as np
from config import lstm_cfg as cfg

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
from mindspore.train.callback import ModelCheckpoint, CheckpointConfig, LossMonitor, TimeMonitor
# Install gensim with 'pip install gensim'
import gensim
```

### Configuring Environment Information

1. The `parser` module is used to transfer necessary information for running, such as storage paths of the dataset and the GloVe file. In this way, the frequently changed configurations can be entered during code running, which is more flexible.
    ```python
    parser = argparse.ArgumentParser(description='MindSpore LSTM Example')
    parser.add_argument('--preprocess', type=str, default='false', choices=['true', 'false'],
                        help='whether to preprocess data.')
    parser.add_argument('--mode', type=str, default="train", choices=['train', 'test'],
                        help='implement phase, set to train or test')
    # Download dataset from 'http://ai.stanford.edu/~amaas/data/sentiment/aclImdb_v1.tar.gz' and extract to 'aclimdb_path'
    parser.add_argument('--aclimdb_path', type=str, default="./aclImdb",
                        help='path where the dataset is stored')
    # Download GloVe from 'http://nlp.stanford.edu/data/glove.6B.zip' and extract to 'glove_path'
    # Add a new line '400000 300' at the beginning of 'glove.6B.300d.txt' with '40000' for total words and '300' for vector length
    parser.add_argument('--glove_path', type=str, default="./glove",
                        help='path where the GloVe is stored')
    # Specify the path to save preprocessed data                
    parser.add_argument('--preprocess_path', type=str, default="./preprocess",
                        help='path where the pre-process data is stored')
    # Specify the path to save the CheckPoint file                    
    parser.add_argument('--ckpt_path', type=str, default="./",
                        help='if mode is test, must provide path where the trained ckpt file.')
    # Specify the target device to run
    parser.add_argument('--device_target', type=str, default="GPU", choices=['GPU', 'CPU'],
                        help='the target device to run, support "GPU", "CPU". Default: "GPU".')
    args = parser.parse_args()
    ```

2. Before implementing code, configure necessary information, including the environment information, execution mode, backend information, and hardware information.
   
    ```python
    context.set_context(
        mode=context.GRAPH_MODE,
        save_graphs=False,
        device_target=args.device_target)
    ```
    For details about the API configuration, see the `context.set_context`.

### Preprocessing the Dataset

1. Process the text dataset, including encoding, word segmentation, alignment, and processing the original GloVe data to adapt to the network structure.

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

2. Convert the dataset format to the `mindrecord` format for MindSpore to read.

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

3. Create a dataset.
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

### Defining the Network

1. Initialize network parameters and status.

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

2. Use the `cell` method to define the network structure.
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

### Defining the Optimizer and Loss Function

The sample code for defining the optimizer and loss function is as follows:

```python
loss = nn.SoftmaxCrossEntropyWithLogits(is_grad=False, sparse=True)
opt = nn.Momentum(network.trainable_params(), cfg.learning_rate, cfg.momentum)
loss_cb = LossMonitor()
```

### Training and Saving the Model

Load the corresponding dataset, configure the CheckPoint generation information, and train the model using the `model.train` API.

```python
model = Model(network, loss, opt, {'acc': Accuracy()})

print("============== Starting Training ==============")
ds_train = create_dataset(args.preprocess_path, cfg.batch_size, cfg.num_epochs, True)
config_ck = CheckpointConfig(save_checkpoint_steps=cfg.save_checkpoint_steps,
                                keep_checkpoint_max=cfg.keep_checkpoint_max)
ckpoint_cb = ModelCheckpoint(prefix="lstm", directory=args.ckpt_path, config=config_ck)
time_cb = TimeMonitor(data_size=ds_train.get_dataset_size())
if args.device_target == "CPU":
    model.train(cfg.num_epochs, ds_train, callbacks=[time_cb, ckpoint_cb, loss_cb], dataset_sink_mode=False)
else:
    model.train(cfg.num_epochs, ds_train, callbacks=[time_cb, ckpoint_cb, loss_cb])
```

### Validating the Model

Load the validation dataset and saved CheckPoint file, perform validation, and view the model quality.

```python
print("============== Starting Testing ==============")
ds_eval = create_dataset(args.preprocess_path, cfg.batch_size, 1, False)
param_dict = load_checkpoint(args.ckpt_path)
load_param_into_net(network, param_dict)
if args.device_target == "CPU":
    acc = model.eval(ds_eval, dataset_sink_mode=False)
else:
    acc = model.eval(ds_eval)
print("============== Accuracy:{} ==============".format(acc))
```

## Experiment Result
After 10 epochs, the accuracy on the training set converges to about 85%, and the accuracy on the test set is about 86%.

**Training Execution**
1. Run the training code and view the running result.
    ```shell
    $ python main.py --preprocess=true --mode=train --ckpt_path=./ --device_target=GPU
    ```

    As shown in the following output, the loss value decreases gradually with the training process and reaches about 0.249. That is, after 10 epochs of training, the accuracy of the current text analysis result is about 85%.

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

2. Check the saved CheckPoint files.
   
   CheckPoint files (model files) are saved during the training. You can view all saved files in the file path.

    ```shell
    $ ls ./*.ckpt
    ```

    The output is as follows:

    ```shell
    lstm-10_390.ckpt  lstm-1_390.ckpt  lstm-2_390.ckpt  lstm-3_390.ckpt  lstm-4_390.ckpt  lstm-5_390.ckpt  lstm-6_390.ckpt  lstm-7_390.ckpt  lstm-8_390.ckpt  lstm-9_390.ckpt
    ```

**Model Validation**

Use the last saved CheckPoint file to load and validate the dataset.

```shell
$ python main.py --mode=test --ckpt_path=./lstm-10_390.ckpt --device_target=GPU
```

As shown in the following output, the sentiment analysis accuracy of the text is about 86%, which is basically satisfactory.

```shell
RegisterOperatorCreator:OperatorCreators init
============== Starting Testing ==============
[INFO] ME(29963:140462460516096,MainProcess):2020-03-09-16:37:19.333.605 [mindspore/train/serialization.py:169] Execute load checkpoint process.
[INFO] ME(29963:140462460516096,MainProcess):2020-03-09-16:37:20.434.749 [mindspore/train/serialization.py:200] Load checkpoint process finish.
[INFO] ME(29963:140462460516096,MainProcess):2020-03-09-16:37:20.467.336 [mindspore/train/serialization.py:233] Execute parameter into net process.
[INFO] ME(29963:140462460516096,MainProcess):2020-03-09-16:37:20.467.649 [mindspore/train/serialization.py:268] Load parameter into net process finish.
============== Accuracy:{'acc': 0.8599358974358975} ==============
```

