# Copyright 2020 Huawei Technologies Co., Ltd
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ============================================================================
"""
LSTM Tutorial
The sample can be run on GPU.
"""
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
from mindspore.train.callback import ModelCheckpoint, CheckpointConfig, LossMonitor
# Install gensim with 'pip install gensim'
import gensim


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

def tokenizer(text):
    return [tok.lower() for tok in text.split(' ')]

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

def get_imdb_data(labels_data, features_data):
    data_list = []
    for i, (label, feature) in enumerate(zip(labels_data, features_data)):
        data_json = {"id": i,
                     "label": int(label),
                     "feature": feature.reshape(-1)}
        data_list.append(data_json)
    return data_list

def convert_to_mindrecord(embed_size, aclimdb_path, proprocess_path, glove_path):
    """ convert imdb dataset to mindrecord """
    num_shard = 4
    train_features, train_labels, test_features, test_labels, weight_np, _ = \
        preprocess(aclimdb_path, glove_path, embed_size)
    np.savetxt(os.path.join(proprocess_path, 'weight.txt'), weight_np)

    # write mindrecord
    schema_json = {"id": {"type": "int32"},
                   "label": {"type": "int32"},
                   "feature": {"type": "int32", "shape":[-1]}}

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
        # (64,500,300)
        embeddings = self.embedding(inputs)
        embeddings = self.trans(embeddings, self.perm)
        output, _ = self.encoder(embeddings, (self.h, self.c))
        # states[i] size(64,200)  -> encoding.size(64,400)
        encoding = self.concat((output[0], output[1]))
        outputs = self.decoder(encoding)
        return outputs


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


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='MindSpore LSTM Example')
    parser.add_argument('--preprocess', type=str, default='false', choices=['true', 'false'],
                        help='Whether to perform data preprocessing')
    parser.add_argument('--mode', type=str, default="train", choices=['train', 'test'],
                        help='implement phase, set to train or test')
    # Download dataset from 'http://ai.stanford.edu/~amaas/data/sentiment/aclImdb_v1.tar.gz' and extract to 'aclimdb_path'
    parser.add_argument('--aclimdb_path', type=str, default="./aclImdb",
                        help='path where the dataset is store')
    # Download glove from 'http://nlp.stanford.edu/data/glove.6B.zip' and extract to 'glove_path'
    # Add a new line '400000 300' at the beginning of 'glove.6B.300d.txt' with '40000' for total words and '300' for vector length
    parser.add_argument('--glove_path', type=str, default="./glove",
                        help='path where the glove is store')
    parser.add_argument('--preprocess_path', type=str, default="./preprocess",
                        help='path where the pre-process data is store')
    parser.add_argument('--ckpt_path', type=str, default="./ckpt", help='if mode is test, must provide\
                        path where the trained ckpt file')
    args = parser.parse_args()

    context.set_context(
        mode=context.GRAPH_MODE,
        save_graphs=False,
        device_target="GPU")

    if args.preprocess == 'true':
        print("============== Starting Data Pre-processing ==============")
        shutil.rmtree(args.preprocess_path)
        os.mkdir(args.preprocess_path)
        convert_to_mindrecord(cfg.embed_size, args.aclimdb_path, args.preprocess_path, args.glove_path)

    embedding_table = np.loadtxt(os.path.join(args.preprocess_path, "weight.txt")).astype(np.float32)
    network = SentimentNet(vocab_size=embedding_table.shape[0],
                           embed_size=cfg.embed_size,
                           num_hiddens=cfg.num_hiddens,
                           num_layers=cfg.num_layers,
                           bidirectional=cfg.bidirectional,
                           num_classes=cfg.num_classes,
                           weight=Tensor(embedding_table),
                           batch_size=cfg.batch_size)

    loss = nn.SoftmaxCrossEntropyWithLogits(is_grad=False, sparse=True)
    opt = nn.Momentum(network.trainable_params(), cfg.learning_rate, cfg.momentum)
    loss_cb = LossMonitor()
    model = Model(network, loss, opt, {'acc': Accuracy()})

    if args.mode == 'train':
        print("============== Starting Training ==============")
        ds_train = create_dataset(args.preprocess_path, cfg.batch_size, cfg.num_epochs, True)
        config_ck = CheckpointConfig(save_checkpoint_steps=cfg.save_checkpoint_steps,
                                     keep_checkpoint_max=cfg.keep_checkpoint_max)
        ckpoint_cb = ModelCheckpoint(prefix="lstm", directory=args.ckpt_path, config=config_ck)
        model.train(cfg.num_epochs, ds_train, callbacks=[ckpoint_cb, loss_cb])
    elif args.mode == 'test':
        print("============== Starting Testing ==============")
        ds_eval = create_dataset(args.preprocess_path, cfg.batch_size, 1, False)
        param_dict = load_checkpoint(args.ckpt_path)
        load_param_into_net(network, param_dict)
        acc = model.eval(ds_eval)
        print("============== Accuracy:{} ==============".format(acc))
    else:
        raise RuntimeError('mode should be train or test, rather than {}'.format(args.mode))
