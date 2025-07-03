[![View Source On Gitee](https://mindspore-website.obs.cn-north-4.myhuaweicloud.com/website-images/master/resource/_static/logo_source_en.svg)](https://gitee.com/mindspore/docs/blob/master/tutorials/source_en/nlp/sentiment_analysis.md)

# Sentiment Classification Implemented by RNN

## Overview

Sentiment classification is a classic task in natural language processing. It is a typical classification problem. The following uses MindSpore to implement an RNN-based sentimental classification model to achieve the following effects:

```text
Input: This film is terrible
Correct label: Negative
Forecast label: Negative

Input: This film is great
Correct label: Positive
Forecast label: Positive
```

## Data Preparation

This section uses the classic [IMDB Movie Review Dataset](https://ai.stanford.edu/~amaas/data/sentiment/) for sentimental classification. The dataset contains positive and negative data. The following is an example:

| Review  | Label  |
|:---|:---:|
| "Quitting" may be as much about exiting a pre-ordained identity as about drug withdrawal. As a rural guy coming to Beijing, class and success must have struck this young artist face on as an appeal to separate from his roots and far surpass his peasant parents' acting success. Troubles arise, however, when the new man is too new, when it demands too big a departure from family, history, nature, and personal identity. The ensuing splits, and confusion between the imaginary and the real and the dissonance between the ordinary and the heroic are the stuff of a gut check on the one hand or a complete escape from self on the other.  |  Negative |
| This movie is amazing because the fact that the real people portray themselves and their real life experience and do such a good job it's like they're almost living the past over again. Jia Hongsheng plays himself an actor who quit everything except music and drugs struggling with depression and searching for the meaning of life while being angry at everyone especially the people who care for him most.  | Positive  |

In addition, the pre-trained word vectors are used to encode natural language words to obtain semantic features of text. In this section, the Global Vectors for Word Representation ([GloVe](https://nlp.stanford.edu/projects/glove/)) are selected as embeddings.

### Data Downloading Module

To facilitate the download of datasets and pre-trained word vectors, a data download module is designed to implement a visualized download process and save the data to a specified path. The data download module uses the `requests` library to send HTTP requests and uses the `tqdm` library to visualize the download percentage. To ensure download security, temporary files are downloaded in I/O mode, saved to a specified path, and returned.

> The `tqdm` and `requests` libraries need to be manually installed. The command is `pip install tqdm requests`.

```python
import os
import shutil
import requests
import tempfile
from tqdm import tqdm
from typing import IO
from pathlib import Path

# Set the storage path to `home_path/.mindspore_examples`.
cache_dir = Path.home() / '.mindspore_examples'

def http_get(url: str, temp_file: IO):
    """Download data by using the requests library and visualize the process by using the tqdm library."""
    req = requests.get(url, stream=True)
    content_length = req.headers.get('Content-Length')
    total = int(content_length) if content_length is not None else None
    progress = tqdm(unit='B', total=total)
    for chunk in req.iter_content(chunk_size=1024):
        if chunk:
            progress.update(len(chunk))
            temp_file.write(chunk)
    progress.close()

def download(file_name: str, url: str):
    """Download data and save it with the specified name."""
    if not os.path.exists(cache_dir):
        os.makedirs(cache_dir)
    cache_path = os.path.join(cache_dir, file_name)
    cache_exist = os.path.exists(cache_path)
    if not cache_exist:
        with tempfile.NamedTemporaryFile() as temp_file:
            http_get(url, temp_file)
            temp_file.flush()
            temp_file.seek(0)
            with open(cache_path, 'wb') as cache_file:
                shutil.copyfileobj(temp_file, cache_file)
    return cache_path
```

After the data download module is complete, download the IMDB dataset for testing. The HUAWEI CLOUD image is used to improve the download speed. The download process and storage path are as follows:

```python
imdb_path = download('aclImdb_v1.tar.gz', 'https://mindspore-website.obs.myhuaweicloud.com/notebook/datasets/aclImdb_v1.tar.gz')
imdb_path
```

```text
'/root/.mindspore_examples/aclImdb_v1.tar.gz'
```

### Loading the IMDB Dataset

The downloaded IMDB dataset is a `tar.gz` file. Use the `tarfile` library of Python to read the dataset and store all data and labels separately. The decompression directory of the original IMDB dataset is as follows:

```text
├── aclImdb
│   ├── imdbEr.txt
│   ├── imdb.vocab
│   ├── README
│   ├── test
│   └── train
│         ├── neg
│         ├── pos
...
```

The dataset has been divided into two parts: train and test. Each part contains the neg and pos folders. You need to use the train and test parts to read and process data and labels, respectively.

```python
import re
import six
import string
import tarfile

class IMDBData():
    """IMDB dataset loader.

    Load the IMDB dataset and process it as a Python iteration object.

    """
    label_map = {
        "pos": 1,
        "neg": 0
    }
    def __init__(self, path, mode="train"):
        self.mode = mode
        self.path = path
        self.docs, self.labels = [], []

        self._load("pos")
        self._load("neg")

    def _load(self, label):
        pattern = re.compile(r"aclImdb/{}/{}/.*\.txt$".format(self.mode, label))
        # Load data to the memory.
        with tarfile.open(self.path) as tarf:
            tf = tarf.next()
            while tf is not None:
                if bool(pattern.match(tf.name)):
                    # Segment text, remove punctuations and special characters, and convert text to lowercase.
                    self.docs.append(str(tarf.extractfile(tf).read().rstrip(six.b("\n\r"))
                                         .translate(None, six.b(string.punctuation)).lower()).split())
                    self.labels.append([self.label_map[label]])
                tf = tarf.next()

    def __getitem__(self, idx):
        return self.docs[idx], self.labels[idx]

    def __len__(self):
        return len(self.docs)
```

After the IMDB dataset loader is completed, load the training dataset for testing and output the number of datasets.

```python
imdb_train = IMDBData(imdb_path, 'train')
len(imdb_train)
```

```text
25000
```

After the IMDB dataset is loaded to the memory and built as an iteration object, you can use the [Generatordataset](https://www.mindspore.cn/docs/en/master/api_python/dataset/mindspore.dataset.GeneratorDataset.html) API provided by `mindspore.dataset` to load the dataset iteration object and then perform data processing. The following encapsulates a function to load train and test using `GeneratorDataset`, and set `column_name` of the text and label in the dataset to `text` and `label`, respectively.

```python
import mindspore.dataset as ds

def load_imdb(imdb_path):
    imdb_train = ds.GeneratorDataset(IMDBData(imdb_path, "train"), column_names=["text", "label"], shuffle=True)
    imdb_test = ds.GeneratorDataset(IMDBData(imdb_path, "test"), column_names=["text", "label"], shuffle=False)
    return imdb_train, imdb_test
```

Load the IMDB dataset. You can see that `imdb_train` is a GeneratorDataset object.

```python
imdb_train, imdb_test = load_imdb(imdb_path)
imdb_train
```

```text
<mindspore.dataset.engine.datasets_user_defined.GeneratorDataset at 0x7fa6cd168ed0>
```

### Loading Pre-trained Word Vectors

A pre-trained word vector is a numerical representation of an input word. The `nn.Embedding` layer uses the table lookup mode to input the index in the vocabulary corresponding to the word to obtain the corresponding expression vector.
Therefore, before model build, word vectors and vocabulary required by the Embedding layer need to be built. Here, we use the classic pre-trained word vectors, GloVe.
The data format is as follows:

| Word |  Vector |
|:---|:---:|
| the | 0.418 0.24968 -0.41242 0.1217 0.34527 -0.044457 -0.49688 -0.17862 -0.00066023 ...|
| , | 0.013441 0.23682 -0.16899 0.40951 0.63812 0.47709 -0.42852 -0.55641 -0.364 ... |

The words in the first column are used as the vocabulary, and `dataset.text.Vocab` is used to load the words in sequence. In addition, the vector of each row is read and converted into `numpy.array` for the `nn.Embedding` to load weights. The sample code is as follows:

```python
import zipfile
import numpy as np

def load_glove(glove_path):
    glove_100d_path = os.path.join(cache_dir, 'glove.6B.100d.txt')
    if not os.path.exists(glove_100d_path):
        glove_zip = zipfile.ZipFile(glove_path)
        glove_zip.extractall(cache_dir)

    embeddings = []
    tokens = []
    with open(glove_100d_path, encoding='utf-8') as gf:
        for glove in gf:
            word, embedding = glove.split(maxsplit=1)
            tokens.append(word)
            embeddings.append(np.fromstring(embedding, dtype=np.float32, sep=' '))
    # Add the embeddings corresponding to the special placeholders <unk> and <pad>.
    embeddings.append(np.random.rand(100))
    embeddings.append(np.zeros((100,), np.float32))

    vocab = ds.text.Vocab.from_list(tokens, special_tokens=["<unk>", "<pad>"], special_first=False)
    embeddings = np.array(embeddings).astype(np.float32)
    return vocab, embeddings
```

The dataset may contain words that are not covered by the vocabulary. Therefore, the `<unk>` token needs to be added. In addition, because the input lengths are different, the `<pad>` tokens need to be added to short text when the text is packed into a batch. The length of the completed vocabulary is the length of the original vocabulary plus 2.

Download the GloVe to generate and load a vocabulary and a word vector weight matrix.

```python
glove_path = download('glove.6B.zip', 'https://mindspore-website.obs.myhuaweicloud.com/notebook/datasets/glove.6B.zip')
vocab, embeddings = load_glove(glove_path)
len(vocab.vocab())
```

```text
400002
```

Use a vocabulary to convert `the` into an index ID, and query a word vector corresponding to the word vector matrix:

```python
idx = vocab.tokens_to_ids('the')
embedding = embeddings[idx]
idx, embedding
```

```text
(0,
 array([-0.038194, -0.24487 ,  0.72812 , -0.39961 ,  0.083172,  0.043953,
        -0.39141 ,  0.3344  , -0.57545 ,  0.087459,  0.28787 , -0.06731 ,
         0.30906 , -0.26384 , -0.13231 , -0.20757 ,  0.33395 , -0.33848 ,
        -0.31743 , -0.48336 ,  0.1464  , -0.37304 ,  0.34577 ,  0.052041,
         0.44946 , -0.46971 ,  0.02628 , -0.54155 , -0.15518 , -0.14107 ,
        -0.039722,  0.28277 ,  0.14393 ,  0.23464 , -0.31021 ,  0.086173,
         0.20397 ,  0.52624 ,  0.17164 , -0.082378, -0.71787 , -0.41531 ,
         0.20335 , -0.12763 ,  0.41367 ,  0.55187 ,  0.57908 , -0.33477 ,
        -0.36559 , -0.54857 , -0.062892,  0.26584 ,  0.30205 ,  0.99775 ,
        -0.80481 , -3.0243  ,  0.01254 , -0.36942 ,  2.2167  ,  0.72201 ,
        -0.24978 ,  0.92136 ,  0.034514,  0.46745 ,  1.1079  , -0.19358 ,
        -0.074575,  0.23353 , -0.052062, -0.22044 ,  0.057162, -0.15806 ,
        -0.30798 , -0.41625 ,  0.37972 ,  0.15006 , -0.53212 , -0.2055  ,
        -1.2526  ,  0.071624,  0.70565 ,  0.49744 , -0.42063 ,  0.26148 ,
        -1.538   , -0.30223 , -0.073438, -0.28312 ,  0.37104 , -0.25217 ,
         0.016215, -0.017099, -0.38984 ,  0.87424 , -0.72569 , -0.51058 ,
        -0.52028 , -0.1459  ,  0.8278  ,  0.27062 ], dtype=float32))
```

## Dataset Preprocessing

Word segmentation is performed on the IMDB dataset loaded by the loader, but the dataset does not meet the requirements for building training data. Therefore, extra preprocessing is required. The preprocessing is as follows:

- Use the Vocab to convert all tokens to index IDs.
- The length of the text sequence is unified. If the length is insufficient, `<pad>` is used to supplement the length. If the length exceeds the limit, the excess part is truncated.

Here, the API provided in `mindspore.dataset` is used for preprocessing. The APIs used here are designed for MindSpore high-performance data engines. The operations corresponding to each API are considered as a part of the data pipeline. For details, see [MindSpore Data Engine](https://www.mindspore.cn/docs/en/master/design/data_engine.html).

For the table query operation from a token to an index ID, use the `text.Lookup` API to load the built vocabulary and specify `unknown_token`. The [PadEnd](https://www.mindspore.cn/docs/en/master/api_python/dataset_transforms/mindspore.dataset.transforms.PadEnd.html) API is used to unify the length of the text sequence. This API defines the maximum length and padding value (`pad_value`). In this example, the maximum length is 500, and the padding value corresponds to the index ID of `<pad>` in the vocabulary.

> In addition to pre-processing the `text` data in the dataset, the `label` data needs to be converted to the float32 format to meet the subsequent model training requirements.

```python
import mindspore as ms

lookup_op = ds.text.Lookup(vocab, unknown_token='<unk>')
pad_op = ds.transforms.PadEnd([500], pad_value=vocab.tokens_to_ids('<pad>'))
type_cast_op = ds.transforms.TypeCast(ms.float32)
```

After the preprocessing is complete, you need to add data to the dataset processing pipeline and use the [map](https://www.mindspore.cn/docs/en/master/api_python/dataset/dataset_method/operation/mindspore.dataset.Dataset.map.html#mindspore.dataset.Dataset.map) API to add operations to the specified column.

```python
imdb_train = imdb_train.map(operations=[lookup_op, pad_op], input_columns=['text'])
imdb_train = imdb_train.map(operations=[type_cast_op], input_columns=['label'])

imdb_test = imdb_test.map(operations=[lookup_op, pad_op], input_columns=['text'])
imdb_test = imdb_test.map(operations=[type_cast_op], input_columns=['label'])
```

The IMDB dataset does not contain the validation set. Therefore, you need to manually divide the dataset into training and validation parts, with the ratio of 0.7 to 0.3.

```python
imdb_train, imdb_valid = imdb_train.split([0.7, 0.3])
```

Finally, specify the batch size of the dataset by using the [batch](https://www.mindspore.cn/docs/en/master/api_python/dataset/dataset_method/operation/mindspore.dataset.Dataset.batch.html#mindspore.dataset.Dataset.batch) API and determine whether to discard the remaining data that cannot be exactly divided by the batch size.

> Call the `map`, `split`, and `batch` APIs of the dataset to add corresponding operations to the dataset processing pipeline. The return value is of the new dataset type. Currently, only the pipeline operation is defined. During execution, the data processing pipeline is executed to obtain the processed data and send the data to the model for training.

```python
imdb_train = imdb_train.batch(64, drop_remainder=True)
imdb_valid = imdb_valid.batch(64, drop_remainder=True)
```

## Model Building

After the dataset is processed, we design the model structure for sentimental classification. First, the input text (that is, the serialized index ID list) needs to be converted into a vectorized representation through table lookup. In this case, the `nn.Embedding` layer needs to be used to load the GloVe, and then the RNN is used to perform feature extraction. Finally, the RNN is connected to a fully-connected layer, that is, `nn.Dense`, to convert the feature into a size that is the same as the number of classifications for subsequent model optimization training. The overall model structure is as follows:

```text
nn.Embedding -> nn.RNN -> nn.Dense
```

The LSTM(Long short-term memory) variant that can avoid the RNN gradient vanishing problem is used as the feature extraction layer. The model is described as follows:

### Embedding

The Embedding layer may also be referred to as an EmbeddingLookup layer. A function of the Embedding layer is to use an index ID to search for a vector of an ID corresponding to the weight matrix. When an input is a sequence including index IDs, a matrix with a same length is searched for and returned. For example:

```text
embedding = nn.Embedding(1000, 100) # The size of the vocabulary (the value range of index) is 1000, and the size of the vector is 100.
input shape: (1, 16)                # The sequence length is 16.
output shape: (1, 16, 100)
```

Here, the processed GloVe word vector matrix is used. `embedding_table` of `nn.Embedding` is set to the pre-trained word vector matrix. The vocabulary size `vocab_size` is 400002, and `embedding_size` is the size of the selected `glove.6B.100d` vector, that is, 100.

### Recurrent Neural Network (RNN)

RNN is a type of neural network that uses sequence data as an input, performs recursion in the evolution direction of a sequence, and connects all nodes (circulating units) in a chain. The following figure shows the general RNN structure.

![RNN-0](https://mindspore-website.obs.cn-north-4.myhuaweicloud.com/website-images/master/tutorials/source_zh_cn/nlp/images/0-RNN-0.png)

> The left part of the figure shows an RNN Cell cycle, and the right part shows the RNN chain connection. Actually, there is only one Cell parameter regardless of a single RNN Cell or an RNN network, and the parameter is updated in continuous cyclic calculation.

The recurrent feature of the RNN matches the sequence feature (a sentence is a sequence composed of words) of the natural language text. Therefore, the RNN is widely used in the research of natural language processing. The following figure shows the disassembled RNN structure.

![RNN](https://mindspore-website.obs.cn-north-4.myhuaweicloud.com/website-images/master/tutorials/source_zh_cn/nlp/images/0-RNN.png)

A structure of a single RNN Cell is simple, causing the gradient vanishing problem. Specifically, when a sequence in the RNN is relatively long, information of a sequence header is basically lost at a tail of the sequence. To solve this problem, the LSTM(Long short-term memory) is proposed. The gating mechanism is used to control the retention and discarding of information flows in each cycle. The following figure shows the disassembled LSTM structure.

![LSTM](https://mindspore-website.obs.cn-north-4.myhuaweicloud.com/website-images/master/tutorials/source_zh_cn/nlp/images/0-LSTM.png)

In this section, the LSTM variant instead of the classic RNN is used for feature extraction to avoid the gradient vanishing problem and obtain a better model effect. The formula corresponding to `nn.LSTM` in MindSpore is as follows:

$$h_{0:t}, (h_t, c_t) = \text{LSTM}(x_{0:t}, (h_0, c_0))$$

Herein, `nn.LSTM` hides a cycle of the entire recurrent neural network on a sequence time step. After the input sequence and the initial state are sent, you can obtain a matrix formed by splicing hidden states of each time step and a hidden state corresponding to the last time step. We use the hidden state of the last time step as the encoding feature of the input sentence and send it to the next layer.

> Time step: Each cycle calculated by the recurrent neural network is a time step. When a text sequence is sent, a time step corresponds to a word. Therefore, in this example, the output $h_{0:t}$ of the LSTM corresponds to the hidden state set of each word, and $h_t$ and $c_t$ correspond to the hidden state corresponding to the last word.

### Dense

After the sentence feature is obtained through LSTM encoding, the sentence feature is sent to a fully-connected layer, that is, `nn.Dense`. The feature dimension is converted into dimension 1 required for binary classification. The output after passing through the Dense layer is the model prediction result.

```python
import math
import mindspore as ms
import mindspore.nn as nn
import mindspore.ops as ops
from mindspore.common.initializer import Uniform, HeUniform

class RNN(nn.Cell):
    def __init__(self, embeddings, hidden_dim, output_dim, n_layers,
                 bidirectional, pad_idx):
        super().__init__()
        vocab_size, embedding_dim = embeddings.shape
        self.embedding = nn.Embedding(vocab_size, embedding_dim, embedding_table=ms.Tensor(embeddings), padding_idx=pad_idx)
        self.rnn = nn.LSTM(embedding_dim,
                           hidden_dim,
                           num_layers=n_layers,
                           bidirectional=bidirectional,
                           batch_first=True)
        weight_init = HeUniform(math.sqrt(5))
        bias_init = Uniform(1 / math.sqrt(hidden_dim * 2))
        self.fc = nn.Dense(hidden_dim * 2, output_dim, weight_init=weight_init, bias_init=bias_init)

    def construct(self, inputs):
        embedded = self.embedding(inputs)
        _, (hidden, _) = self.rnn(embedded)
        hidden = ops.concat((hidden[-2, :, :], hidden[-1, :, :]), axis=1)
        output = self.fc(hidden)
        return output
```

### Loss Function and Optimizer

After the model body is built, instantiate the network based on the specified parameters, select the loss function and optimizer. For a feature of the sentimental classification problem in this section, that is, a binary classification problem for predicting positive or negative, [nn.BCEWithLogitsLoss](https://www.mindspore.cn/docs/en/master/api_python/nn/mindspore.nn.BCEWithLogitsLoss.html) (binary cross entropy loss function) is selected.

```python
hidden_size = 256
output_size = 1
num_layers = 2
bidirectional = True
lr = 0.001
pad_idx = vocab.tokens_to_ids('<pad>')

model = RNN(embeddings, hidden_size, output_size, num_layers, bidirectional, pad_idx)
loss_fn = nn.BCEWithLogitsLoss(reduction='mean')
optimizer = nn.Adam(model.trainable_params(), learning_rate=lr)
```

### Training Logic

After the model is built, design the training logic. Generally, the training logic consists of the following steps:

1. Read the data of a batch.
2. Send the data to the network for forward computation and backward propagation, and update the weight.
3. Return the loss.

Based on this logic, use the `tqdm` library to design an epoch training function for visualization of the training process and loss.

```python
def forward_fn(data, label):
    logits = model(data)
    loss = loss_fn(logits, label)
    return loss

grad_fn = ms.value_and_grad(forward_fn, None, optimizer.parameters)

def train_step(data, label):
    loss, grads = grad_fn(data, label)
    optimizer(grads)
    return loss

def train_one_epoch(model, train_dataset, epoch=0):
    model.set_train()
    total = train_dataset.get_dataset_size()
    loss_total = 0
    step_total = 0
    with tqdm(total=total) as t:
        t.set_description('Epoch %i' % epoch)
        for i in train_dataset.create_tuple_iterator():
            loss = train_step(*i)
            loss_total += loss.asnumpy()
            step_total += 1
            t.set_postfix(loss=loss_total/step_total)
            t.update(1)
```

### Evaluation Metrics and Logic

After the training logic is complete, you need to evaluate the model. That is, compare the prediction result of the model with the correct label of the test set to obtain the prediction accuracy. Because sentimental classification of the IMDB is a binary classification problem, you can directly round off the predicted value to obtain a classification label (0 or 1), and then determine whether the classification label is equal to a correct label. The following describes the implementation of the function for calculating the binary classification accuracy:

```python
def binary_accuracy(preds, y):
    """
    Calculate the accuracy of each batch.
    """

    # Round off the predicted value.
    rounded_preds = np.around(ops.sigmoid(preds).asnumpy())
    correct = (rounded_preds == y).astype(np.float32)
    acc = correct.sum() / len(correct)
    return acc
```

After the accuracy calculation function is available, similar to the training logic, the evaluation logic is designed in the following steps:

1. Read the data of a batch.
2. Send the data to the network for forward computation to obtain the prediction result.
3. Calculate the accuracy.

Similar to the training logic, `tqdm` is used to visualize the loss and process. In addition, the loss evaluation result is returned for determining the model quality when the model is saved.

> During the evaluation, the model used is the network body that does not contain the loss function and optimizer.
> Before evaluation, you need to use `model.set_train(False)` to set the model to the evaluation state. In this case, Dropout does not take effect.

```python
def evaluate(model, test_dataset, criterion, epoch=0):
    total = test_dataset.get_dataset_size()
    epoch_loss = 0
    epoch_acc = 0
    step_total = 0
    model.set_train(False)

    with tqdm(total=total) as t:
        t.set_description('Epoch %i' % epoch)
        for i in test_dataset.create_tuple_iterator():
            predictions = model(i[0])
            loss = criterion(predictions, i[1])
            epoch_loss += loss.asnumpy()

            acc = binary_accuracy(predictions, i[1])
            epoch_acc += acc

            step_total += 1
            t.set_postfix(loss=epoch_loss/step_total, acc=epoch_acc/step_total)
            t.update(1)

    return epoch_loss / total
```

## Model Training and Saving

The model building, training, and evaluation logic design are complete. The following describes how to train a model. In this example, the number of training epochs is set to 5. In addition, maintain the `best_valid_loss` variable for saving the optimal model. Based on the loss value of each epoch of evaluation, select the epoch with the minimum loss value and save the model.

```python
num_epochs = 5
best_valid_loss = float('inf')
ckpt_file_name = os.path.join(cache_dir, 'sentiment-analysis.ckpt')

for epoch in range(num_epochs):
    train_one_epoch(model, imdb_train, epoch)
    valid_loss = evaluate(model, imdb_valid, loss_fn, epoch)

    if valid_loss < best_valid_loss:
        best_valid_loss = valid_loss
        ms.save_checkpoint(model, ckpt_file_name)
```

```text
Epoch 0: 100%|██████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████| 273/273 [00:30<00:00,  9.00it/s, loss=0.674]
Epoch 0: 100%|███████████████████████████████████████████████████████████████████████████████████████████████████████████████| 117/117 [00:12<00:00,  9.43it/s, acc=0.511, loss=0.692]
Epoch 1: 100%|██████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████| 273/273 [00:24<00:00, 11.04it/s, loss=0.683]
Epoch 1: 100%|███████████████████████████████████████████████████████████████████████████████████████████████████████████████| 117/117 [00:11<00:00, 10.06it/s, acc=0.674, loss=0.614]
Epoch 2: 100%|██████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████| 273/273 [00:24<00:00, 11.06it/s, loss=0.623]
Epoch 2: 100%|███████████████████████████████████████████████████████████████████████████████████████████████████████████████| 117/117 [00:11<00:00, 10.12it/s, acc=0.799, loss=0.458]
Epoch 3: 100%|██████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████| 273/273 [00:24<00:00, 10.95it/s, loss=0.408]
Epoch 3: 100%|███████████████████████████████████████████████████████████████████████████████████████████████████████████████| 117/117 [00:11<00:00, 10.19it/s, acc=0.875, loss=0.306]
Epoch 4: 100%|██████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████| 273/273 [00:24<00:00, 11.03it/s, loss=0.305]
Epoch 4: 100%|███████████████████████████████████████████████████████████████████████████████████████████████████████████████| 117/117 [00:11<00:00,  9.93it/s, acc=0.899, loss=0.251]
```

You can see that the loss decreases gradually in each epoch and the accuracy of the verification set increases gradually.

## Model Loading and Testing

After model training is complete, you need to test or deploy the model. In this case, you need to load the saved optimal model (that is, checkpoint) for subsequent tests. The checkpoint loading and network weight loading APIs provided by MindSpore are used to load the saved model checkpoint to the memory and load the checkpoint to the model.

> The [load_param_into_net](https://www.mindspore.cn/docs/en/master/api_python/mindspore/mindspore.load_param_into_net.html) API returns the weight name that does not match the checkpoint in the model. If the weight name matches the checkpoint, an empty list is returned.

```python
param_dict = ms.load_checkpoint(ckpt_file_name)
ms.load_param_into_net(model, param_dict)
```

```text
[]
```

Batch the test set, and then use the evaluation method to evaluate the effect of the model on the test set.

```python
imdb_test = imdb_test.batch(64)
evaluate(model, imdb_test, loss_fn)
```

```text
Epoch 0: 100%|███████████████████████████████████████████████████████████████████████████████████████████████████████████████| 391/391 [00:25<00:00, 15.10it/s, acc=0.857, loss=0.357]
```

```text
0.356888720432244
```

## Custom Input Test

Finally, we design a prediction function to implement the effect described at the beginning. Enter a comment to obtain the sentimental classification of the comment. Specifically, the following steps are included:

1. Perform word segmentation on an input sentence.
2. Obtain index ID sequence by using the vocabulary.
3. Convert the index IDs sequence into tensors.
4. Send tensors to the model to obtain the prediction result.
5. Print the prediction result.

The sample code is as follows:

```python
score_map = {
    1: "Positive",
    0: "Negative"
}

def predict_sentiment(model, vocab, sentence):
    model.set_train(False)
    tokenized = sentence.lower().split()
    indexed = vocab.tokens_to_ids(tokenized)
    tensor = ms.Tensor(indexed, ms.int32)
    tensor = tensor.expand_dims(0)
    prediction = model(tensor)
    return score_map[int(np.round(ops.sigmoid(prediction).asnumpy()))]
```

Finally, predict the examples in the preceding section. It shows that the model can classify the sentiments of the statements.

```python
predict_sentiment(model, vocab, "This film is terrible")
```

```text
'Negative'
```

```python
predict_sentiment(model, vocab, "This film is great")
```

```text
'Positive'
```
