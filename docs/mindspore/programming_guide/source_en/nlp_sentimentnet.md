# Realizing Sentiment Classification With SentimentNet

`CPU` `CPU` `Natural Language Processing` `Whole Process`

<a href="https://gitee.com/mindspore/docs/blob/master/docs/mindspore/programming_guide/source_en/nlp_sentimentnet.md" target="_blank"><img src="https://mindspore-website.obs.cn-north-4.myhuaweicloud.com/website-images/master/resource/_static/logo_source_en.png"></a>

## Overview

Sentiment classification is a subset of text classification in NLP, and is one of the most basic applications of NLP. It is a process of analyzing and inferencing affective states and subjective information, that is, analyzing whether a person's sentiment is positive or negative.

> Generally, sentiments are classified into three categories: positive, negative, and neutral. In most cases, only positive and negative sentiments are used for training regardless of the neutral sentiments. The following dataset is a good example.

[20 Newsgroups](http://qwone.com/~jason/20Newsgroups/) is a typical reference dataset for traditional text classification. It is a collection of approximately 20,000 news documents partitioned across 20 different newsgroups.
Some of the newsgroups are very closely related to each other (such as comp.sys.ibm.pc.hardware and comp.sys.mac.hardware), while others are highly unrelated (such as misc.forsale and soc.religion.christian).

In terms of the network itself, the network structure of text classification is roughly similar to that of sentiment classification. After mastering how to construct the sentiment classification network, it is easy to construct a similar network which can be used in a text classification task after fine-tuning some parameters.

In the service context, text classification is to analyze the objective content discussed in the text, but sentiment classification is to find a viewpoint, which is supported by the content in the text. For example, "Forrest Gump has a clear theme and smooth pacing, which is excellent." In the text classification, this sentence is classified into a "movie" theme, but in the sentiment classification, this movie review is used to explore whether the sentiment is positive or negative.

Compared with traditional text classification, sentiment classification is simpler and more practical. High-quality datasets can be collected from common shopping websites and movie websites to benefit the business domains. For example, based on the domain context, the system can automatically analyze opinions of specific types of customers on the current product, analyze sentiments by subject and user type, and even recommend products based on the analysis result, therefore to improve the conversion rate and bring more business benefits.

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

Download and unzip the Glove file, add a new line at the beginning of each unzipped file, which means that a total of 400,000 words are read, and each word is represented by a word vector of 300 latitudes.

```text
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

1. Load the dataset in use and process data if necessary.
2. Use the SentimentNet network based on LSTM to train data and generate a model.
    Long short-term memory (LSTM) is an artificial recurrent neural network (RNN) architecture used for processing and predicting an important event with a long interval and delay in a time sequence. For details, refer to the online documentation.
3. After the model is obtained, use the validation dataset to check the accuracy of model.

> The current sample is for the Ascend 910 AI processor. You can find the complete executable sample code at <https://gitee.com/mindspore/models/tree/master/official/nlp/lstm>.
>
> - `default_config.yaml, config_ascend.yaml`: some configurations in the network, including `batch size`, several epoch training, etc.
> - `src/config.py`: some configurations of the network, including the batch size and number of training epochs.
> - `src/dataset.py`: dataset related definition, including converted MindRecord file and preprocessed data.
> - `src/imdb.py`: the utility class for parsing IMDb dataset.
> - `src/lstm.py`: the definition of semantic net.
> - `train.py`: the training script.
> - `eval.py`: the evaluation script.

## Implementation

### Importing Library Files

The following are the required public modules and MindSpore modules and library files.

```python
import os
import numpy as np

from src.model_utils.config import config
from src.model_utils.moxing_adapter import moxing_wrapper
from src.dataset import convert_to_mindrecord
from src.dataset import lstm_create_dataset
from src.lr_schedule import get_lr
from src.lstm import SentimentNet

from mindspore import Tensor, nn, Model, context
from mindspore.nn import Accuracy
from mindspore.train.callback import LossMonitor, CheckpointConfig, ModelCheckpoint, TimeMonitor
from mindspore import load_param_into_net, load_checkpoint
from mindspore.communication import init, get_rank
from mindspore.context import ParallelMode
```

### Configuring Environment Information

1. The `parser` module is used to transfer necessary information for running, such as storage paths of the dataset and the GloVe file. In this way, the frequently changed configurations can be entered during runtime, which is more flexible.

    ```python
    def parse_cli_to_yaml(parser, cfg, helper=None, choices=None, cfg_path="default_config.yaml"):
        """
        Parse command line arguments to the configuration according to the default yaml.
        Args:
            parser: Parent parser.
            cfg: Base configuration.
            helper: Helper description.
            cfg_path: Path to the default yaml config.
        """
        parser = argparse.ArgumentParser(description="[REPLACE THIS at config.py]",
                                         parents=[parser])
        helper = {} if helper is None else helper
        choices = {} if choices is None else choices
        for item in cfg:
            if not isinstance(cfg[item], list) and not isinstance(cfg[item], dict):
                help_description = helper[item] if item in helper else "Please reference to {}".format(cfg_path)
                choice = choices[item] if item in choices else None
                if isinstance(cfg[item], bool):
                    parser.add_argument("--" + item, type=ast.literal_eval, default=cfg[item], choices=choice,
                                        help=help_description)
                else:
                    parser.add_argument("--" + item, type=type(cfg[item]), default=cfg[item], choices=choice,
                                        help=help_description)
        args = parser.parse_args()
        return args
    ```

2. Before implementing code, configure the necessary information, including the environment information, execution mode, backend information, and hardware information.

    ```python
    context.set_context(
        mode=context.GRAPH_MODE,
        save_graphs=False,
        device_target=config.device_target)
    ```

    For details about the API configuration, see the `context.set_context`.

### Preprocessing the Dataset

Convert the dataset format to the MindRecord format for MindSpore to read.

```python
if config.preprocess == "true":
    print("============== Starting Data Pre-processing ==============")
    convert_to_mindrecord(config.embed_size, config.aclimdb_path, config.preprocess_path, config.glove_path)
```

> After successful conversion, `mindrecord` files are generated under the directory `preprocess_path`. Usually, this operation does not need to be performed every time if the dataset is unchanged.
> For `convert_to_mindrecord`, you can find the complete definition at: <https://gitee.com/mindspore/models/blob/master/official/nlp/lstm/src/dataset.py>.
> It consists of two steps:
>
>1. Process the text dataset, including encoding, word segmentation, alignment, and processing the original GloVe data to adapt to the network structure.
>2. Convert the dataset format to the MindRecord format.

### Defining the Network

```python
embedding_table = np.loadtxt(os.path.join(config.preprocess_path, "weight.txt")).astype(np.float32)
if config.device_target == 'Ascend':
    pad_num = int(np.ceil(config.embed_size / 16) * 16 - config.embed_size)
    if pad_num > 0:
        embedding_table = np.pad(embedding_table, [(0, 0), (0, pad_num)], 'constant')
    config.embed_size = int(np.ceil(config.embed_size / 16) * 16)
network = SentimentNet(vocab_size=embedding_table.shape[0],
                       embed_size=config.embed_size,
                       num_hiddens=config.num_hiddens,
                       num_layers=config.num_layers,
                       bidirectional=config.bidirectional,
                       num_classes=config.num_classes,
                       weight=Tensor(embedding_table),
                       batch_size=config.batch_size)
```

> For `SentimentNet`, you can find the complete definition at: <https://gitee.com/mindspore/models/blob/master/official/nlp/lstm/src/lstm.py>.

### Pre-Training

The parameter `pre_trained` specifies the preloading CheckPoint file for pre-training, which is empty by default.

```python
if config.pre_trained:
    load_param_into_net(network, load_checkpoint(config.pre_trained))
```

### Defining the Optimizer and Loss Function

The sample code for defining the optimizer and loss function is as follows:

```python
loss = nn.SoftmaxCrossEntropyWithLogits(sparse=True, reduction='mean')
opt = nn.Momentum(network.trainable_params(), config.learning_rate, config.momentum)
loss_cb = LossMonitor()
```

### Training and Saving the Model

Load the corresponding dataset, configure the CheckPoint generation information, and train the model using the `model.train` API.

```python
model = Model(network, loss, opt, {'acc': Accuracy()})

print("============== Starting Training ==============")
config_ck = CheckpointConfig(save_checkpoint_steps=config.save_checkpoint_steps,
                             keep_checkpoint_max=config.keep_checkpoint_max)
ckpoint_cb = ModelCheckpoint(prefix="lstm", directory=config.ckpt_path, config=config_ck)
time_cb = TimeMonitor(data_size=ds_train.get_dataset_size())
if config.device_target == "CPU":
    model.train(config.num_epochs, ds_train, callbacks=[time_cb, ckpoint_cb, loss_cb], dataset_sink_mode=False)
else:
    model.train(config.num_epochs, ds_train, callbacks=[time_cb, ckpoint_cb, loss_cb])
print("============== Training Success ==============")
```

> For `lstm_create_dataset`, you can find the complete definition at: <https://gitee.com/mindspore/models/blob/master/official/nlp/lstm/src/dataset.py>.

### Validating the Model

Load the validation dataset and saved CheckPoint file, perform validation, and view the model quality.

```python
model = Model(network, loss, metrics={'acc': Accuracy(), 'recall': Recall(), 'f1': F1()})

print("============== Starting Testing ==============")
param_dict = load_checkpoint(config.ckpt_file)
load_param_into_net(network, param_dict)
if config.device_target == "CPU":
    acc = model.eval(ds_eval, dataset_sink_mode=False)
else:
    acc = model.eval(ds_eval)
print("============== {} ==============".format(acc))
```

## Experimental Result

After 20 epochs, the accuracy on the test set is about 84.19%.

**Training Execution:**

1. Run the training code and view the running result.

    ```bash
    python train.py --config_path=CONFIG_FILE --device_target="Ascend" --aclimdb_path=$ACLIMDB_DIR --glove_path=$GLOVE_DIR --preprocess=true --preprocess_path=./preprocess > log.txt 2>&1 &
    ```

    As shown in the following output, the loss value decreases gradually with the training process and reaches about 0.2855.

    ```text
    ============== Starting Data Pre-processing ==============
    vocab_size:  252192
    ============== Starting Training ==============
    epoch: 1 step: 1, loss is 0.6935
    epoch: 1 step: 2, loss is 0.6924
    ...
    epoch: 10 step: 389, loss is 0.2675
    epoch: 10 step: 390, loss is 0.3232
    ...
    epoch: 20 step: 389, loss is 0.1354
    epoch: 20 step: 390, loss is 0.2855
    ```

2. Check the saved CheckPoint files.

   CheckPoint files (model files) are saved during the training. You can view all saved files in the file path.

    ```bash
    ls ./ckpt_lstm/*.ckpt
    ```

    The output is as follows:

    ```text
    lstm-11_390.ckpt  lstm-12_390.ckpt  lstm-13_390.ckpt  lstm-14_390.ckpt  lstm-15_390.ckpt  lstm-16_390.ckpt  lstm-17_390.ckpt  lstm-18_390.ckpt  lstm-19_390.ckpt  lstm-20_390.ckpt
    ```

**Model Validation:**

Use the last saved CheckPoint file to load and validate the dataset.

```bash
python eval.py --config_path=$CONFIG_FILE --device_target="Ascend" --preprocess=false --preprocess_path=$PREPROCESS_DIR --ckpt_file=$CKPT_FILE > log.txt 2>&1 &
```

Parameter interpretation:

- `--config_path`: The path of the parameter file, i.e. the source code `default_config.yaml` file.
- `--device_target`: The device used for model training, `Ascend` is selected in this article, The options are `CPU`、`GPU` and `Ascend`
- `--preprocess`: Preprocess data or not.
- `--preprocess_path`: Preprocessed dataset path.
- `--ckpt_file`: The path to load the model weights file. (use `./ckpt_lstm/lstm-20_390.ckpt`)

As shown in the following output, the sentiment analysis accuracy of the text is about 84.19%, which is basically satisfactory.

```text
============== Starting Testing ==============
============== {'acc': 0.8419471153846154} ==============
```
