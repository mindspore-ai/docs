# 使用SentimentNet实现情感分类

`GPU` `CPU` `自然语言处理` `全流程`

<a href="https://authoring-modelarts-cnnorth4.huaweicloud.com/console/lab?share-url-b64=aHR0cHM6Ly9vYnMuZHVhbHN0YWNrLmNuLW5vcnRoLTQubXlodWF3ZWljbG91ZC5jb20vbWluZHNwb3JlLXdlYnNpdGUvbm90ZWJvb2svbW9kZWxhcnRzL21pbmRzcG9yZV9ubHBfYXBwbGljYXRpb24uaXB5bmI=&imageid=65f636a0-56cf-49df-b941-7d2a07ba8c8c" target="_blank"><img src="https://gitee.com/mindspore/docs/raw/r1.6/resource/_static/logo_modelarts.png"></a>&nbsp;&nbsp;
<a href="https://obs.dualstack.cn-north-4.myhuaweicloud.com/mindspore-website/notebook/r1.6/notebook/mindspore_nlp_application.ipynb"><img src="https://gitee.com/mindspore/docs/raw/r1.6/resource/_static/logo_notebook.png"></a>&nbsp;&nbsp;
<a href="https://obs.dualstack.cn-north-4.myhuaweicloud.com/mindspore-website/notebook/r1.6/notebook/mindspore_nlp_application.py"><img src="https://gitee.com/mindspore/docs/raw/r1.6/resource/_static/logo_download_code.png"></a>&nbsp;&nbsp;
<a href="https://gitee.com/mindspore/docs/blob/r1.6/docs/mindspore/programming_guide/source_zh_cn/nlp_sentimentnet.md" target="_blank"><img src="https://gitee.com/mindspore/docs/raw/r1.6/resource/_static/logo_source.png"></a>

## 概述

情感分类是自然语言处理中文本分类问题的子集，属于自然语言处理最基础的应用。它是对带有感情色彩的主观性文本进行分析和推理的过程，即分析说话人的态度，是倾向正面还是反面。

> 通常情况下，我们会把情感类别分为正面、反面和中性三类。虽然“面无表情”的评论也有不少；不过，大部分时候会只采用正面和反面的案例进行训练，下面这个数据集就是很好的例子。

传统的文本主题分类问题的典型参考数据集为[20 Newsgroups](http://qwone.com/~jason/20Newsgroups/)，该数据集由20组新闻数据组成，包含约20000个新闻文档。
其主题列表中有些类别的数据比较相似，例如comp.sys.ibm.pc.hardware和comp.sys.mac.hardware都是和电脑系统硬件相关的题目，相似度比较高。而有些主题类别的数据相对来说就毫无关联，例如misc.forsale和soc.religion.christian。

就网络本身而言，文本主题分类的网络结构和情感分类的网络结构大致相似。在掌握了情感分类网络如何构造之后，可以很容易构造一个类似的网络，稍作调参即可用于文本主题分类任务。

但在业务上下文侧，文本主题分类是分析文本讨论的客观内容，而情感分类是要从文本中得到它是否支持某种观点的信息。比如，“《阿甘正传》真是好看极了，影片主题明确，节奏流畅。”这句话，在文本主题分类是要将其归为类别为“电影”主题，而情感分类则要挖掘出这一影评的态度是正面还是负面。

相对于传统的文本主题分类，情感分类较为简单，实用性也较强。常见的购物网站、电影网站都可以采集到相对高质量的数据集，也很容易给业务领域带来收益。例如，可以结合领域上下文，自动分析特定类型客户对当前产品的意见，可以分主题分用户类型对情感进行分析，以作针对性的处理，甚至基于此进一步推荐产品，提高转化率，带来更高的商业收益。

特殊领域中，某些非极性词也充分表达了用户的情感倾向，比如下载使用APP时，“卡死了”、“下载太慢了”就表达了用户的负面情感倾向；股票领域中，“看涨”、“牛市”表达的就是用户的正面情感倾向。所以，本质上，我们希望模型能够在垂直领域中，挖掘出一些特殊的表达，作为极性词给情感分类系统使用：

$垂直极性词 = 通用极性词 + 领域特有极性词$

按照处理文本的粒度不同，情感分析可分为词语级、短语级、句子级、段落级以及篇章级等几个研究层次。这里以“段落级”为例，输入为一个段落，输出为影评是正面还是负面的信息。

## 准备及设计

### 下载数据集

采用IMDb影评数据集作为实验数据。
> 数据集下载地址：<http://ai.stanford.edu/~amaas/data/sentiment/>

以下是负面影评（Negative）和正面影评（Positive）的案例。

| Review  | Label  |
|---|---|
| "Quitting" may be as much about exiting a pre-ordained identity as about drug withdrawal. As a rural guy coming to Beijing, class and success must have struck this young artist face on as an appeal to separate from his roots and far surpass his peasant parents' acting success. Troubles arise, however, when the new man is too new, when it demands too big a departure from family, history, nature, and personal identity. The ensuing splits, and confusion between the imaginary and the real and the dissonance between the ordinary and the heroic are the stuff of a gut check on the one hand or a complete escape from self on the other.  |  Negative |  
| This movie is amazing because the fact that the real people portray themselves and their real life experience and do such a good job it's like they're almost living the past over again. Jia Hongsheng plays himself an actor who quit everything except music and drugs struggling with depression and searching for the meaning of life while being angry at everyone especially the people who care for him most.  | Positive  |

同时，我们要下载GloVe文件并解压，在每个解压后的文件开头处添加新的一行，意思是总共读取400000个单词，每个单词用300纬度的词向量表示。

```text
400000 300
```

GloVe文件下载地址：<http://nlp.stanford.edu/data/glove.6B.zip>。

### 确定评价标准

作为典型的分类问题，情感分类的评价标准可以比照普通的分类问题处理。常见的精度（Accuracy）、精准度（Precision）、召回率（Recall）和F_beta分数都可以作为参考。

$精度（Accuracy）= 分类正确的样本数目 / 总样本数目$

$精准度（Precision）= 真阳性样本数目 / 所有预测类别为阳性的样本数目$

$召回率（Recall）= 真阳性样本数目 / 所有真实类别为阳性的样本数目$

$F1分数 = (2 \times Precision \times Recall) / (Precision + Recall)$

在IMDb这个数据集中，正负样本数差别不大，可以简单地用精度（accuracy）作为分类器的衡量标准。

### 确定网络及流程

我们使用基于LSTM构建的SentimentNet网络进行自然语言处理。

1. 加载使用的数据集，并进行必要的数据处理。
2. 使用基于LSTM构建的SentimentNet网络训练数据，生成模型。
    > LSTM（Long short-term memory，长短期记忆）网络是一种时间循环神经网络，适合于处理和预测时间序列中间隔和延迟非常长的重要事件。具体介绍可参考网上资料，在此不再赘述。
3. 得到模型之后，使用验证数据集，查看模型精度情况。

> 本例面向GPU或CPU硬件平台，你可以在这里下载完整的样例代码：<https://gitee.com/mindspore/models/tree/r1.6/official/nlp/lstm>
>
> - `default_config.yaml、config_ascend.yaml`：网络中的一些配置，包括`batch size`、进行几次epoch训练等。
> - `src/dataset.py`：数据集相关，包括转换成MindRecord文件，数据预处理等。
> - `src/imdb.py`： 解析IMDb数据集的工具。
> - `src/lstm.py`： 定义情感网络。
> - `train.py`：模型的训练脚本。
> - `eval.py`：模型的推理脚本。

## 实现阶段

### 导入需要的库文件

下列是我们所需要的公共模块及MindSpore的模块及库文件。

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

### 配置环境信息

1. 使用`parser`模块，传入运行必要的信息，如数据集存放路径，GloVe存放路径，这样的好处是，对于经常变化的配置，可以在运行代码时输入，使用更加灵活。

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

2. 实现代码前，需要配置必要的信息，包括环境信息、执行的模式、后端信息及硬件信息。

    ```python
    context.set_context(
        mode=context.GRAPH_MODE,
        save_graphs=False,
        device_target=config.device_target)
    ```

    详细的接口配置信息，请参见`context.set_context`接口说明。

### 预处理数据集

将数据集格式转换为MindRecord格式，便于MindSpore读取。

```python
if config.preprocess == "true":
    print("============== Starting Data Pre-processing ==============")
    convert_to_mindrecord(config.embed_size, config.aclimdb_path, config.preprocess_path, config.glove_path)
```

> 转换成功后会在`preprocess_path`路径下生成`mindrecord`文件； 通常该操作在数据集不变的情况下，无需每次训练都执行。
> `convert_to_mindrecord`函数的具体实现请参考<https://gitee.com/mindspore/models/blob/r1.6/official/nlp/lstm/src/dataset.py>
> 其中包含两大步骤：
>
> 1. 解析文本数据集，包括编码、分词、对齐、处理GloVe原始数据，使之能够适应网络结构。
> 2. 转换并保存为MindRecord格式数据集。

### 定义网络

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

> `SentimentNet`网络结构的具体实现请参考<https://gitee.com/mindspore/models/blob/r1.6/official/nlp/lstm/src/lstm.py>

### 预训练模型

通过参数`pre_trained`指定预加载CheckPoint文件来进行预训练，默认该参数为空。

```python
if config.pre_trained:
    load_param_into_net(network, load_checkpoint(config.pre_trained))
```

### 定义优化器及损失函数

定义优化器及损失函数的示例代码如下：

```python
loss = nn.SoftmaxCrossEntropyWithLogits(sparse=True, reduction='mean')
opt = nn.Momentum(network.trainable_params(), config.learning_rate, config.momentum)
loss_cb = LossMonitor()
```

### 训练并保存模型

加载对应数据集并配置好CheckPoint生成信息，然后使用`model.train`接口，进行模型训练。

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

> `lstm_create_dataset`函数的具体实现请参考<https://gitee.com/mindspore/models/blob/r1.6/official/nlp/lstm/src/dataset.py>

### 模型验证

加载验证数据集及保存的CheckPoint文件，进行验证，查看模型质量。

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

## 实验结果

在经历了20轮epoch之后，在测试集上的精度约为84.19%。

### 执行训练

1. 运行训练代码，查看运行结果。

    ```bash
    python train.py --config_path=CONFIG_FILE --device_target="Ascend" --aclimdb_path=$ACLIMDB_DIR --glove_path=$GLOVE_DIR --preprocess=true --preprocess_path=./preprocess > log.txt 2>&1 &
    ```

    输出如下，可以看到loss值随着训练逐步降低，最后达到0.2855左右：

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

2. 查看保存的CheckPoint文件。

   训练过程中保存了CheckPoint文件，即模型文件，我们可以查看文件保存的路径下的所有保存文件。

    ```bash
    ls ./ckpt_lstm/*.ckpt
    ```

    输出如下：

    ```text
    lstm-11_390.ckpt  lstm-12_390.ckpt  lstm-13_390.ckpt  lstm-14_390.ckpt  lstm-15_390.ckpt  lstm-16_390.ckpt  lstm-17_390.ckpt  lstm-18_390.ckpt  lstm-19_390.ckpt  lstm-20_390.ckpt
    ```

### 验证模型

使用最后保存的CheckPoint文件，加载验证数据集，进行验证。

```bash
python eval.py --config_path=$CONFIG_FILE --device_target="Ascend" --preprocess=false --preprocess_path=$PREPROCESS_DIR --ckpt_file=$CKPT_FILE > log.txt 2>&1 &
```

参数解释：

- `--config_path`：参数文件的路径，即源代码文件中的`default_config.yaml`文件。
- `--device_target`：模型训练使用的硬件。本文选用的是`Ascend`。可选`CPU`、`GPU`和`Ascend`。
- `--preprocess`：是否对数据进行预处理。选择`false`。
- `--preprocess_path`：预处理后的数据集路径。
- `--ckpt_file`：载入模型权重文件的路径。（可选用`./ckpt_lstm/lstm-20_390.ckpt`）。

输出如下，可以看到使用验证的数据集，对文本的情感分析正确率在84.19%左右，达到一个基本满意的结果。

```text
============== Starting Testing ==============
============== {'acc': 0.8419471153846154} ==============
```
