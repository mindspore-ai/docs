# 使用Mindtransformer中的BERT微调

<a href="https://gitee.com/mindspore/docs/blob/master/docs/mindtransformer/docs/source_zh_cn/mindtransformer_bert_finetune.md" target="_blank"><img src="https://mindspore-website.obs.cn-north-4.myhuaweicloud.com/website-images/master/resource/_static/logo_source.png"></a>

## 背景

通常用户在完成模型的预训练之后，需要在下游任务数据集上进行微调，从而满足更加细化的任务的要求。库中提供了完整的下有任务微调模板，本文演示了如何使用库中的BERT模型进行下游任务微调的过程。

## 数据集

- 生成下游任务数据集
    - 下载数据集进行微调和评估，如中文实体识别任务[CLUENER](https://github.com/CLUEbenchmark/CLUENER2020)、中文文本分类任务[TNEWS](https://github.com/CLUEbenchmark/CLUE)、中文实体识别任务[ChineseNER](https://github.com/zjy-ucas/ChineseNER)、英文问答任务[SQuAD v1.1训练集](https://rajpurkar.github.io/SQuAD-explorer/dataset/train-v1.1.json)、[SQuAD v1.1验证集](https://rajpurkar.github.io/SQuAD-explorer/dataset/dev-v1.1.json)、英文分类任务集合[GLUE](https://gluebenchmark.com/tasks)等。
    - 将数据集文件从JSON格式转换为TFRecord格式。详见[BERT](https://github.com/google-research/bert)代码仓中的[run_classifier.py](https://github.com/google-research/bert/blob/master/run_classifier.py)或[run_squad.py](https://github.com/google-research/bert/blob/master/run_squad.py)文件。

## 预训练模型

除了使用预训练方法训练得到的权重，我们支持从第三方库(如[huggingface](https://huggingface.co/))的权重进行转换。以huggingface中的bert-large为例，我们提供了权重转换的脚本，可以一键，将PyTorch的权重格式，转换为MindSpore支持的权重格式。代码如下：

```bash
python convert_bert_weight.py --layers 24 --torch_path pytorch_model.bin --mindspore_path ./converted_mindspore_bert.ckpt
```

其中，`pytorch_model.bin`为下载得到的预训练权重。各个参数意义如下：

- layers                      需要转换的模型的层数。
- torch_path                  需要转换的Pytorch权重。
- mindspore_path              转换完成的MindSpore的权重地址。

获取得到模型的权重后，用户可以使用其进行下游任务的微调。

## 环境要求

- 硬件（Ascend处理器）
    - 准备Ascend或GPU处理器搭建硬件环境。
- 框架
    - [MindSpore](https://gitee.com/mindspore/mindspore)
- 更多关于Mindspore的信息，请查看以下资源：
    - [MindSpore教程](https://www.mindspore.cn/tutorials/zh-CN/master/index.html)
    - [MindSpore Python API](https://www.mindspore.cn/docs/zh-CN/master/index.html)

## 运行流程

### 用法

#### 微调GLUE数据集RTE任务

[GLUE](https://gluebenchmark.com/)(General Language Understanding Evaluation)是一个包含9种自然语言理解任务的标准数据集，包含MNLI、QNLI、RTE、WNLI、CoLA等任务，均为英文。下面展示了使用库中运行脚本微调BERT模型，得到RTE任务的结果。

RTE(The Recognizing Textual Entailment datasets，识别文本蕴含数据集)。该任务来源于新闻及维基百科，并且被分为两类：蕴含(entailment)和非蕴含(not entailment)。最终评价指标为准确率(accuracy)。库上提供了两种不同模式的运行命令，如下：

- standalone模式：

```bash
bash ./examples/finetune/run_classifier.sh 0 RTE
```

以上命令后台运行，可以在当前目录的RTE.txt中查看训练日志。

- distributed模式：

```bash
bash ./examples/finetune/run_classifier.sh 8 /path/hostfile RTE
```

其中，/path/hostfile为用户指定的文件路径，可以按照如下格式设置hostfile：

```text
192.168.0.1 slots=8
```

表示在ip为192.168.0.1的机器上有8张卡。以上命令后台运行，可以在当前目录的RTE.txt中查看训练日志。

运行完成后，会在日志中打印下游任务的结果，以最终评价指标为准确率(accuracy)为例：

```text
acc_num XXX, total_num XXX, accuracy XXX
```

使用bert-large模型，RTE任务上的最终准确率为70.2。
