# 推理指南

[![查看源文件](https://mindspore-website.obs.cn-north-4.myhuaweicloud.com/website-images/master/resource/_static/logo_source.svg)](https://gitee.com/mindspore/docs/blob/master/docs/mindformers/docs/source_zh_cn/guide/inference.md)

## 概述

MindSpore Transformers 提供了大模型推理能力，用户可以执行 `run_mindformer` 统一脚本进行推理。用户使用 `run_mindformer` 统一脚本可以不编写代码，直接通过配置文件启动，用法便捷。

## 基本流程

推理流程可以分解成以下几个步骤：

### 1. 选择推理的模型

根据需要的推理任务，选择不同的模型，如文本生成可以选择Qwen3等。

### 2. 准备模型文件

获取Hugging Face模型文件：权重、配置与分词器，将下载的文件存放在同一个文件夹目录，方便后续使用。

### 3. 准备YAML配置文件

用户需要配置一份YAML文件，来定义任务的所有配置。MindSpore Transformers提供了一份YAML配置模板，用户可以基于模板，根据实际场景自定义配置。详细可见[推理配置模板使用指南](https://www.mindspore.cn/mindformers/docs/zh-CN/master/advanced_development/yaml_config_inference.html)。

### 4. 执行推理任务

使用 `run_mindformer` 统一脚本执行推理任务。

## 使用 run_mindformer 一键启动脚本推理

单卡推理可以直接执行[run_mindformer.py](https://gitee.com/mindspore/mindformers/blob/master/run_mindformer.py)脚本，多卡推理需要借助[scripts/msrun_launcher.sh](https://gitee.com/mindspore/mindformers/blob/master/scripts/msrun_launcher.sh)来启动。

run_mindformer.py的参数说明如下：

| 参数                     | 参数说明                                         |
| :----------------------- |:---------------------------------------------|
| config                   | yaml配置文件的路径                                  |
| run_mode                 | 运行的模式，推理设置为predict                           |
| use_parallel             | 是否使用多卡推理                                     |
| predict_data             | 推理的输入数据，多batch推理时需要传入输入数据的txt文件路径，包含多行输入     |
| predict_batch_size       | 多batch推理的batch_size大小                        |

msrun_launcher.sh包括run_mindformer.py命令和推理卡数两个参数。

下面将以Qwen3-8B为例介绍单卡和多卡推理的用法，推荐配置为[predict_qwen3.yaml](https://gitee.com/mindspore/mindformers/blob/master/configs/qwen3/predict_qwen3.yaml)文件。

### 配置修改

当前推理可以直接复用Hugging Face的配置文件和tokenizer，并且在线加载Hugging Face的safetensors格式的权重。使用时的配置修改如下：

```yaml
use_legacy: False
pretrained_model_dir: '/path/hf_dir'
```

参数说明：

- use_legacy：决定是否使用老架构，默认值：`True`；
- pretrained_model_dir：Hugging Face模型目录路径，放置模型配置、Tokenizer等文件。`/path/hf_dir`中的内容如下：

```text
📂Qwen3-8B
├── 📄config.json
├── 📄generation_config.json
├── 📄merges.txt
├── 📄model-xxx.safetensors
├── 📄model-xxx.safetensors
├── 📄model.safetensors.index.json
├── 📄tokenizer.json
├── 📄tokenizer_config.json
└── 📄vocab.json
```

默认配置是单卡推理配置，相关配置如下：

```yaml
use_parallel: False
parallel_config:
  data_parallel: 1
  model_parallel: 1
```

如果需要执行多卡推理任务，相关配置的修改如下：

```yaml
use_parallel: True
parallel_config:
  data_parallel: 1
  model_parallel: 2 # 修改为实际使用的卡数
```

具体配置说明均可参考[yaml配置说明](https://www.mindspore.cn/mindformers/docs/zh-CN/master/feature/configuration.html)。

### 单卡推理

按照[配置修改](#配置修改)章节修改完成后，执行以下命令即可启动单卡推理任务：

```shell
python run_mindformer.py \
--config configs/qwen3/predict_qwen3.yaml \
--run_mode predict \
--use_parallel False \
--predict_data '帮助我制定一份去上海的旅游攻略'
```

出现如下结果，证明推理成功。推理结果也会保存到当前目录下的 `text_generation_result.txt` 文件中。

```text
'text_generation_text': [帮助我制定一份去上海的旅游攻略，包括景点、美食、住宿等信息...]
```

### 多卡推理

多卡推理的配置要求与单卡存在差异，需参考下面修改配置：

1. 模型并行model_parallel的配置和使用的卡数需保持一致。下文用例为4卡推理，需将model_parallel设置成4；
2. 当前版本的多卡推理不支持数据并行，需将data_parallel设置为1。

按照[配置修改](#配置修改)章节修改完成后，执行以下命令即可启动多卡推理任务：

```shell
bash scripts/msrun_launcher.sh "run_mindformer.py \
 --config configs/qwen3/predict_qwen3.yaml \
 --run_mode predict \
 --use_parallel True \
 --predict_data '帮助我制定一份去上海的旅游攻略'" 4
```

出现如下结果，证明推理成功。推理结果也会保存到当前目录下的 text_generation_result.txt 文件中。详细日志可通过`./output/msrun_log`目录查看。

```text
'text_generation_text': [帮助我制定一份去上海的旅游攻略，包括景点、美食、住宿等信息...]
```

### 多卡多batch推理

多卡多batch推理的启动方式可参考上述[多卡推理](#多卡推理)，但是需要增加`predict_batch_size`的入参，并修改`predict_data`的入参。

`input_predict_data.txt`文件的内容和格式是每一行都是一个输入，问题的个数与`predict_batch_size`一致，可以参考以下格式：

```text
帮助我制定一份去上海的旅游攻略
帮助我制定一份去上海的旅游攻略
帮助我制定一份去上海的旅游攻略
帮助我制定一份去上海的旅游攻略
```

以完整权重推理为例，可以参考以下命令启动推理任务：

```shell
bash scripts/msrun_launcher.sh "run_mindformer.py \
 --config configs/qwen3/predict_qwen3.yaml \
 --run_mode predict \
 --predict_batch_size 4 \
 --use_parallel True \
 --predict_data path/to/input_predict_data.txt" 4
```

推理结果查看方式，与多卡推理相同。

## 更多信息

更多关于不同模型的推理示例，请访问[MindSpore Transformers 已支持模型库](https://www.mindspore.cn/mindformers/docs/zh-CN/master/introduction/models.html)。
