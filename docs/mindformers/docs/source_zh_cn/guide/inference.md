# 推理

[![查看源文件](https://mindspore-website.obs.cn-north-4.myhuaweicloud.com/website-images/master/resource/_static/logo_source.svg)](https://gitee.com/mindspore/docs/blob/master/docs/mindformers/docs/source_zh_cn/guide/inference.md)

## 概述

MindSpore Transformers 提供了大模型推理能力，用户可以执行 `run_mindformer` 统一脚本进行推理。用户使用 `run_mindformer` 统一脚本可以不编写代码，直接通过配置文件启动，用法便捷。

## 基本流程

推理流程可以分解成以下几个步骤：

### 1. 选择推理的模型

根据需要的推理任务，选择不同的模型，如文本生成可以选择 `Qwen2.5-7B` 等。

### 2. 准备模型权重

目前推理权重可以在线加载完整权重进行推理，权重可以通过以下两种方式获得：

1. 从Hugging Face模型库中下载相应模型的开源的完整权重。
2. 预训练或者微调后的分布式权重，通过[合并](https://www.mindspore.cn/mindformers/docs/zh-CN/dev/feature/safetensors.html#%E6%9D%83%E9%87%8D%E5%90%88%E5%B9%B6)生成一个完整权重。

### 3. 执行推理任务

使用 `run_mindformer` 统一脚本执行推理任务。

## 使用 run_mindformer 一键启动脚本推理

单卡推理可以直接执行[run_mindformer.py](https://gitee.com/mindspore/mindformers/blob/dev/run_mindformer.py)脚本，多卡推理需要借助[scripts/msrun_launcher.sh](https://gitee.com/mindspore/mindformers/blob/dev/scripts/msrun_launcher.sh)来启动。

run_mindformer.py的参数说明如下：

| 参数                     | 参数说明                                         |
| :----------------------- |:---------------------------------------------|
| config                   | yaml配置文件的路径                                  |
| run_mode                 | 运行的模式，推理设置为predict                           |
| use_parallel             | 是否使用多卡推理                                     |
| load_checkpoint          | 加载的权重路径                                      |
| predict_data             | 推理的输入数据，多batch推理时需要传入输入数据的txt文件路径，包含多行输入     |
| auto_trans_ckpt          | 自动权重切分，默认值为False                             |
| src_strategy_path_or_dir | 权重的策略文件路径                                    |
| predict_batch_size       | 多batch推理的batch_size大小                        |
| modal_type               | 多模态推理场景下，模型推理输入对应模态，图片路径对应'image'，文本对应'text' |

msrun_launcher.sh包括run_mindformer.py命令和推理卡数两个参数。

下面将以 `Qwen2.5-7B` 为例介绍单卡和多卡推理的用法，推荐配置为[predict_qwen2_5_7b_instruct.yaml](https://gitee.com/mindspore/mindformers/blob/dev/research/qwen2_5/predict_qwen2_5_7b_instruct.yaml)文件。

### 配置修改

权重相关配置修改如下：

```yaml
load_checkpoint: "path/to/Qwen2_5_7b_instruct/"
load_ckpt_format: 'safetensors'
auto_trans_ckpt: True
```

默认配置是单卡推理配置，并行相关配置修改如下：

```yaml
use_parallel: False
parallel_config:
  data_parallel: 1
  model_parallel: 1
  pipeline_stage: 1
```

`tokenizer`相关配置修改如下：

```yaml
processor:
  tokenizer:
    vocab_file: "path/to/vocab.json"
    merges_file: "path/to/merges.txt"
```

具体配置说明均可参考[yaml配置说明](https://www.mindspore.cn/mindformers/docs/zh-CN/dev/feature/configuration.html)。

### 单卡推理

当使用完整权重推理时，推荐使用默认配置，执行以下命令即可启动推理任务：

```shell
python run_mindformer.py \
--register_path /path/to/research/qwen2_5/ \
--config /path/to/research/qwen2_5/predict_qwen2_5_7b_instruct \
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

1. 模型并行model_parallel的配置和使用的卡数需保持一致，下文用例为4卡推理，需将model_parallel设置成4；
2. 当前版本的多卡推理不支持数据并行，需将data_parallel设置为1。

当使用完整权重推理时，需要开启在线切分方式加载权重，参考以下命令：

```shell
bash scripts/msrun_launcher.sh "run_mindformer.py \
 --register_path /path/to/research/qwen2_5 \
 --config /path/to/research/qwen2_5/qwen2_5_72b/predict_qwen2_5_72b_instruct.yaml \
 --run_mode predict \
 --use_parallel True \
 --auto_trans_ckpt True \
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
 --register_path /path/to/research/qwen2_5 \
 --config /path/to/research/qwen2_5/qwen2_5_72b/predict_qwen2_5_72b_instruct.yaml \
 --run_mode predict \
 --predict_batch_size 4 \
 --use_parallel True \
 --auto_trans_ckpt True \
 --predict_data '帮助我制定一份去上海的旅游攻略'" 4
```

推理结果查看方式，与多卡推理相同。

## 更多信息

更多关于不同模型的推理示例，请访问[MindSpore Transformers 已支持模型库](https://www.mindspore.cn/mindformers/docs/zh-CN/dev/introduction/models.html)。