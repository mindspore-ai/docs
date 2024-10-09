# 评测

[![查看源文件](https://mindspore-website.obs.cn-north-4.myhuaweicloud.com/website-images/master/resource/_static/logo_source.svg)](https://gitee.com/mindspore/docs/blob/master/docs/mindformers/docs/source_zh_cn/usage/evaluation.md)

## Harness评测

### 基本介绍

[LM Evaluation Harness](https://github.com/EleutherAI/lm-evaluation-harness)是一个开源语言模型评测框架，提供60多种标准学术数据集的评测，支持HuggingFace模型评测、PEFT适配器评测、vLLM推理评测等多种评测方式，支持自定义prompt和评测指标，包含loglikelihood、generate_until、loglikelihood_rolling三种类型的评测任务。基于Harness评测框架对MindFormers进行适配后，支持加载MindFormers模型进行评测。

### 安装

```shell
pip install lm_eval==0.4.3
```

### 使用方式

执行脚本[eval_with_harness.py](https://gitee.com/mindspore/mindformers/blob/dev/toolkit/benchmarks/eval_with_harness.py)

#### 查看数据集评测任务

```shell
#!/bin/bash

python toolkit/benchmarks/eval_with_harness.py --tasks list
```

#### 启动单卡评测脚本

```shell
#!/bin/bash

python toolkit/benchmarks/eval_with_harness.py --model mf --model_args "pretrained=MODEL_DIR,device_id=0" --tasks TASKS
```

#### 启动多卡并行评测脚本

```shell
#!/bin/bash

export ASCEND_RT_VISIBLE_DEVICES=0,1,2,3

bash  mindformers/scripts/msrun_launcher.sh "toolkit/benchmarks/eval_with_harness.py \
    --model mf \
    --model_args pretrained=MODEL_DIR,use_parallel=True,tp=1,dp=4 \
    --tasks TASKS \
    --batch_size 4" 4
```

可通过环境变量ASCEND_RT_VISIBLE_DEVICES设置多卡卡号。

#### 评测参数

Harness主要参数

| 参数            | 类型  | 参数介绍                      | 是否必须 |
|---------------|-----|---------------------------|------|
| --model       | str | 须设置为mf，对应为MindFormers评估策略 | 是    |
| --model_args  | str | 模型及评估相关参数，见下方模型参数介绍       | 是    |
| --tasks       | str | 数据集名称，可传入多个数据集，逗号分割       | 是    |
| --batch_size  | int | 批处理样本数                    | 否    |
| --num_fewshot | int | Few_shot的样本数              | 否    |
| --limit       | int | 每个任务的样本数，多用于功能测试          | 否    |

MindFormers模型参数

| 参数           | 类型   | 参数介绍                              | 是否必须 |
|--------------|------|-----------------------------------|------|
| pretrained   | str  | 模型目录路径                            | 是    |
| use_past     | bool | 是否开启增量推理，generate_until类型的评测任务须开启 | 否    |
| device_id    | int  | 设备id                              | 否    |
| use_parallel | bool | 开启并行策略                            | 否    |
| dp           | int  | 数据并行                              | 否    |
| tp           | int  | 模型并行                              | 否    |

#### 评测前准备

1. 创建模型目录MODEL_DIR；
2. 模型目录下须放置MindFormers权重、yaml配置文件、分词器文件，获取方式参考MindFormers模型README文档；
3. 配置yaml配置文件。

yaml配置参考：

```yaml
run_mode: 'predict'
model:
  model_config:
    use_past: True
    checkpoint_name_or_path: "model.ckpt"
processor:
  tokenizer:
    vocab_file: "tokenizer.model"
```

### 评测样例

```shell
#!/bin/bash

python toolkit/benchmarks/eval_with_harness.py --model mf --model_args "pretrained=./llama3-8b,use_past=True" --tasks gsm8k

```

评测结果如下，其中Filter对应匹配模型输出结果的方式，Metric对应评测指标，Value对应评测分数，stderr对应分数误差。

| Tasks | Version | Filter           | n-shot | Metric      |   | Value  |   | Stderr |
|-------|--------:|------------------|-------:|-------------|---|--------|---|--------|
| gsm8k |       3 | flexible-extract |      5 | exact_match | ↑ | 0.5034 | ± | 0.0138 |
|       |         | strict-match     |      5 | exact_match | ↑ | 0.5011 | ± | 0.0138 |

### 支持特性说明

支持Harness全量评测任务，见[查看数据集评测任务](#查看数据集评测任务)。

## VLMEvalKit评测

### 基本介绍

[VLMEvalKit](https://github.com/open-compass/VLMEvalKit)
是一款专为大型视觉语言模型评测而设计的开源工具包，支持在各种基准测试上对大型视觉语言模型进行一键评估，无需进行繁重的数据准备工作，让评估过程更加简便。 它支持多种图文多模态评测集和视频多模态评测集，支持多种API模型以及基于PyTorch和HF的开源模型，支持自定义prompt和评测指标。基于VLMEvalKit评测框架对MindFormers进行适配后，支持加载MindFormers中多模态大模型进行评测。

### 支持特性说明

1. 支持自动下载评测数据集；
2. 支持用户自定义输入多种数据集和模型（目前仅支持`cogvlm2-llama3-chat-19B`，后续版本会逐渐增加）；
3. 一键生成评测结果。

### 安装

```shell
git clone https://github.com/open-compass/VLMEvalKit.git
cd VLMEvalKit
pip install -e .
```

### 使用方式

执行脚本[eval_with_vlmevalkit.py](https://gitee.com/mindspore/mindformers/blob/dev/toolkit/benchmarks/eval_with_vlmevalkit.py)。

#### 启动单卡评测脚本

```shell
#!/bin/bash

python eval_with_vlmevalkit.py \
  --data MME \
  --model cogvlm2-llama3-chat-19B \
  --verbose \
  --work-dir /{path}/evaluate_result \
  --model-path /{path}/cogvlm2_model_path \
  --config-path /{path}/cogvlm2_config_path
```

#### 评测参数

VLMEvalKit主要参数

| 参数            | 类型  | 参数介绍                                                            | 是否必须 |
|---------------|-----|-----------------------------------------------------------------|------|
| --data      | str | 数据集名称，可传入多个数据集，空格分割。                                            | 是    |
| --model  | str | 模型名称，可传入多个模型，空格分割。                                              | 是    |
| --verbose       | /   | 输出评测运行过程中的日志。                                                   | 否    |
| --work-dir  | str | 存放评测结果的目录，默认存储在当前目录与模型名称相同的文件夹下。                                | 否    |
| --model-path | str | 包含模型所有相关文件(权重、分词器文件、配置文件、processor文件)的路径，可传入多个路径，按照模型顺序填写，空格分割。 | 是    |
| --config-path       | str | 模型配置文件路径，可传入多个路径，按照模型顺序填写，空格分割。                                 | 是    |

#### 评测前准备

1. 创建模型目录model_path；
2. 模型目录下须放置MindFormers权重、yaml配置文件、分词器文件，获取方式参考MindFormers模型README文档；
3. 配置yaml配置文件。

yaml配置参考：

```yaml
load_checkpoint: "/{path}/model.ckpt"  # 指定权重文件路径
model:
  model_config:
    use_past: True                         # 开启增量推理
    is_dynamic: False                       # 关闭动态shape

  tokenizer:
    vocab_file: "/{path}/tokenizer.model"  # 指定tokenizer文件路径
```

### 评测样例

```shell
#!/bin/bash

export USE_ROPE_SELF_DEFINE=True
python eval_with_vlmevalkit.py \
  --data COCO_VAL \
  --model cogvlm2-llama3-chat-19B \
  --verbose \
  --work-dir /{path}/evaluate_result \
  --model-path /{path}/cogvlm2_model_path \
  --config-path /{path}/cogvlm2_config_path
```

评测结果如下，其中`Bleu`和`ROUGE_L`表示评估翻译质量的指标，`CIDEr`表示评估图像描述任务的指标。

```json
{
   "Bleu": [
      15.523950970070652,
      8.971141548228058,
      4.702477458554666,
      2.486860744700995
   ],
   "ROUGE_L": 15.575063213115946,
   "CIDEr": 0.01734615519604295
}
```
