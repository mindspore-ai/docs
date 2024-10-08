# Evaluation

[![View Source On Gitee](https://mindspore-website.obs.cn-north-4.myhuaweicloud.com/website-images/master/resource/_static/logo_source_en.svg)](https://gitee.com/mindspore/docs/blob/master/docs/mindformers/docs/source_en/usage/evaluation.md)

## Harness Evaluation

### Introduction

[LM Evaluation Harness](https://github.com/EleutherAI/lm-evaluation-harness) is an open-source language model evaluation framework that provides evaluation of more than 60 standard academic datasets, supports multiple evaluation modes such as HuggingFace model evaluation, PEFT adapter evaluation, and vLLM inference evaluation, and supports customized prompts and evaluation metrics, including the evaluation tasks of the loglikelihood, generate_until, and loglikelihood_rolling types.
After MindFormers is adapted based on the Harness evaluation framework, the MindFormers model can be loaded for evaluation.

### Installation

```shell
pip install lm_eval==0.4.3
```

### Usage

Run the [eval_with_harness.py](https://gitee.com/mindspore/mindformers/blob/dev/toolkit/benchmarks/eval_with_harness.py) script.

#### Viewing a Dataset Evaluation Task

```shell
#!/bin/bash

python toolkit/benchmarks/eval_with_harness.py --tasks list
```

#### Starting the Single-Device Evaluation Script

```shell
#!/bin/bash

python toolkit/benchmarks/eval_with_harness.py --model mf --model_args "pretrained=MODEL_DIR,device_id=0" --tasks TASKS
```

#### Starting the Multi-Device Parallel Evaluation Script

```shell
#!/bin/bash

export ASCEND_RT_VISIBLE_DEVICES=0,1,2,3

bash  mindformers/scripts/msrun_launcher.sh "toolkit/benchmarks/eval_with_harness.py \
    --model mf \
    --model_args pretrained=MODEL_DIR,use_parallel=True,tp=1,dp=4 \
    --tasks TASKS \
    --batch_size 4" 4
```

You can set multiple device numbers through the environment variable ASCEND_RT_VISIBLE_DEVICES.

#### Evaluation Parameters

Harness parameters

| Parameter           | Type | Description                     | Required|
|---------------|-----|---------------------------|------|
| --model       | str | The value must be **mf**, indicating the MindFormers evaluation policy.| Yes   |
| --model_args  | str | Model and evaluation parameters. For details, see "MindFormers model parameters."      | Yes   |
| --tasks       | str | Dataset name. Multiple datasets can be specified and separated by commas (,).      | Yes   |
| --batch_size  | int | Number of batch processing samples.                   | No   |
| --num_fewshot | int | Number of few-shot samples.             | No   |
| --limit       | int | Number of samples for each task. This parameter is mainly used for function tests.         | No   |

MindFormers model parameters

| Parameter          | Type  | Description                             | Required|
|--------------|------|-----------------------------------|------|
| pretrained   | str  | Model directory.                           | Yes   |
| use_past     | bool | Specifies whether to enable incremental inference. This parameter must be enabled for evaluation tasks of the generate_until type.| No   |
| device_id    | int  | Device ID.                             | No   |
| use_parallel | bool | Specifies whether to enable the parallel policy.                           | No   |
| dp           | int  | Data parallelism.                             | No   |
| tp           | int  | Model parallelism.                             | No   |

#### Preparations Before Evaluation

1. Create a model directory MODEL_DIR.
2. Store the MindFormers weight, YAML file, and tokenizer file in the model directory. For details about how to obtain the weight and files, see the README file of the MindFormers model.
3. Configure the yaml file.

YAML configuration references:

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

### Evaluation Example

```shell
#!/bin/bash

python toolkit/benchmarks/eval_with_harness.py --model mf --model_args "pretrained=./llama3-8b,use_past=True" --tasks gsm8k

```

The evaluation result is as follows. Filter indicates the output mode of the matching model, Metric indicates the evaluation metric, Value indicates the evaluation score, and Stderr indicates the score error.

| Tasks | Version | Filter           | n-shot | Metric      |   | Value  |   | Stderr |
|-------|--------:|------------------|-------:|-------------|---|--------|---|--------|
| gsm8k |       3 | flexible-extract |      5 | exact_match | ↑ | 0.5034 | ± | 0.0138 |
|       |         | strict-match     |      5 | exact_match | ↑ | 0.5011 | ± | 0.0138 |

### Features

For details about all Harness evaluation tasks, see [Viewing a Dataset Evaluation Task](#viewing-a-dataset-evaluation-task).