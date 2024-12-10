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
| `--model`       | str | The value must be **mf**, indicating the MindFormers evaluation policy.| Yes   |
| `--model_args`  | str | Model and evaluation parameters. For details, see "MindFormers model parameters."      | Yes   |
| `--tasks`       | str | Dataset name. Multiple datasets can be specified and separated by commas (,).      | Yes   |
| `--batch_size`  | int | Number of batch processing samples.                   | No   |
| `--num_fewshot` | int | Number of few-shot samples.             | No   |
| `--limit`       | int | Number of samples for each task. This parameter is mainly used for function tests.         | No   |

MindFormers model parameters

| Parameter          | Type  | Description                             | Required|
|--------------|------|-----------------------------------|------|
| `pretrained`   | str  | Model directory.                           | Yes   |
| `use_past`     | bool | Specifies whether to enable incremental inference. This parameter must be enabled for evaluation tasks of the generate_until type.| No   |
| `device_id`    | int  | Device ID.                             | No   |
| `use_parallel` | bool | Specifies whether to enable the parallel policy.                           | No   |
| `dp`           | int  | Data parallelism.                             | No   |
| `tp`           | int  | Model parallelism.                             | No   |

#### Preparations Before Evaluation

1. Create a model directory MODEL_DIR.
2. Store the MindFormers weight(\*.ckpt), YAML file(\*.yaml), and tokenizer file(\*_tokenizer.model) in the model directory. For details, Please refer to the README documentation of each MindFormers model for the method of obtaining, which is usually located in [model_cards](https://gitee.com/mindspore/mindformers/tree/dev/model_cards) directory or in [research](https://gitee.com/mindspore/mindformers/tree/dev/research) directory, depending on the model used by the user;
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

## VLMEvalKit Evaluation

### Overview

[VLMEvalKit](https://github.com/open-compass/VLMEvalKit)
is an open source toolkit designed for large visual language model evaluation, supporting one-click evaluation of large visual language models on various benchmarks, without the need for complicated data preparation, making the evaluation process easier. It supports a variety of graphic multimodal evaluation sets and video multimodal evaluation sets, a variety of API models and open source models based on PyTorch and HF, and customized prompts and evaluation metrics. After adapting MindFormers based on VLMEvalKit evaluation framework, it supports loading multimodal large models in MindFormers for evaluation.

### Supported Feature Descriptions

1. Supports automatic download of evaluation datasets;
2. Support for user-defined input of multiple datasets and models (currently only `cogvlm2-llama3-chat-19B` is supported and will be added gradually in subsequent releases);
3. Generate results with one click.

### Installation

```shell
git clone https://github.com/open-compass/VLMEvalKit.git
cd VLMEvalKit
pip install -e .
```

### Usage

Run the script [eval_with_vlmevalkit.py](https://gitee.com/mindspore/mindformers/blob/dev/toolkit/benchmarks/eval_with_vlmevalkit.py).

#### Launching a Single-Card Evaluation Script

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

#### Evaluation Parameters

VLMEvalKit main parameters

| Parameters            | Type  | Descriptions                                                            | Compulsory(Y/N)|
|---------------|-----|-----------------------------------------------------------------|------|
| --data      | str | Name of the dataset, multiple datasets can be passed in, split by spaces.                                            | Y    |
| --model  | str | Name of the model, multiple models can be passed in, split by spaces.                                              | Y    |
| --verbose       | /   | Outputs logs from the evaluation run.                                                   | N    |
| --work-dir  | str | The directory where the evaluation results are stored, by default, is stored in the folder with the same name as the model in the current directory.                                | N    |
| --model-path | str | Contains the paths of all relevant files of the model (weights, tokenizer files, configuration files, processor files), multiple paths can be passed in, filled in according to the order of the model, split by spaces. | Y    |
| --config-path       | str | Model configuration file path, multiple paths can be passed in, fill in according to the model order, split by space.                                 | Y   |

#### Preparation Before Evaluation

1. Create model directory model_path;
2. Model directory must be placed MindFormers weights, yaml configuration file, tokenizer file, which can refer to the MindFormers model README document;
3. Configure the yaml configuration file.

The yaml configuration reference:

```yaml
load_checkpoint: "/{path}/model.ckpt"  # Specify the path to the weights file
model:
  model_config:
    use_past: True                         # Turn on incremental inference
    is_dynamic: False                       # Turn off dynamic shape

  tokenizer:
    vocab_file: "/{path}/tokenizer.model"  # Specify the tokenizer file path
```

### Evaluation Sample

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

The results of the evaluation are as follows, where `Bleu` and `ROUGE_L` denote the metrics for evaluating the quality of the translation, and `CIDEr` denotes the metrics for evaluating the image description task.

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
