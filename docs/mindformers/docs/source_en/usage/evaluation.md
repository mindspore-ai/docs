# Evaluation

[![View Source On Gitee](https://mindspore-website.obs.cn-north-4.myhuaweicloud.com/website-images/master/resource/_static/logo_source_en.svg)](https://gitee.com/mindspore/docs/blob/master/docs/mindformers/docs/source_en/usage/evaluation.md)

## Harness Evaluation

### Introduction

[LM Evaluation Harness](https://github.com/EleutherAI/lm-evaluation-harness) is an open-source language model evaluation framework that provides evaluation of more than 60 standard academic datasets, supports multiple evaluation modes such as HuggingFace model evaluation, PEFT adapter evaluation, and vLLM inference evaluation, and supports customized prompts and evaluation metrics, including the evaluation tasks of the loglikelihood, generate_until, and loglikelihood_rolling types.
After MindFormers is adapted based on the Harness evaluation framework, the MindFormers model can be loaded for evaluation.

The currently adapted models and supported evaluation tasks are shown in the table below (the remaining models and evaluation tasks are actively being adapted, please pay attention to version updates):

| Adapted models | Supported evaluation tasks |
|----------------|----------------------------|
| Llama3-8B      | Gsm8k、Boolq、Mmlu、Ceval     |
| Qwen2-7B       | Gsm8k、Boolq、Mmlu、Ceval     |

### Installation

Harness supports two installation methods: pip installation and source code compilation installation. Pip installation is simpler and faster, source code compilation and installation are easier to debug and analyze, and users can choose the appropriate installation method according to their needs.

#### pip Installation

Users can execute the following command to install Harness:

```shell
pip install lm_eval==0.4.4
```

#### Source Code Compilation Installation

Users can execute the following command to compile and install Harness:

```bash
git clone --depth 1 https://github.com/EleutherAI/lm-evaluation-harness
cd lm-evaluation-harness
git checkout v0.4.4
pip install -e .
```

### Usage

#### Viewing a Dataset Evaluation Task

Users can view all the evaluation tasks supported by Harness through the following command:

```shell
#!/bin/bash

python toolkit/benchmarks/eval_with_harness.py --tasks list
```

#### Starting the Single-Device Evaluation Script

- Preparations Before Evaluation

  1. Create a model directory MODEL_DIR.
  2. Store the YAML file(\*.yaml), and tokenizer file(\*_tokenizer.py) in the model directory. For details, Please refer to the description documents of each model in the [model library](../start/models.md);
  3. Configure the yaml file. Refer to [configuration description](../appendix/conf_files.md).

      YAML configuration example:

      ```yaml
      run_mode: 'predict'    # Set inference mode
      model:
        model_config:
          use_past: True
          checkpoint_name_or_path: "model.ckpt"    # path of ckpt
      processor:
        tokenizer:
          vocab_file: "tokenizer.model"    # path of tokenizer
      ```

- Executing the Following Evaluation Command

   ```shell
   #!/bin/bash

   python toolkit/benchmarks/eval_with_harness.py --model mf --model_args "pretrained=MODEL_DIR,device_id=0" --tasks TASKS
   ```

   > Notice: Execute script path:[eval_with_harness.py](https://gitee.com/mindspore/mindformers/blob/dev/toolkit/benchmarks/eval_with_harness.py)

#### Evaluation Parameters

Harness parameters

| Parameter           | Type | Description                     | Required|
|---------------|-----|---------------------------|------|
| `--model`       | str | The value must be **mf**, indicating the MindFormers evaluation policy.| Yes   |
| `--model_args`  | str | Model and evaluation parameters. For details, see "MindFormers model parameters."      | Yes   |
| `--tasks`       | str | Dataset name. Multiple datasets can be specified and separated by commas (,).      | Yes   |
| `--batch_size`  | int | Number of batch processing samples.                   | No   |
| `--limit`       | int | Number of samples for each task. This parameter is mainly used for function tests.         | No   |

MindFormers model parameters

| Parameter          | Type  | Description                             | Required|
|--------------|------|-----------------------------------|------|
| `pretrained`   | str  | Model directory.                           | Yes   |
| `use_past`     | bool | Specifies whether to enable incremental inference. This parameter must be enabled for evaluation tasks of the generate_until type.| No   |
| `device_id`    | int  | Device ID.                             | No   |

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

## VLMEvalKit Evaluation

### Overview

[VLMEvalKit](https://github.com/open-compass/VLMEvalKit)
is an open source toolkit designed for large visual language model evaluation, supporting one-click evaluation of large visual language models on various benchmarks, without the need for complicated data preparation, making the evaluation process easier. It supports a variety of graphic multimodal evaluation sets and video multimodal evaluation sets, a variety of API models and open source models based on PyTorch and HF, and customized prompts and evaluation metrics. After adapting MindFormers based on VLMEvalKit evaluation framework, it supports loading multimodal large models in MindFormers for evaluation.

The currently adapted models and supported evaluation datasets are shown in the table below (the remaining models and evaluation datasets are actively being adapted, please pay attention to version updates):

| Adapted models | Supported evaluation datasets              |
|--|--------------------------------------------|
| cogvlm2-llama3-chat-19B | MME, MMBench, COCO Caption, MMMU, Text-VQA |
| cogvlm2-video-llama3-chat | MMBench-Video, MVBench                     |

### Supported Feature Descriptions

1. Supports automatic download of evaluation datasets;
2. Support for user-defined input of multiple datasets and models;
3. Generate results with one click.

### Installation

Users can follow the following steps to compile and install:

1. Download and modify the code: Due to issues with open source frameworks running MVBench datasets, it is necessary to modify the code by importing [patch](https://github.com/open-compass/VLMEvalKit/issues/633).

    Execute the following command：

    ```bash
    git clone https://github.com/open-compass/VLMEvalKit.git
    cd VLMEvalKit
    git checkout 78a8cef3f02f85734d88d534390ef93ecc4b8bed
    git apply eval.patch
    ```

2. Installation

    There are two installation methods to choose from:

   (1) Used to install Python packages in the current directory (.)(Long time-consuming, easily to debug, commonly used in development environments):

    ```bash
    pip install -e .
    ```

   (2) Read dependencies list from the [requirements.txt](https://github.com/open-compass/VLMEvalKit/blob/main/requirements.txt) file and install these dependencies(Short time-consumption)：

    ```bash
    pip install -r requirements.txt
    ```

### Evaluation

#### Preparations Before Evaluation

1. Create a model directory model_path;
2. Store the YAML file(\*.yaml), and tokenizer file(\*_tokenizer.py) in the model directory. For details, Please refer to the description documents of each model in the [model library](../start/models.md);
3. Configure the yaml file. Refer to [configuration description](../appendix/conf_files.md).

    yaml configuration example:

    ```yaml
    load_checkpoint: "/{path}/model.ckpt"  # Specify the path to the weights file
    model:
      model_config:
        use_past: True                         # Turn on incremental inference
        is_dynamic: False                       # Turn off dynamic shape

      tokenizer:
        vocab_file: "/{path}/tokenizer.model"  # Specify the tokenizer file path
    ```

#### Launching a Single-Card Evaluation Script

```shell
#!/bin/bash

python eval_with_vlmevalkit.py \
  --data dataset \
  --model model_name \
  --verbose \
  --work-dir /{path}/evaluate_result \
  --model-path /{path}/model_path \
  --config-path /{path}/config_path
```

Execute script path: [eval_with_vlmevalkit.py](https://gitee.com/mindspore/mindformers/blob/dev/toolkit/benchmarks/eval_with_vlmevalkit.py)

#### Evaluation Parameters

| Parameters            | Type  | Descriptions                                                                                                                                        | Compulsory(Y/N)|
|---------------|-----|-----------------------------------------------------------------------------------------------------------------------------------------------------|------|
| --data      | str | Name of the dataset, multiple datasets can be passed in, split by spaces.                                                                           | Y    |
| --model  | str | Name of the model.                                                                                                                                  | Y    |
| --verbose       | /   | Outputs logs from the evaluation run.                                                                                                               | N    |
| --work-dir  | str | Directory for storing evaluation results. By default, evaluation results are stored in the folder whose name is the same as the model name. | N    |
| --model-path | str | The folder path containing the model tokenizer files and configuration files.                                       | Y    |
| --config-path       | str | Model configuration file path.                               | Y   |

If the server does not support online downloading of image datasets due to network limitations, you can upload the downloaded .tsv dataset file to the ~/LMUData directory on the server for offline evaluation. (For example: ~/LMUData/MME.tsv or ~/LMUData/MMBench_DEV_EN.tsv or ~/LMUData/COCO_VAL.tsv)

The MMbench-Video dataset evaluation requires the use of the gpt-4-turb model for evaluation and scoring. Please prepare the corresponding apikey in advance.

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

### Viewing Review Results

After evaluating in the above way, find the file ending in .json or .csv in the directory where the evaluation results are stored to view the evaluation results.

The results of the evaluation examples are as follows, where `Bleu` and `ROUGE_L` denote the metrics for evaluating the quality of the translation, and `CIDEr` denotes the metrics for evaluating the image description task.

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

## Using the VideoBench Dataset for Model Evaluation

### Overview

[Video-Bench](https://github.com/PKU-YuanGroup/Video-Bench/tree/main) is the first comprehensive evaluation benchmark for Video-LLMs, featuring a three-level ability assessment that systematically evaluates models in video-exclusive understanding, prior knowledge incorporation, and video-based decision-making abilities.

### Preparations Before Evaluation

1. Download Dataset

    Download[Videos of Video-Bench](https://huggingface.co/datasets/LanguageBind/Video-Bench), the achieved effect is as follows:

    ```text
    egs/VideoBench/
    ├── Eval_video
    │   └── ActivityNet
    │       └── Mp4 and other files
    │   └── Driving-decision-making
    │       └── Mp4 and other files
    |    ...
    ```

2. Download Json

    Download[Jsons of Video-Bench](https://github.com/PKU-YuanGroup/Video-Bench/tree/main?tab=readme-ov-file), the achieved effect is as follows:

    ```text
    egs/Video-Bench/
    ├── Eval_QA
    │   └── QA and other json files
    |    ...
    ```

3. Download the correct answers to all questions

    Download[Answers of Video-Bench](https://huggingface.co/spaces/LanguageBind/Video-Bench/resolve/main/file/ANSWER.json).

### Evaluation

#### Executing Inference Script to Obtain Inference Results

```shell
    python eval_with_videobench.py \
    --model_path model_path \
    --config_path config_path \
    --dataset_name dataset_name \
    --Eval_QA_root Eval_QA_root \
    --Eval_Video_root Eval_Video_root \
    --chat_conversation_output_folder output
```

Execute script path: [eval_with_videobench.py](https://gitee.com/mindspore/mindformers/blob/dev/toolkit/benchmarks/eval_with_videobench.py)

**Parameters Description**

| **Parameters**                 | **Compulsory(Y/N)** | **Description**                                                                                                 |
|----------------------|---------------------|-----------------------------------------------------------------------------------------------------------------|
| model_path           | Y                   | The folder path for storing model related files, including model configuration files and model vocabulary files. |
| config_path          | Y                   | Model configuration file path.                                                                                  |
| dataset_name         | N                   | Evaluation datasets name, default to None, evaluates all subsets of VideoBench.                                 |
| Eval_QA_root         | Y                   | Directory for storing JSON files of VideoBench dataset.                         |
| Eval_Video_root      | Y                   | The video file directory for storing the VideoBench dataset.                                                    |
| chat_conversation_output_folder | N                   | Directory for generating result files. By default, it is stored in the Chat_desults folder of the current directory.                                                                         |

After running, a dialogue result file will be generated in the chat_conversation_output_folder directory.

#### Evaluating and Scoring Based on the Generated Results

Video-Bench can evaluate the answers generated by the model using ChatGPT or T5, and ultimately obtain the final scores for 13 subsets of data.

For example, using ChatGPT for evaluation and scoring:

```shell
python Step2_chatgpt_judge.py \
--model_chat_files_folder ./Chat_results \
--apikey sk-xxxxxxxxxxxxxxxxxxxxxxxxxxxxxx \
--chatgpt_judge_output_folder ./ChatGPT_Judge

python Step3_merge_into_one_json.py \
--chatgpt_judge_files_folder ./ChatGPT_Judge \
--merge_file ./Video_Bench_Input.json
```

The script path in the above evaluation scoring command is: [Step2_chatgpt_judge.py](https://github.com/PKU-YuanGroup/Video-Bench/blob/main/Step2_chatgpt_judge.py), or [Step3_merge_into_one_json.py](https://github.com/PKU-YuanGroup/Video-Bench/blob/main/Step3_merge_into_one_json.py).

Since ChatGPT may answer some formatting errors, you need to run below Step2_chatgpt_judge.py multiple times to ensure that each question is validated by chatgpt.