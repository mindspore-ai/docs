# Evaluation

[![View Source On Gitee](https://mindspore-website.obs.cn-north-4.myhuaweicloud.com/website-images/master/resource/_static/logo_source_en.svg)](https://gitee.com/mindspore/docs/blob/master/docs/mindformers/docs/source_en/feature/evaluation.md)

## Harness Evaluation

### Introduction

[LM Evaluation Harness](https://github.com/EleutherAI/lm-evaluation-harness) is an open-source language model evaluation framework that provides evaluation of more than 60 standard academic datasets, supports multiple evaluation modes such as HuggingFace model evaluation, PEFT adapter evaluation, and vLLM inference evaluation, and supports customized prompts and evaluation metrics, including the evaluation tasks of the loglikelihood, generate_until, and loglikelihood_rolling types. After MindSpore Transformers is adapted based on the Harness evaluation framework, the MindSpore Transformers model can be loaded for evaluation.

The currently verified models and supported evaluation tasks are shown in the table below (the remaining models and evaluation tasks are actively being verified and adapted, please pay attention to version updates):

| Verified models | Supported evaluation tasks                     |
|-----------------|------------------------------------------------|
| Llama3   | gsm8k, ceval-valid, mmlu, cmmlu, race, lambada |
| Llama3.1 | gsm8k, ceval-valid, mmlu, cmmlu, race, lambada |
| Qwen2    | gsm8k, ceval-valid, mmlu, cmmlu, race, lambada |

### Installation

Harness supports two installation methods: pip installation and source code compilation installation. Pip installation is simpler and faster, source code compilation and installation are easier to debug and analyze, and users can choose the appropriate installation method according to their needs.

#### pip Installation

Users can execute the following command to install Harness (Recommend using version 0.4.4):

```shell
pip install lm_eval==0.4.4
```

#### Source Code Compilation Installation

Users can execute the following command to compile and install Harness:

```bash
git clone --depth 1 -b v0.4.4 https://github.com/EleutherAI/lm-evaluation-harness
cd lm-evaluation-harness
pip install -e .
```

### Usage

#### Preparations Before Evaluation

  1. Create a new directory with e.g. the name `model_dir` for storing the model yaml files.
  2. Place the model inference yaml configuration file (predict_xxx_.yaml) in the directory created in the previous step. The directory location of the reasoning yaml configuration file for different models refers to [model library](../introduction/models.md).
  3. Configure the yaml file. If the model class, model Config class, and model Tokenzier class in yaml use cheat code, that is, the code files are in [research](https://gitee.com/mindspore/mindformers/tree/master/research) directory or other external directories, it is necessary to modify the yaml file: under the corresponding class `type` field, add the `auto_register` field in the format of `module.class`. (`module` is the file name of the script where the class is located, and `class` is the class name. If it already exists, there is no need to modify it.).

      Using [predict_1lama3_1_8b. yaml](https://gitee.com/mindspore/mindformers/blob/master/research/llama3_1/llama3_1_8b/predict_llama3_1_8b.yaml) configuration as an example, modify some of the configuration items as follows:

      ```yaml
      run_mode: 'predict'    # Set inference mode
      load_checkpoint: 'model.ckpt'    # path of ckpt
      processor:
        tokenizer:
          vocab_file: "tokenizer.model"    # path of tokenizer
          type: Llama3Tokenizer
          auto_register: llama3_tokenizer.Llama3Tokenizer
      ```

      For detailed instructions on each configuration item, please refer to the [configuration description](../feature/configuration.md).
  4. If you use the `ceval-valid`, `mmlu`, `cmmlu`, `race`, and `lambada` datasets for evaluation, you need to set `use_flash_attention` to `False`. Using `predict_lama3_1_8b.yaml` as an example, modify the yaml as follow:

      ```yaml
      model:
        model_config:
          # ...
          use_flash_attention: False  # Set to False
          # ...
       ```

#### Evaluation Example

Execute the script of [run_harness.sh](https://gitee.com/mindspore/mindformers/blob/master/toolkit/benchmarks/run_harness.sh) to evaluate.

The following table lists the parameters of the script of `run_harness.sh`:

| Parameter           | Type | Description                                                                                                                                                                                   | Required |
|---------------|------|-----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|----------|
| `--register_path`| str | The absolute path of the directory where the cheat code is located. For example, the model directory under the [research](https://gitee.com/mindspore/mindformers/tree/master/research) directory. | No(The cheat code is required)     |
| `--model`       | str  | The value must be `mf`, indicating the MindSpore Transformers evaluation policy.                                                                                                                          | Yes      |
| `--model_args`  | str  | Model and evaluation parameters. For details, see MindSpore Transformers model parameters.                                                                                                            | Yes      |
| `--tasks`       | str  | Dataset name. Multiple datasets can be specified and separated by commas (,).                                                                                                                 | Yes      |
| `--batch_size`  | int  | Number of batch processing samples.                                                                                                                                                           | No       |

The following table lists the parameters of `model_args`:

| Parameter          | Type | Description                                                              | Required |
|--------------|------|--------------------------------------------------------------------------|----------|
| `pretrained`   | str  | Model directory.                                                         | Yes      |
| `max_length`   | int  | Maximum length of model generation.                                      | No       |
| `use_parallel` | bool | Enable parallel strategy (It must be enabled for multi card evaluation). | No       |
| `tp`           | int  | The number of parallel tensors.                                          | No       |
| `dp`           | int  | The number of parallel data.                                             | No       |

Harness evaluation supports single-device single-card, single-device multiple-card, and multiple-device multiple-card scenarios, with sample evaluations for each scenario listed below:

1. Single Card Evaluation Example

   ```shell
      source toolkit/benchmarks/run_harness.sh \
       --register_path mindformers/research/llama3_1 \
       --model mf \
       --model_args pretrained=model_dir \
       --tasks gsm8k
   ```

2. Multi Card Evaluation Example

   ```shell
      source toolkit/benchmarks/run_harness.sh \
       --register_path mindformers/research/llama3_1 \
       --model mf \
       --model_args pretrained=model_dir,use_parallel=True,tp=4,dp=1 \
       --tasks ceval-valid \
       --batch_size BATCH_SIZE WORKER_NUM
   ```

    - `BATCH_SIZE` is the sample size for batch processing of models;
    - `WORKER_NUM` is the number of compute devices.

3. Multi-Device and Multi-Card Example

   Node 0 (Master) Command:

      ```shell
         source toolkit/benchmarks/run_harness.sh \
          --register_path mindformers/research/llama3_1 \
          --model mf \
          --model_args pretrained=model_dir,use_parallel=True,tp=8,dp=1 \
          --tasks lambada \
          --batch_size 2 8 4 192.168.0.0 8118 0 output/msrun_log False 300
      ```

   Node 1 (Secondary Node) Command:

      ```shell
         source toolkit/benchmarks/run_harness.sh \
          --register_path mindformers/research/llama3_1 \
          --model mf \
          --model_args pretrained=model_dir,use_parallel=True,tp=8,dp=1 \
          --tasks lambada \
          --batch_size 2 8 4 192.168.0.0 8118 1 output/msrun_log False 300
      ```

   Node n (Nth Node) Command:

      ```shell
         source toolkit/benchmarks/run_harness.sh \
          --register_path mindformers/research/llama3_1 \
          --model mf \
          --model_args pretrained=model_dir,use_parallel=True,tp=8,dp=1 \
          --tasks lambada \
          --batch_size BATCH_SIZE WORKER_NUM LOCAL_WORKER MASTER_ADDR MASTER_PORT NODE_RANK output/msrun_log False CLUSTER_TIME_OUT
      ```

   - `BATCH_SIZE` is the sample size for batch processing of models;
   - `WORKER_NUM` is the total number of compute devices used on all nodes;
   - `LOCAL_WORKER` is the number of compute devices used on the current node;
   - `MASTER_ADDR` is the ip address of the primary node to be started in distributed mode;
   - `MASTER_PORT` is the Port number bound for distributed startup;
   - `NODE_RANK` is the Rank ID of the current node;
   - `CLUSTER_TIME_OUT`is the waiting time for distributed startup, in seconds.

   To execute the multi-node multi-device script for evaluating, you need to run the script on different nodes and set MASTER_ADDR to the IP address of the primary node. The IP address should be the same across all nodes, and only the NODE_RANK parameter varies across nodes.

### Viewing the Evaluation Results

After executing the evaluation command, the evaluation results will be printed out on the terminal. Taking gsm8k as an example, the evaluation results are as follows, where Filter corresponds to the way the matching model outputs results, n-shot corresponds to content format of dataset, Metric corresponds to the evaluation metric, Value corresponds to the evaluation score, and Stderr corresponds to the score error.

| Tasks | Version | Filter           | n-shot | Metric      |   | Value  |   | Stderr |
|-------|--------:|------------------|-------:|-------------|---|--------|---|--------|
| gsm8k |       3 | flexible-extract |      5 | exact_match | ↑ | 0.5034 | ± | 0.0138 |
|       |         | strict-match     |      5 | exact_match | ↑ | 0.5011 | ± | 0.0138 |

### FAQ

1. Use Harness for evaluation, when loading the HuggingFace datasets, report `SSLError`:

   Refer to [SSL Error reporting solution](https://stackoverflow.com/questions/71692354/facing-ssl-error-with-huggingface-pretrained-models).

   Note: Turning off SSL verification is risky and may be exposed to MITM. It is only recommended to use it in the test environment or in the connection you fully trust.

## Evaluation after training

### Overview

After training, the model generally uses the trained model weights to run evaluation tasks to verify the training effect. This chapter introduces the necessary steps from training to evaluation, including:

1. Processing of distributed weights after training (this step can be ignored for single-card training);
2. Writing inference configuration files for evaluation based on the training configuration;
3. Running a simple inference task to verify the correctness of the above steps;
4. Performing the evaluation task.

Users can refer to this document to evaluate their trained models.

### Distributed Weight Merging

If the weights generated after training are distributed, the existing distributed weights need to be merged into complete weights first, and then the weights can be loaded through online slicing to complete the inference task. Using the [safetensors weight merging script](https://gitee.com/mindspore/mindformers/blob/master/toolkit/safetensors/unified_safetensors.py) provided by MindSpore Transformers, the merged weights are in the format of complete weights.

Parameters can be filled in as follows:

```shell
python toolkit/safetensors/unified_safetensors.py \
  --src_strategy_dirs src_strategy_path_or_dir \
  --mindspore_ckpt_dir mindspore_ckpt_dir\
  --output_dir output_dir \
  --file_suffix "1_1" \
  --filter_out_param_prefix "adam_"
```

Script parameter description:

- src_strategy_dirs: The path to the distributed strategy file corresponding to the source weight, usually saved in the output/strategy/ directory by default after starting the training task. Distributed weights need to be filled in according to the following situations:

   1. Source weights enable pipeline parallelism: Weight conversion is based on the merged strategy file, fill in the path of the distributed strategy folder. The script will automatically merge all ckpt_strategy_rank_x.ckpt files in the folder and generate merged_ckpt_strategy.ckpt in the folder. If merged_ckpt_strategy.ckpt already exists, you can directly fill in the path of this file.
   2. Source weights do not enable pipeline parallelism: Weight conversion can be based on any strategy file, just fill in the path of any ckpt_strategy_rank_x.ckpt file.

   Note: If merged_ckpt_strategy.ckpt already exists in the strategy folder and the folder path is still passed in, the script will first delete the old merged_ckpt_strategy.ckpt and then merge to generate a new merged_ckpt_strategy.ckpt for weight conversion. Therefore, please ensure that the folder has sufficient write permissions, otherwise the operation will report an error.

- mindspore_ckpt_dir: Path to distributed weights, please fill in the path of the folder where the source weights are located. The source weights should be stored in the format model_dir/rank_x/xxx.safetensors, and fill in the folder path as model_dir.
- output_dir: Save path of target weights, the default value is `/new_llm_data/******/ckpt/nbg3_31b/tmp`, that is, the target weights will be placed in the `/new_llm_data/******/ckpt/nbg3_31b/tmp` directory.
- file_suffix: Naming suffix of target weight files, the default value is "1_1", that is, the target weights will be searched in the format *1_1.safetensors.
- has_redundancy: Whether the merged source weights are redundant weights, the default is True.
- filter_out_param_prefix: When merging weights, you can customize to filter out some parameters, and the filtering rules match by prefix name, such as optimizer parameters "adam_".
- max_process_num: Maximum number of processes for merging. Default value: 64.

### Inference Configuration Development

After completing the merging of weight files, you need to develop the corresponding inference configuration file based on the training configuration file.

Taking Qwen3 as an example, modify the [Qwen3 training configuration](https://gitee.com/mindspore/mindformers/blob/master/configs/qwen3/finetune_qwen3.yaml) based on the [Qwen3 inference configuration](https://gitee.com/mindspore/mindformers/blob/master/configs/qwen3/predict_qwen3.yaml):

Main modification points of Qwen3 training configuration include:

- Modify the value of run_mode to "predict".
- Add pretrained_model_dir: Hugging Face or ModelScope model directory path, place model configuration, Tokenizer and other files.
- In parallel_config, only keep data_parallel and model_parallel.
- In model_config, only keep compute_dtype, layernorm_compute_dtype, softmax_compute_dtype, rotary_dtype, params_dtype, and keep the precision consistent with the inference configuration.
- In the parallel module, only keep parallel_mode and enable_alltoall, and modify the value of parallel_mode to "MANUAL_PARALLEL".

### Inference Function Verification

After the weights and configuration files are ready, use a single data input for inference to check whether the output content meets the expected logic. Refer to the [inference document](https://gitee.com/mindspore/docs/blob/master/docs/mindformers/docs/source_en/guide/inference.md) to start the inference task.

For example:

```shell
python run_mindformer.py \
--config configs/qwen3/predict_qwen3.yaml \
--run_mode predict \
--use_parallel False \
--predict_data '帮助我制定一份去上海的旅游攻略'
```

If the output content appears garbled or does not meet expectations, you need to locate the precision problem.

1. Check the correctness of the model configuration

    Confirm that the model structure is consistent with the training configuration. Refer to the training configuration template usage tutorial to ensure that the configuration file complies with specifications and avoid inference exceptions caused by parameter errors.

2. Verify the completeness of weight loading

    Check whether the model weight files are loaded completely, and ensure that the weight names strictly match the model structure. Refer to the new model weight conversion adaptation tutorial to view the weight log, that is, whether the weight slicing method is correct, to avoid inference errors caused by mismatched weights.

3. Locate inference precision issues

    If the model configuration and weight loading are both correct, but the inference results still do not meet expectations, precision comparison analysis is required. Refer to the inference precision comparison document to compare the output differences between training and inference layer by layer, and troubleshoot potential data preprocessing, computational precision, or operator issues.

### Evaluation using AISBench

Refer to the AISBench evaluation section and use the AISBench tool for evaluation to verify model precision.