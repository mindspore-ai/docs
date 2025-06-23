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
  3. Configure the yaml file. If the model class, model Config class, and model Tokenzier class in yaml use cheat code, that is, the code files are in [research](https://gitee.com/mindspore/mindformers/tree/dev/research) directory or other external directories, it is necessary to modify the yaml file: under the corresponding class `type` field, add the `auto_register` field in the format of `module.class`. (`module` is the file name of the script where the class is located, and `class` is the class name. If it already exists, there is no need to modify it.).

      Using [predict_1lama3_1_8b. yaml](https://gitee.com/mindspore/mindformers/blob/dev/research/llama3_1/llama3_1_8b/predict_llama3_1_8b.yaml) configuration as an example, modify some of the configuration items as follows:

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

Execute the script of [run_harness.sh](https://gitee.com/mindspore/mindformers/blob/dev/toolkit/benchmarks/run_harness.sh) to evaluate.

The following table lists the parameters of the script of `run_harness.sh`:

| Parameter           | Type | Description                                                                                                                                                                                   | Required |
|---------------|------|-----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|----------|
| `--register_path`| str | The absolute path of the directory where the cheat code is located. For example, the model directory under the [research](https://gitee.com/mindspore/mindformers/tree/dev/research) directory. | No(The cheat code is required)     |
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

## FAQ

1. Use Harness for evaluation, when loading the HuggingFace datasets, report `SSLError`:

   Refer to [SSL Error reporting solution](https://stackoverflow.com/questions/71692354/facing-ssl-error-with-huggingface-pretrained-models).

   Note: Turning off SSL verification is risky and may be exposed to MITM. It is only recommended to use it in the test environment or in the connection you fully trust.