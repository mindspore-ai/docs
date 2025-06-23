# Inference

[![View Source On Gitee](https://mindspore-website.obs.cn-north-4.myhuaweicloud.com/website-images/master/resource/_static/logo_source_en.svg)](https://gitee.com/mindspore/docs/blob/master/docs/mindformers/docs/source_en/guide/inference.md)

## Overview

MindSpore Transformers offers large model inference capabilities. Users can execute the `run_mindformer` unified script for inference. By using the `run_mindformer` unified script, users can start the process directly through configuration files without writing any code, making it very convenient to use.

## Basic Process

The inference process can be categorized into the following steps:

### 1. Models of Selective Inference

Depending on the required inference task, different models are chosen, e.g. for text generation one can choose `Qwen2.5-7B`, etc.

### 2. Preparing Model Weights

Currently, the inference weights can be loaded online to perform inference with the complete weights. The weights can be obtained through the following two methods:

1. Download the complete open-source weights of the corresponding model from the Hugging Face model library.
2. Pre-trained or fine-tuned distributed weights through [merger](https://www.mindspore.cn/mindformers/docs/en/dev/feature/safetensors.html#weight-merging) Generate a complete weight.

### 3. Executing Inference Tasks

Use the unified script `run_mindformer` to execute inference tasks.

## Inference Based on the run_mindformer Script

For single-device inference, you can directly run [run_mindformer.py](https://gitee.com/mindspore/mindformers/blob/dev/run_mindformer.py). For multi-device inference, you need to run [scripts/msrun_launcher.sh](https://gitee.com/mindspore/mindformers/blob/dev/scripts/msrun_launcher.sh).

The arguments to run_mindformer.py are described below:

| Parameters               | Parameter Descriptions                                                                                                                             |
|:-------------------------|:---------------------------------------------------------------------------------------------------------------------------------------------------|
| config                   | Path to the yaml configuration file                                                                                                                |
| run_mode                 | The running mode, with inference set to predict                                                                                                    |
| use_parallel             | Whether to use multicard inference                                                                                                                 |
| load_checkpoint          | the loaded weight path                                                                                                                             |
| predict_data             | Input data for inference. Multi-batch inference needs to pass the path to the txt file of the input data, which contains multiple lines of inputs. |
| auto_trans_ckpt          | Automatic weight slicing. Default value is False                                                                                                   |
| src_strategy_path_or_dir | Path to the strategy file for weights                                                                                                              |
| predict_batch_size       | batch_size for multi-batch inference                                                                                                               |
| modal_type               | Given modal type corresponds to predict data in multimodal inference scenario.                                                                     |

msrun_launcher.sh includes the run_mindformer.py command and the number of inference cards as two parameters.

The following will describe the usage of single and multi-card inference using `Qwen2.5-7B` as an example, with the recommended configuration of the [predict_qwen2_5_7b_instruct.yaml](https://gitee.com/mindspore/mindformers/blob/dev/research/qwen2_5/predict_qwen2_5_7b_instruct.yaml) file.

### Configuration Modification

The configuration related to weights is modified as follows:

```yaml
load_checkpoint: "path/to/Qwen2_5_7b_instruct/"
load_ckpt_format: 'safetensors'
auto_trans_ckpt: True
```

The default configuration is the single-card inference configuration. The parallel related configuration is modified as follows:

```yaml
use_parallel: False
parallel_config:
  data_parallel: 1
  model_parallel: 1
  pipeline_stage: 1
```

The configuration related to `tokenizer` is modified as follows:

```yaml
processor:
  tokenizer:
    vocab_file: "path/to/vocab.json"
    merges_file: "path/to/merges.txt"
```

For specific configuration instructions, please refer to [yaml Configuration Instructions](https://www.mindspore.cn/mindformers/docs/en/dev/feature/configuration.html).

### Single-Device Inference

When using full weight reasoning, it is recommended to use the default configuration and execute the following command to start the reasoning task:

```shell
python run_mindformer.py \
--register_path /path/to/research/qwen2_5/ \
--config /path/to/research/qwen2_5/predict_qwen2_5_7b_instruct \
--run_mode predict \
--use_parallel False \
--predict_data '帮助我制定一份去上海的旅游攻略'
```

The following results appear, proving the success of the reasoning. The reasoning results will also be saved to the `text_generation_result.txt` file in the current directory.

```text
'text_generation_text': [帮助我制定一份去上海的旅游攻略，包括景点、美食、住宿等信息...]
```

### Multi-Card Inference

The configuration requirements for multi-card inference are different from those for single-card inference. Please refer to the following for configuration modification:

1. The configuration of model_parallel and the number of cards used need to be consistent. The following use case is 4-card inference, and model_parallel needs to be set to 4;
2. The current version of multi-card inference does not support data parallelism. data_parallel needs to be set to 1.

When using full weight reasoning, it is necessary to enable the online splitting mode to load the weights. Refer to the following command:

```shell
bash scripts/msrun_launcher.sh "run_mindformer.py \
 --register_path /path/to/research/qwen2_5 \
 --config /path/to/research/qwen2_5/qwen2_5_72b/predict_qwen2_5_72b_instruct.yaml \
 --run_mode predict \
 --use_parallel True \
 --auto_trans_ckpt True \
 --predict_data '帮助我制定一份去上海的旅游攻略'" 4
```

The following results appear, proving the success of the reasoning. The reasoning results will also be saved to the text_generation_result.txt file in the current directory. Detailed logs can be viewed through the directory `./output/msrun_log`.

```text
'text_generation_text': [帮助我制定一份去上海的旅游攻略，包括景点、美食、住宿等信息...]
```

### Multi-Device Multi-Batch Inference

Multi-card multi-batch inference is initiated in the same way as [multi-card inference](#multi-card-inference), but requires the addition of the `predict_batch_size` inputs and the modification of the `predict_data` inputs.

The content and format of the `input_predict_data.txt` file is an input each line, and the number of questions is the same as the `predict_batch_size`, which can be found in the following format:

```text
帮助我制定一份去上海的旅游攻略
帮助我制定一份去上海的旅游攻略
帮助我制定一份去上海的旅游攻略
帮助我制定一份去上海的旅游攻略
```

Take full weight reasoning as an example. The reasoning task can be started by referring to the following command:

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

Inference results are viewed in the same way as multi-card inference.

## More Information

For more inference examples of different models, see [the models supported by MindSpore Transformers](https://www.mindspore.cn/mindformers/docs/en/dev/introduction/models.html).
