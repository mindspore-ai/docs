# Inference

[![View Source On Gitee](https://mindspore-website.obs.cn-north-4.myhuaweicloud.com/website-images/master/resource/_static/logo_source_en.svg)](https://gitee.com/mindspore/docs/blob/master/docs/mindformers/docs/source_en/guide/inference.md)

## Overview

MindSpore Transformers offers large model inference capabilities. Users can execute the `run_mindformer` unified script for inference. By using the `run_mindformer` unified script, users can start the process directly through configuration files without writing any code, making it very convenient to use.

## Basic Process

The inference process can be categorized into the following steps:

### 1. Models of Selective Inference

Depending on the required inference task, different models are chosen, e.g. for text generation one can choose Qwen3.

### 2. Preparing Model Files

Obtain the Hugging Face model file: weights, configurations, and tokenizers. Store the downloaded files in the same directory for convenient subsequent use.

### 3. YAML Configuration File Modification

The user needs to configure a YAML file to define all the configurations of the task. MindSpore Transformers provides a YAML configuration template. Users can customize the configuration based on the template according to the actual scenario. For detailed information, please refer to the [Guide to Using Inference Configuration Templates](https://www.mindspore.cn/mindformers/docs/en/master/advanced_development/yaml_config_inference.html).

### 4. Executing Inference Tasks

Use the unified script `run_mindformer` to execute inference tasks.

## Inference Based on the run_mindformer Script

For single-device inference, you can directly run [run_mindformer.py](https://gitee.com/mindspore/mindformers/blob/master/run_mindformer.py). For multi-device inference, you need to run [scripts/msrun_launcher.sh](https://gitee.com/mindspore/mindformers/blob/master/scripts/msrun_launcher.sh).

The arguments to run_mindformer.py are described below:

| Parameters               | Parameter Descriptions                                                                                                                             |
|:-------------------------|:---------------------------------------------------------------------------------------------------------------------------------------------------|
| config                   | Path to the yaml configuration file                                                                                                                |
| run_mode                 | The running mode, with inference set to predict                                                                                                    |
| use_parallel             | Whether to use multi-card inference                                                                                                                 |
| predict_data             | Input data for inference. Multi-batch inference needs to pass the path to the txt file of the input data, which contains multiple lines of inputs. |
| predict_batch_size       | batch_size for multi-batch inference                                                                                                               |

msrun_launcher.sh includes the run_mindformer.py command and the number of inference cards as two parameters.

The following will describe the usage of single and multi-card inference using Qwen3-8B as an example, with the recommended configuration of the [predict_qwen3.yaml](https://gitee.com/mindspore/mindformers/blob/master/configs/qwen3/predict_qwen3.yaml) file.

### Configuration Modification

The current inference can directly reuse Hugging Face's configuration file and tokenizer, and load the weights of Hugging Face's safetensors format online. The configuration modification when in use is as follows:

```yaml
use_legacy: False
pretrained_model_dir: '/path/hf_dir'
```

Parameter Description:

- use_legacy: Determine whether to use the old architecture. Default value: 'True';
- pretrained_model_dir: Hugging Face model directory path, where files such as model configuration and Tokenizer are placed. The contents in `/path/hf_dir` are as follows:

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

The default configuration is single-card inference configuration. The relevant configuration is as follows:

```yaml
use_parallel: False
parallel_config:
  data_parallel: 1
  model_parallel: 1
```

If multi-card inference tasks need to be executed, the relevant configuration modifications are as follows:

```yaml
use_parallel: True
parallel_config:
  data_parallel: 1
  model_parallel: 2 # Modify to the actual number of cards used
```

For specific configuration instructions, please refer to [yaml Configuration Instructions](https://www.mindspore.cn/mindformers/docs/en/master/feature/configuration.html).

### Single-Device Inference

After completing the modification according to the [Configuration Modification](#configuration-modification) section, execute the following command to start the single-card inference task:

```shell
python run_mindformer.py \
--config configs/qwen3/predict_qwen3.yaml \
--run_mode predict \
--use_parallel False \
--predict_data '帮助我制定一份去上海的旅游攻略'
```

The following results appear, proving the success of the inference. The inference results will also be saved to the `text_generation_result.txt` file in the current directory.

```text
'text_generation_text': [帮助我制定一份去上海的旅游攻略，包括景点、美食、住宿等信息...]
```

### Multi-Card Inference

The configuration requirements for multi-card inference are different from those for single-card inference. Please refer to the following for configuration modification:

1. The configuration of model_parallel and the number of cards used need to be consistent. The following use case is 4-card inference, and model_parallel needs to be set to 4;
2. The current version of multi-card inference does not support data parallelism. data_parallel needs to be set to 1.

After completing the modification according to the [Configuration Modification](#configuration-modification) section, execute the following command to start the multi-card inference task:

```shell
bash scripts/msrun_launcher.sh "run_mindformer.py \
 --config configs/qwen3/predict_qwen3.yaml \
 --run_mode predict \
 --use_parallel True \
 --predict_data '帮助我制定一份去上海的旅游攻略'" 4
```

The following results appear, proving the success of the inference. The inference results will also be saved to the text_generation_result.txt file in the current directory. Detailed logs can be viewed through the directory `./output/msrun_log`.

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

Take full weight inference as an example. The inference task can be started by referring to the following command:

```shell
bash scripts/msrun_launcher.sh "run_mindformer.py \
 --config configs/qwen3/predict_qwen3.yaml \
 --run_mode predict \
 --predict_batch_size 4 \
 --use_parallel True \
 --predict_data path/to/input_predict_data.txt" 4
```

Inference results are viewed in the same way as multi-card inference.

## More Information

For more inference examples of different models, see [the models supported by MindSpore Transformers](https://www.mindspore.cn/mindformers/docs/en/master/introduction/models.html).
