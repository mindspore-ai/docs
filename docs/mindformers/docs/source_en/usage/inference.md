# Inference

[![View Source On Gitee](https://mindspore-website.obs.cn-north-4.myhuaweicloud.com/website-images/r2.6.0rc1/resource/_static/logo_source_en.svg)](https://gitee.com/mindspore/docs/blob/r2.6.0rc1/docs/mindformers/docs/source_en/usage/inference.md)

## Overview

MindSpore Transformers provides the foundation model inference capability. Users can run the unified script `run_mindformer` or write a script to call the high-level API to start inference. If the unified script `run_mindformer` is used, you can directly start the system through the configuration file without writing code.

## Basic Process

The inference process can be categorized into the following steps:

### 1. Models of Selective Inference

Depending on the required inference task, different models are chosen, e.g. for text generation one can choose Llama2, etc.

### 2. Preparing Model Weights

Model weights can be categorized into two types: complete weights and distributed weights, and the following instructions should be referred to when using them.

#### 2.1 Complete Weights

Complete weights can be obtained in two ways:

1. After downloading the open source weights of the corresponding model from the HuggingFace model library, refer to [Weight Format Conversion](https://www.mindspore.cn/mindformers/docs/en/r1.5.0/function/weight_conversion.html) to convert them to the ckpt format.
2. Pre-trained or fine-tuned distributed weights are used to generate a complete weight by [merging](https://www.mindspore.cn/mindformers/docs/en/r1.5.0/function/transform_weight.html).

#### 2.2 Distributed Weights

Distributed weights are typically obtained by pre-training or after fine-tuning and are stored by default in the `. /output/checkpoint_network` directory, which needs to be converted to single-card or multi-card weights before performing single-card or multi-card inference.

If the inference uses a weight slicing that is different from the model slicing provided in the inference task, such as in these cases below, the weights need to be additionally converted to a slice that matches the slicing of the model in the actual inference task.

1. The weights obtained from multi-card training are reasoned on a single card;
2. The weights of the eight-card training are reasoned over two cards;
3. Already sliced distributed weights are reasoned on a single card, and so on.

The command samples in the following contents are all used in the way of online autoslicing. It is recommended to use online autoslicing by setting the command parameters `--auto_trans_ckpt` to `-True` and `-src_strategy_path_or_dir` to the weighted slicing strategy file or directory path (which is saved by default after training under `./output/strategy`) are automatically sliced in the inference task. Details can be found in [Distributed Weight Slicing and Merging](https://www.mindspore.cn/mindformers/docs/en/r1.5.0/function/transform_weight.html).

> Since both the training and inference tasks use `. /output` as the default output path, when using the strategy file output by the training task as the source weight strategy file for the inference task, you need to move the strategy file directory under the default output path to another location to avoid it being emptied by the process of the inference task, for example:
>
> ```mv ./output/strategy/ ./strategy```

### 3. Executing Inference Tasks

Call the high-level API or use the unified script `run_mindformer` to execute inference tasks.

## Inference Based on the run_mindformer Script

For single-device inference, you can directly run [run_mindformer.py](https://gitee.com/mindspore/mindformers/blob/r1.5.0/run_mindformer.py). For multi-device inference, you need to run [scripts/msrun_launcher.sh](https://gitee.com/mindspore/mindformers/blob/r1.5.0/scripts/msrun_launcher.sh).

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

The following will describe the usage of single and multi-card inference using Llama2 as an example, with the recommended configuration of the [predict_llama2_7b.yaml](https://gitee.com/mindspore/mindformers/blob/r1.5.0/configs/llama2/predict_llama2_7b.yaml) file.

> During inference, the vocabulary file `tokenizer.model` required for the Llama2 model will be automatically downloaded (ensuring smooth network connectivity). If the file exists locally, you can place it in the `./checkpoint_download/Llama2/` directory in advance.

### Single-Device Inference

When using complete weight inference, the following command is executed to start the inference task:

```shell
python run_mindformer.py \
--config configs/llama2/predict_llama2_7b.yaml \
--run_mode predict \
--use_parallel False \
--load_checkpoint path/to/checkpoint.ckpt \
--predict_data 'I love Beijing, because'
```

If you use distributed weight files for inference, you need to add the `--auto_trans_ckpt` and `-src_strategy_path_or_dir` entries, with the following startup commands:

```shell
python run_mindformer.py \
--config configs/llama2/predict_llama2_7b.yaml \
--run_mode predict \
--use_parallel False \
--auto_trans_ckpt True \
--src_strategy_path_or_dir ./strategy \
--load_checkpoint path/to/checkpoint.ckpt \
--predict_data 'I love Beijing, because'
```

The following result appears, proving that the inference was successful. The inference result is also saved to the `text_generation_result.txt` file in the current directory. The detailed log can be viewed in the `. /output/msrun_log` directory.

```text
'text_generation_text': [I love Beijing, because it is a city that is constantly constantly changing. I have been living here for ......]
```

### Multi-Card Inference

The configuration requirements for multi-card inference differ from those of single card, and you need to refer to the following instructions to modify the [predict_llama2_7b.yaml](https://gitee.com/mindspore/mindformers/blob/r1.5.0/configs/llama2/predict_llama2_7b.yaml) configuration.

1. The configuration of model_parallel and the number of cards used need to be consistent. The following use case is 2-card inference, and model_parallel needs to be set to 2;
2. The current version of multi-card inference does not support data parallelism, you need to set data_parallel to 1.

**Configuration before modification:**

```yaml
parallel_config:
  data_parallel: 8
  model_parallel: 1
  pipeline_stage: 1
```

**Configuration after modifications:**

```yaml
parallel_config:
  data_parallel: 1
  model_parallel: 2
  pipeline_stage: 1
```

When full weight inference is used, you need to enable the online slicing mode to load weights. For details, see the following command:

```shell
bash scripts/msrun_launcher.sh "python run_mindformer.py \
--config configs/llama2/predict_llama2_7b.yaml \
--run_mode predict \
--use_parallel True \
--auto_trans_ckpt True \
--load_checkpoint path/to/checkpoint.ckpt \
--predict_data 'I love Beijing, because'" \
2
```

Refer to the following commands when distributed weight inference is used and the slicing strategy for the weights is the same as the slicing strategy for the model:

```shell
bash scripts/msrun_launcher.sh "python run_mindformer.py \
--config configs/llama2/predict_llama2_7b.yaml \
--run_mode predict \
--use_parallel True \
--load_checkpoint path/to/checkpoint_dir \
--predict_data 'I love Beijing, because'" \
2
```

When distributed weight inference is used and the slicing strategy of the weights is not consistent with the slicing strategy of the model, you need to enable the online slicing function to load the weights. Refer to the following command:

```shell
bash scripts/msrun_launcher.sh "python run_mindformer.py \
--config configs/llama2/predict_llama2_7b.yaml \
--run_mode predict \
--use_parallel True \
--auto_trans_ckpt True \
--src_strategy_path_or_dir ./strategy \
--load_checkpoint path/to/checkpoint_dir \
--predict_data 'I love Beijing, because'" \
2
```

Inference results are viewed in the same way as single-card inference.

### Multi-Device Multi-Batch Inference

Multi-card multi-batch inference is initiated in the same way as [multi-card inference](#multi-card-inference), but requires the addition of the `predict_batch_size` inputs and the modification of the `predict_data` inputs.

The content and format of the `input_predict_data.txt` file is an input each line, and the number of questions is the same as the `predict_batch_size`, which can be found in the following format:

```txt
I love Beijing, because
I love Beijing, because
I love Beijing, because
I love Beijing, because
```

Refer to the following commands to perform inference tasks, taking the full weight inference as an example:

```shell
bash scripts/msrun_launcher.sh "python run_mindformer.py \
--config configs/llama2/predict_llama2_7b.yaml \
--run_mode predict \
--predict_batch_size 4 \
--use_parallel True \
--auto_trans_ckpt True \
--load_checkpoint path/to/checkpoint.ckpt \
--predict_data path/to/input_predict_data.txt" \
2
```

Inference results are viewed in the same way as single-card inference.

### Multimodal Inference

Use `cogvlm2-llama3-chat-19B` model as example and see the following process with details:

Modify configuration yaml file[predict_cogvlm2_image_llama3_chat_19b.yaml](https://gitee.com/mindspore/mindformers/blob/r1.5.0/configs/cogvlm2/predict_cogvlm2_image_llama3_chat_19b.yaml).

```shell
model:
  model_config:
    use_past: True                         # Turn on incremental inference
    is_dynamic: False                      # Turn off dynamic shape

  tokenizer:
    vocab_file: "/{path}/tokenizer.model"  # Specify the tokenizer file path
```

Run inference scripts.

```shell
python run_mindformer.py \
 --config configs/cogvlm2/predict_cogvlm2_image_llama3_chat_19b.yaml \
 --run_mode predict \
 --predict_data "/path/image.jpg" "Please describe this image." \  # input data,first input is image path,second input is text path.
 --modal_type image text \                                         # modal type for input data, 'image' type for image path, 'text' type for text path.
 --load_checkpoint /{path}/cogvlm2-image-llama3-chat.ckpt
```

## Inference Based on High-level Interface

> For security reasons, it is not recommended to use high-level interfaces for inference. This chapter will be deprecated in the next version. If you have any questions or suggestions, please submit feedback through [Community Issue](https://gitee.com/mindspore/mindformers/issues/new). Thank you for your understanding and support!

MindSpore Transformers not only provides a unified script for `run_mindformer` inference, but also supports user-defined calls to high-level interfaces such as `pipeline` or `chat` for implementation.

### Pipeline Interface

Customized text generation inference task flow based on `pipeline` interface, supporting single card inference and multi-card inference. About how to use `pipeline` interface to start the task and output the result, you can refer to the following implementation. The specific parameter description can be viewed [pipeline interface API documentation](https://www.mindspore.cn/mindformers/docs/en/r1.5.0/mindformers/mindformers.pipeline.html#mindformers.pipeline).

#### Incremental Inference

```python
from mindformers import build_context
from mindformers import AutoModel, AutoTokenizer, pipeline, TextStreamer

# Construct the input content.
inputs = ["I love Beijing, because", "LLaMA is a", "Huawei is a company that"]

# Initialize the environment.
build_context({'context': {'mode': 0}, 'run_mode': 'predict', 'parallel': {}, 'parallel_config': {}})

# Instantiate a tokenizer.
tokenizer = AutoTokenizer.from_pretrained('llama2_7b')

# Instantiate a model.
# Modify the path to the local weight path.
model = AutoModel.from_pretrained('llama2_7b', checkpoint_name_or_path="path/to/llama2_7b.ckpt", use_past=True)
# Model instantiation is also supported from modelers.cn.Given repo id which format is MindSpore-Lab/model_name
# model = AutoModel.from_pretrained('MindSpore-Lab/qwen1_5_7b-chat')

# Start a non-stream inference task in the pipeline.
text_generation_pipeline = pipeline(task="text_generation", model=model, tokenizer=tokenizer)
outputs = text_generation_pipeline(inputs, max_length=512, do_sample=False, top_k=3, top_p=1)
for output in outputs:
    print(output)
```

Save the example to `pipeline_inference.py`, modify the path for loading the weight, and run the `pipeline_inference.py` script.

```shell
python pipeline_inference.py
```

The inference result is as follows:

```text
'text_generation_text': [I love Beijing, because it is a city that is constantly constantly changing. I have been living here for ......]
'text_generation_text': [LLaMA is a large-scale, open-source, multimodal, multilingual, multitask, and multimodal pretrained language model. It is ......]
'text_generation_text': [Huawei is a company that has been around for a long time. ......]
```

#### Stream Inference

```python
from mindformers import build_context
from mindformers import AutoModel, AutoTokenizer, pipeline, TextStreamer

# Construct the input content.
inputs = ["I love Beijing, because", "LLaMA is a", "Huawei is a company that"]

# Initialize the environment.
build_context({'context': {'mode': 0}, 'run_mode': 'predict', 'parallel': {}, 'parallel_config': {}})

# Instantiate a tokenizer.
tokenizer = AutoTokenizer.from_pretrained('llama2_7b')

# Instantiate a model.
# Modify the path to the local weight path.
model = AutoModel.from_pretrained('llama2_7b', checkpoint_name_or_path="path/to/llama2_7b.ckpt", use_past=True)
# Model instantiation is also supported from modelers.cn.Given repo id which format is MindSpore-Lab/model_name
# model = AutoModel.from_pretrained('MindSpore-Lab/qwen1_5_7b-chat')

# Start a stream inference task in the pipeline.
streamer = TextStreamer(tokenizer)
text_generation_pipeline = pipeline(task="text_generation", model=model, tokenizer=tokenizer, streamer=streamer)
_ = text_generation_pipeline(inputs, max_length=512, do_sample=False, top_k=3, top_p=1)
```

Save the example to `pipeline_inference.py`, modify the path for loading the weight, and run the `pipeline_inference.py` script.

```shell
python pipeline_inference.py
```

The inference result is as follows:

```text
'text_generation_text': [I love Beijing, because it is a city that is constantly constantly changing. I have been living here for ......]
'text_generation_text': [LLaMA is a large-scale, open-source, multimodal, multilingual, multitask, and multimodal pretrained language model. It is ......]
'text_generation_text': [Huawei is a company that has been around for a long time. ......]
```

### chat Interface

Based on the `chat` interface, the process of generating dialogue text inference tasks involves adding chat templates through the provided tokenizer to infer user queries. You can refer to the following implementation methods, and specific parameter descriptions can be viewed [chat interface API documentation](https://www.mindspore.cn/mindformers/docs/en/r1.5.0/generation/mindformers.generation.GenerationMixin.html#mindformers.generation.GenerationMixin.chat).

```python
from mindformers import build_context
from mindformers import AutoModel, AutoTokenizer

# Construct the input content.
query = "Hello!"

# Initialize the environment.
build_context({'context': {'mode': 0}, 'run_mode': 'predict', 'parallel': {}, 'parallel_config': {}})

# Instantiate a tokenizer.
tokenizer = AutoTokenizer.from_pretrained('llama2_7b')

# Instantiate a model.
# Modify the path to the local weight path.
model = AutoModel.from_pretrained('llama2_7b', checkpoint_name_or_path="path/to/llama2_7b.ckpt", use_past=True)
# Model instantiation is also supported from modelers.cn.Given repo id which format is MindSpore-Lab/model_name
# model = AutoModel.from_pretrained('MindSpore-Lab/qwen1_5_7b-chat')

# Start a stream inference task with chat.
response, history = model.chat(tokenizer=tokenizer, query=query, max_length=32)
print(response)
```

Save the example to `chat_inference.py`, modify the path for loading the weight, and run the `chat_inference.py` script.

```shell
python chat_inference.py
```

The inference result is as follows:

```text
Thanks, sir.
```

## More Information

For more inference examples of different models, see [the models supported by MindSpore Transformers](https://www.mindspore.cn/mindformers/docs/en/r1.5.0/start/models.html).
