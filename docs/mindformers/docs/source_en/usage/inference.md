# Inference

[![View Source On Gitee](https://mindspore-website.obs.cn-north-4.myhuaweicloud.com/website-images/r2.4.10/resource/_static/logo_source_en.svg)](https://gitee.com/mindspore/docs/blob/r2.4.10/docs/mindformers/docs/source_en/usage/inference.md)

## Overview

MindFormers provides the foundation model inference capability. Users can run the unified script `run_mindformer` or write a script to call the high-level `pipeline` API to start inference. If the unified script `run_mindformer` is used, you can directly start the system through the configuration file without writing code. The `pipeline` interface helps users to easily set up and perform model inference tasks.

The following table lists the features supported by MindFormers text generation and inference.

|Feature|Concept|Function|
|:------------|:---------------------------------------|:-----------------------------------------------------|
|[Incremental inference](#incremental-inference)|Incremental inference indicates that the model can generate text step by step instead of generating all content at a time.|You can accelerate the text generation speed when the `text_generator` method is called to generate autoregressive text. use_past is set to True in the YAML file by default to enable incremental inference.|
|[Batch inference](#multi-device-multi-batch-inference)|Batch inference is a method of processing multiple input samples at the same time.|You can input multiple samples to perform inference in batches. When the computing power of a single batch is insufficient, multi-batch inference can improve the inference throughput.|
|[Stream inference](#stream-inference)|Stream inference is a processing method that allows a model to start to output a result after receiving a part of an input, instead of waiting for the entire input sequence to be completely received.|With the Streamer class provided, when the `text_generator` method is called to generate text, you can view each generated word in real time without waiting for all results to be generated.|
|[Distributed inference](#multi-device-inference)|Distributed inference is a method of distributing computing tasks on multiple compute nodes for execution.|For models that cannot be deployed on a single device, you need to split the models using the multi-device distributed model before inference.|

## Procedure

Based on actual operations, the inference process can be divided into the following steps:

1. **Selecting a model to be inferred:**
  Select a model based on the required inference task. For example, select Llama2 for text generation.

2. **Preparing the model weight:**
  There are two ways to obtain weights. One is to use open-source weights, which can be downloaded from the HuggingFace model library for the corresponding model and then converted to the ckpt format by referring to the [Weight Format Conversion](https://www.mindspore.cn/mindformers/docs/en/r1.3.2/function/weight_conversion.html) document. The other is to use distributed weights after pre-training or fine-tuning. The obtained distributed weights (saved by default in `./output/checkpoint_network`)  are converted into single or multi-card weights, then perform single or multi-card inference. Guidance on this based on the command line configuration approach has been provided below, and more usage can be found in the [Merging and Splitting of Distributed Weights](https://www.mindspore.cn/mindformers/docs/en/r1.3.2/function/transform_weight.html) document.

3. **Executing inference tasks:**
  Call the `pipeline` API or use the unified script `run_mindformer` to execute inference tasks.

## Inference Based on the run_mindformer Script

For single-device inference, you can directly run [run_mindformer.py](https://gitee.com/mindspore/mindformers/blob/dev/run_mindformer.py). For multi-device inference, you need to run [scripts/msrun_launcher.sh](https://gitee.com/mindspore/mindformers/blob/dev/scripts/msrun_launcher.sh). Take Llama2 as an example. You are advised to configure the [predict_llama2_7b.yaml](https://gitee.com/mindspore/mindformers/blob/dev/configs/llama2/predict_llama2_7b.yaml) file.

> During inference, the vocabulary file `tokenizer.model` required for the Llama2 model will be automatically downloaded (ensuring smooth network connectivity). If the file exists locally, you can place it in the `./checkpoint_download/Llama2/` directory in advance.

Note: If the inference uses weights that are sliced differently from the way the model is sliced in the inference task, such as in the following cases:

- The weights obtained from multi-card training are reasoned on a single card;
- The weights of the eight-card training are reasoned over two cards;
- Already sliced distributed weights are reasoned on a single card, and so on.

The weights need to be additionally transformed in terms of the way they are sliced to match the way the model is sliced in the actual inference task. It is recommended to use online autoslicing by setting the command parameters `--auto_trans_ckpt` to `-True` and `-src_strategy_path_or_dir` to the weighted slicing strategy file or directory path (which is saved by default after training under `./output/strategy`) are automatically sliced in the inference task. Details can be found in [Distributed Weight Slicing and Merging](https://www.mindspore.cn/mindformers/docs/en/dev/function/transform_weight.html).

## Single-Device Inference

The startup of single-card inference is relatively simple. You just need to execute the following command to start the inference task:

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
--src_strategy_path_or_dir ./output/strategy
--load_checkpoint path/to/checkpoint.ckpt \
--predict_data 'I love Beijing, because'
```

## Multi-Device Inference

In addition to the startup mode dependent on the `msrun_launcher.sh` script, there are also two places to pay attention to, one is the parallel configuration, the other is the weight loading method.

The current version of inference mode only supports model parallelism. The original [predict_llama2_7b.yaml](https://gitee.com/mindspore/mindformers/blob/dev/configs/llama2/predict_llama2_7b.yaml) parallel configuration needs to be modified before running the command:

```yaml
# Configuration before modification
parallel_config:
  data_parallel: 8
  model_parallel: 1
  pipeline_stage: 1
```

```yaml
# The modified configuration
parallel_config:
  data_parallel: 1
  model_parallel: 2
  pipeline_stage: 1
```

> model_parallel configuration is consistent with the number of cards used, and the parallel configuration used by the weight offline segmentation generation policy file needs to be consistent with the parallel configuration of the actual inference task. The current use case model_parallel is set to 2.

When full weight inference is used, you need to enable the online segmentation mode to load weights. For details, see the following command:

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
--src_strategy_path_or_dir ./output/strategy
--load_checkpoint path/to/checkpoint_dir \
--predict_data 'I love Beijing, because'" \
2
```

Executing the script will start the multi card process, and the logs will be redirected to the `./output/msrun_log` directory. When the `text_generation_result.txt` file appears in the current directory, it proves successful inference. If the file does not appear, you can view the log file.

## Multi-Device Multi-Batch Inference

Multi-card multi-batch inference is initiated in the same way as multi-card inference, but requires the addition of the `predict_batch_size` inputs and the modification of the `predict_data` inputs.

The content and format of the `input_predict_data.txt` file is an input each line, and the number of questions is the same as the `predict_batch_size`, which can be found in the following format:

```txt
I love Beijing, because
I love Beijing, because
I love Beijing, because
I love Beijing, because
```

Refer to the following commands to perform inference tasks:

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

The following table describes the input parameters for script execution.

|Parameter|Description|
|:---------------------------------|:-------------------------------------------------------------------------|
|config|Path of the YAML file.|
|run_mode|Running mode. Set it to **predict** for inference.|
|predict_batch_size|Size of inferences in batches.|
|use_parallel|Specifies whether to use the multi-device inference.|
|auto_trans_ckpt|For multi-device inference, set this parameter to **True**, indicating automatic weight segmentation. The default value is **False**.|
|load_checkpoint|Loaded weight path.|
|predict_data|Input data for inference. For multi-batch inference, the path of the TXT file containing the input data needs to be specified.|
|2|In the multi-device inference command, **2** indicates the number of devices used for inference.|

The results of running the preceding single-device and multi-device inference commands are as follows:

```text
'text_generation_text': [I love Beijing, because it is a city that is constantly constantly changing. I have been living here for ......]
```

## Inference Based on Pipeline Interface

Customized text generation inference task flow based on `pipeline` interface, supporting single card inference and multi-card inference. About how to use `pipeline` interface to start the task and output the result, you can refer to the following implementation. The specific parameter description can be viewed [pipeline interface API documentation](https://www.mindspore.cn/mindformers/docs/en/dev/mindformers/mindformers.pipeline.html#mindformers.pipeline).

### Incremental Inference

```python
from mindformers import build_context
from mindformers import AutoModel, AutoTokenizer, pipeline, TextStreamer

# Construct the input content.
inputs = ["I love Beijing, because", "LLaMA is a", "Huawei is a company that"]

# Initialize the environment.
build_context({'context': {'mode': 0}, 'parallel': {}, 'parallel_config': {}})

# Instantiate a tokenizer.
tokenizer = AutoTokenizer.from_pretrained('llama2_7b')

# Instantiate a model.
# Modify the path to the local weight path.
model = AutoModel.from_pretrained('llama2_7b', checkpoint_name_or_path="path/to/llama2_7b.ckpt", use_past=True)

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

### Stream Inference

```python
from mindformers import build_context
from mindformers import AutoModel, AutoTokenizer, pipeline, TextStreamer

# Construct the input content.
inputs = ["I love Beijing, because", "LLaMA is a", "Huawei is a company that"]

# Initialize the environment.
build_context({'context': {'mode': 0}, 'parallel': {}, 'parallel_config': {}})

# Instantiate a tokenizer.
tokenizer = AutoTokenizer.from_pretrained('llama2_7b')

# Instantiate a model.
# Modify the path to the local weight path.
model = AutoModel.from_pretrained('llama2_7b', checkpoint_name_or_path="path/to/llama2_7b.ckpt", use_past=True)

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

## More Information

For more inference examples of different models, see [the models supported by MindFormers](https://www.mindspore.cn/mindformers/docs/en/r1.3.2/start/models.html).
