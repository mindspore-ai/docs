# Inference

[![View Source On Gitee](https://mindspore-website.obs.cn-north-4.myhuaweicloud.com/website-images/master/resource/_static/logo_source_en.svg)](https://gitee.com/mindspore/docs/blob/master/docs/mindformers/docs/source_en/usage/inference.md)

## Overview

MindFormers provides the foundation model inference capability. You can write a script to call the high-level `pipeline` API or run the unified script `run_mindformer` to start inference. In inference mode, you can easily set and execute model inference tasks using the `pipeline` API. The `pipeline` API simplifies the overall process from data preparation to model inference. The modular design allows users to define each phase of data processing and inference through configuration files or APIs. In addition, users can customize data processing logic and inference policies based on requirements. If the unified script `run_mindformer` is used, you can directly start the system through the configuration file without writing code.

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
  Download the weight of the corresponding model from the HuggingFace model library and convert the model to the CKPT format by referring to [Weight Conversion](https://www.mindspore.cn/mindformers/docs/en/dev/function/weight_conversion.html).

3. **Executing inference tasks:**
  Call the `pipeline` API or use the unified script `run_mindformer` to execute inference tasks.

## Inference Based on the Pipeline API

An inference task process can be generated based on the customized text of the `pipeline` API. Single-device inference and multi-device inference are supported. For details about how to use the `pipeline` API to start a task and output the result, see the following implementation. For details about the parameters, see [the pipeline API document](https://gitee.com/mindspore/mindformers/blob/dev/docs/api/api_python/mindformers.pipeline.rst).

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

## Inference Based on the run_mindformer Script

For single-device inference, you can directly run [run_mindformer.py](https://gitee.com/mindspore/mindformers/blob/dev/run_mindformer.py). For multi-device inference, you need to run [scripts/msrun_launcher.sh](https://gitee.com/mindspore/mindformers/blob/dev/scripts/msrun_launcher.sh). Take Llama2 as an example. You are advised to configure the [predict_llama2_7b.yaml](https://gitee.com/mindspore/mindformers/blob/dev/configs/llama2/predict_llama2_7b.yaml) file. During inference, the vocabulary file `tokenizer.model` required for the Llama2 model will be automatically downloaded (ensuring smooth network connectivity). If the file exists locally, you can place it in the `./checkpoint_dewnload/Llama2/` directory in advance.

## Single-Device Inference

```shell
python run_mindformer.py \
--config configs/llama2/predict_llama2_7b.yaml \
--run_mode predict \
--use_parallel False \
--load_checkpoint path/to/checkpoint.ckpt \
--predict_data 'I love Beijing, because'
```

## Multi-Device Inference

Executing the script will start the multi card process, and the logs will be redirected to the `./output/msrun_log` directory. Please check the log files in it. When the inference result is printed, it proves that the inference is successful.

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

## Multi-Device Multi-Batch Inference

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

## More Information

For more inference examples of different models, see [the models supported by MindFormers](https://www.mindspore.cn/mindformers/docs/en/dev/start/models.html).
