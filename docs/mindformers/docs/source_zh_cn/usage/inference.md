# 推理

[![查看源文件](https://mindspore-website.obs.cn-north-4.myhuaweicloud.com/website-images/r2.4.10/resource/_static/logo_source.svg)](https://gitee.com/mindspore/docs/blob/r2.4.10/docs/mindformers/docs/source_zh_cn/usage/inference.md)

## 概述

MindFormers 提供了大模型推理能力，用户可以编写脚本调用 `pipeline` 高阶接口或者执行 `run_mindformer` 统一脚本启动推理。在推理模式下， `pipeline` 接口可以帮助用户轻松地设置和执行模型推理任务。 `pipeline` 接口简化了从数据准备到模型推理的整体流程。模块化设计允许用户通过配置文件或编程接口定义数据处理和推理的各个阶段，并且用户可以根据自己的需求定制数据处理逻辑和推理策略。使用 `run_mindformer` 统一脚本则可以不编写代码而直接通过配置文件启动。

目前 MindFormers 文本生成推理支持的特性如下表：

| 特性                     |概念| 作用                                                                            |
|:-----------------------|:---------------------------------------|:------------------------------------------------------------------------------|
| [增量推理](#增量推理)          |增量推理指的是模型能够逐步生成文本，而不是一次性生成全部内容| 加快用户在调用 `text_generator` 方法进行自回归文本生成时的文本生成速度，在yaml文件中默认配置use_past=True，开启增量推理 |
| [Batch推理](#多卡多batch推理) |batch推理是指同时处理多个输入样本的一种方式| 支持同时多个样本输入进行batch推理，在单batch推理算力不足的情况下，多 batch 推理能够提升推理时的吞吐率                   |
| [流式推理](#流式推理)          |流式推理是一种处理方式，它允许模型在接收到输入的一部分后就开始输出结果，而不是等待整个输入序列完全接收完毕| 提供 Streamer 类，用户在调用 `text_generator` 方法进行文本生成时能够实时看到生成的每一个词，而不必等待所有结果均生成结束    |
| [分布式推理](#多卡推理)         |分布式推理是指将计算任务分布在多个计算节点上执行的一种方法| 对于无法在单卡上部署的模型，需要通过多卡分布式对模型进行切分后再推理                                            |

## 基本流程

结合实际操作，推理流程可以分解成以下步骤：

1. **选择推理的模型：**
  根据需要的推理任务选择不同的模型，如文本生成可以选择 Llama2 等。

2. **准备模型权重：**
  从 HuggingFace 模型库中下载相应模型的权重，参考[权重转换](https://www.mindspore.cn/mindformers/docs/zh-CN/r1.3.2/function/weight_conversion.html)文档转换为ckpt格式。

3. **执行推理任务：**
  调用 `pipeline` 接口或使用 `run_mindformer` 统一脚本执行推理任务。

## 基于 pipeline 接口推理

基于 `pipeline` 接口的自定义文本生成推理任务流程，支持单卡推理和多卡推理。关于如何使用 `pipeline` 接口启动任务并输出结果，可以参考以下实现方式，具体参数说明可以查看 [pipeline 接口的API文档](https://www.mindspore.cn/mindformers/docs/zh-CN/r1.3.2/mindformers/mindformers.pipeline.html#mindformers.pipeline)。

### 增量推理

```python
from mindformers import build_context
from mindformers import AutoModel, AutoTokenizer, pipeline, TextStreamer

# 构造输入
inputs = ["I love Beijing, because", "LLaMA is a", "Huawei is a company that"]

# 初始化环境
build_context({'context': {'mode': 0}, 'parallel': {}, 'parallel_config': {}})

# 实例化tokenizer
tokenizer = AutoTokenizer.from_pretrained('llama2_7b')

# 模型实例化
# 修改成本地的权重路径
model = AutoModel.from_pretrained('llama2_7b', checkpoint_name_or_path="path/to/llama2_7b.ckpt", use_past=True)

# pipeline启动非流式推理任务
text_generation_pipeline = pipeline(task="text_generation", model=model, tokenizer=tokenizer)
outputs = text_generation_pipeline(inputs, max_length=512, do_sample=False, top_k=3, top_p=1)
for output in outputs:
    print(output)
```

通过将示例保存到 `pipeline_inference.py` 中，并且修改加载权重的路径，然后直接执行 `pipeline_inference.py` 脚本。

```shell
python pipeline_inference.py
```

执行以上命令的推理结果如下：

```text
'text_generation_text': [I love Beijing, because it is a city that is constantly constantly changing. I have been living here for ......]
'text_generation_text': [LLaMA is a large-scale, open-source, multimodal, multilingual, multitask, and multimodal pretrained language model. It is ......]
'text_generation_text': [Huawei is a company that has been around for a long time. ......]
```

### 流式推理

```python
from mindformers import build_context
from mindformers import AutoModel, AutoTokenizer, pipeline, TextStreamer

# 构造输入
inputs = ["I love Beijing, because", "LLaMA is a", "Huawei is a company that"]

# 初始化环境
build_context({'context': {'mode': 0}, 'parallel': {}, 'parallel_config': {}})

# 实例化tokenizer
tokenizer = AutoTokenizer.from_pretrained('llama2_7b')

# 模型实例化
# 修改成本地的权重路径
model = AutoModel.from_pretrained('llama2_7b', checkpoint_name_or_path="path/to/llama2_7b.ckpt", use_past=True)

# pipeline启动流式推理任务
streamer = TextStreamer(tokenizer)
text_generation_pipeline = pipeline(task="text_generation", model=model, tokenizer=tokenizer, streamer=streamer)
_ = text_generation_pipeline(inputs, max_length=512, do_sample=False, top_k=3, top_p=1)
```

通过将示例保存到 `pipeline_inference.py` 中，并且修改加载权重的路径，然后直接执行 `pipeline_inference.py` 脚本。

```shell
python pipeline_inference.py
```

执行以上命令的推理结果如下：

```text
'text_generation_text': [I love Beijing, because it is a city that is constantly constantly changing. I have been living here for ......]
'text_generation_text': [LLaMA is a large-scale, open-source, multimodal, multilingual, multitask, and multimodal pretrained language model. It is ......]
'text_generation_text': [Huawei is a company that has been around for a long time. ......]
```

## 基于 run_mindformer 脚本推理

单卡推理可以直接执行[run_mindformer.py](https://gitee.com/mindspore/mindformers/blob/r1.3.2/run_mindformer.py)，多卡推理需要借助 [scripts/msrun_launcher.sh](https://gitee.com/mindspore/mindformers/blob/r1.3.2/scripts/msrun_launcher.sh) 启动。以 Llama2 为例，推荐配置为[predict_llama2_7b.yaml](https://gitee.com/mindspore/mindformers/blob/r1.3.2/configs/llama2/predict_llama2_7b.yaml)文件。推理时会自动下载Llama2模型所需词表文件 `tokenizer.model` （需要保障网络畅通）。如果本地有这个文件，可以提前把它放在 `./checkpoint_download/llama2/` 目录下。

## 单卡推理

```shell
python run_mindformer.py \
--config configs/llama2/predict_llama2_7b.yaml \
--run_mode predict \
--use_parallel False \
--load_checkpoint path/to/checkpoint.ckpt \
--predict_data 'I love Beijing, because'
```

## 多卡推理

执行脚本会拉起多卡进程，日志会重定向至 `./output/msrun_log` 下。当前目录下出现 `text_generation_result.txt` 文件时，证明推理成功。若未出现该文件，可查看日志文件。

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

## 多卡多batch推理

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

脚本执行入参的说明列表：
|参数|参数说明|
|:---------------------------------|:-------------------------------------------------------------------------|
|config|yaml 配置文件的路径|
|run_mode|运行的模式，推理设置为 predict|
|predict_batch_size|batch 推理的 batch_size 大小|
|use_parallel|是否使用多卡推理|
|auto_trans_ckpt|多卡推理时需要配置为 True，自动权重切分，默认值为 False|
|load_checkpoint|加载的权重路径|
|predict_data|推理的输入数据，多 batch 推理时需要传输入数据的txt文件路径，包含多行输入|
|2|多卡推理命令中的 2 是推理时使用的卡数|

执行以上单卡推理和多卡推理命令的结果如下：

```text
'text_generation_text': [I love Beijing, because it is a city that is constantly constantly changing. I have been living here for ......]
```

## 更多信息

更多关于不同模型的推理示例，请访问[MindFormers 已支持模型库](https://www.mindspore.cn/mindformers/docs/zh-CN/r1.3.2/start/models.html)
