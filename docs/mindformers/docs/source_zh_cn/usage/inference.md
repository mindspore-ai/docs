# 推理

[![查看源文件](https://mindspore-website.obs.cn-north-4.myhuaweicloud.com/website-images/master/resource/_static/logo_source.svg)](https://gitee.com/mindspore/docs/blob/master/docs/mindformers/docs/source_zh_cn/usage/inference.md)

## 概述

MindFormers 提供了大模型推理能力，用户可以执行 `run_mindformer` 统一脚本，或者编写代码调用 `pipeline` 高阶接口进行推理。使用 `run_mindformer` 统一脚本可以不编写代码，直接通过配置文件启动，用法更便捷。

## 基本流程

推理流程可以分解成以下几个步骤：

### 1. 选择推理的模型

根据需要的推理任务，选择不同的模型，如文本生成可以选择 Llama2 等。

### 2. 准备模型权重

模型权重可分为完整权重和分布式权重两种，使用时需参考以下说明。

#### 完整权重

完整权重可以通过以下两种方式获得：

1. 从HuggingFace模型库中下载相应模型的开源权重后，参考[权重格式转换](https://www.mindspore.cn/mindformers/docs/zh-CN/dev/function/weight_conversion.html)将其转换为ckpt格式。
2. 预训练或者微调后的分布式权重，通过[合并](https://www.mindspore.cn/mindformers/docs/zh-CN/dev/function/transform_weight.html)生成一个完整权重。

#### 分布式权重

分布式权重一般通过预训练或者微调后获得，默认保存在`./output/checkpoint_network`目录，需要先转换为单卡或多卡权重，再进行单卡或多卡推理。

如果推理使用的权重切分方式，与推理任务中提供的模型切分方式不同，例如以下这几种情况，则需要额外对权重进行切分方式的转换，以匹配实际推理任务中模型的切分方式。

- 多卡训练得到的权重在单卡上推理；
- 8卡训练的权重在2卡上推理；
- 已经切分好的分布式权重在单卡上推理等。

下文的命令示例均采用了在线自动切分的方式，通过设置参数 `--auto_trans_ckpt` 为 `True` 和 `--src_strategy_path_or_dir` 为权重的切分策略文件或目录路径（预训练或者微调后，默认保存在`./output/strategy`下）在推理任务中自动完成切分。更多用法可参考[分布式权重的合并和切分](https://www.mindspore.cn/mindformers/docs/zh-CN/dev/function/transform_weight.html)。

> 由于训练和推理任务都使用 `./output` 作为默认输出路径，当使用训练任务所输出的策略文件，作为推理任务的源权重策略文件时，需要将默认输出路径下的策略文件目录移动到其他位置，避免被推理任务的进程清空，如：
>
> ```mv ./output/strategy/ ./strategy```

### 3. 执行推理任务

使用 `run_mindformer` 统一脚本或调用 `pipeline` 接口执行推理任务。

## 基于 run_mindformer 脚本推理

单卡推理可以直接执行[run_mindformer.py](https://gitee.com/mindspore/mindformers/blob/dev/run_mindformer.py)脚本，多卡推理需要借助[scripts/msrun_launcher.sh](https://gitee.com/mindspore/mindformers/blob/dev/scripts/msrun_launcher.sh)来启动。

run_mindformer.py的参数说明如下：

|参数|参数说明|
|:---------------------------------|:-------------------------------------------------------------------------|
|config|yaml配置文件的路径|
|run_mode|运行的模式，推理设置为predict|
|use_parallel|是否使用多卡推理|
|load_checkpoint|加载的权重路径|
|predict_data|推理的输入数据，多batch推理时需要传输入数据的txt文件路径，包含多行输入|
|auto_trans_ckpt|自动权重切分，默认值为False|
|src_strategy_path_or_dir|权重的策略文件路径|
|predict_batch_size|多batch推理的batch_size大小|

msrun_launcher.sh包括run_mindformer.py命令和推理卡数两个参数。

下面将以 Llama2 为例介绍单卡和多卡推理的用法，推荐配置为[predict_llama2_7b.yaml](https://gitee.com/mindspore/mindformers/blob/dev/configs/llama2/predict_llama2_7b.yaml)文件。

> 推理时会自动下载Llama2模型所需的词表文件 `tokenizer.model` （需要保障网络畅通）。如果本地有这个文件，可以提前把它放在 `./checkpoint_download/llama2/` 目录下。

### 单卡推理

当使用完整权重推理时，执行以下命令即可启动推理任务：

```shell
python run_mindformer.py \
--config configs/llama2/predict_llama2_7b.yaml \
--run_mode predict \
--use_parallel False \
--load_checkpoint path/to/checkpoint.ckpt \
--predict_data 'I love Beijing, because'
```

当使用分布式权重推理时，需要增加 ``--auto_trans_ckpt`` 和 ``--src_strategy_path_or_dir`` 的入参，启动命令如下：

```shell
python run_mindformer.py \
--config configs/llama2/predict_llama2_7b.yaml \
--run_mode predict \
--use_parallel False \
--auto_trans_ckpt True \
--src_strategy_path_or_dir ./strategy
--load_checkpoint path/to/checkpoint_dir \
--predict_data 'I love Beijing, because'
```

出现如下结果，证明推理成功。推理结果也会保存到当前目录下的 `text_generation_result.txt` 文件中。详细日志可通过`./output/msrun_log` 目录查看。

```text
'text_generation_text': [I love Beijing, because it is a city that is constantly constantly changing. I have been living here for ......]
```

### 多卡推理

多卡推理的配置要求与单卡存在差异，需参考如下说明修改[predict_llama2_7b.yaml](https://gitee.com/mindspore/mindformers/blob/dev/configs/llama2/predict_llama2_7b.yaml)配置。

1. 模型并行model_parallel的配置和使用的卡数需保持一致，下文用例为2卡推理，需将model_parallel设置成2；
2. 当前版本的多卡推理不支持数据并行，需将data_parallel设置为1。

**修改前的配置：**

```yaml

parallel_config:
  data_parallel: 8
  model_parallel: 1
  pipeline_stage: 1
```

**修改后的配置：**

```yaml
parallel_config:
  data_parallel: 1
  model_parallel: 2
  pipeline_stage: 1
```

当使用完整权重推理时，需要开启在线切分方式加载权重，参考以下命令：

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

当使用分布式权重推理，且权重的切分策略与模型的切分策略一致时，参考以下命令：

```shell
bash scripts/msrun_launcher.sh "python run_mindformer.py \
--config configs/llama2/predict_llama2_7b.yaml \
--run_mode predict \
--use_parallel True \
--load_checkpoint path/to/checkpoint_dir \
--predict_data 'I love Beijing, because'" \
2
```

当使用分布式权重推理，且权重的切分策略与模型的切分策略不一致时，需要打开在线切分功能加载权重，参考以下命令：

```shell
bash scripts/msrun_launcher.sh "python run_mindformer.py \
--config configs/llama2/predict_llama2_7b.yaml \
--run_mode predict \
--use_parallel True \
--auto_trans_ckpt True \
--src_strategy_path_or_dir ./strategy
--load_checkpoint path/to/checkpoint_dir \
--predict_data 'I love Beijing, because'" \
2
```

推理结果查看方式，与单卡推理相同。

### 多卡多batch推理

多卡多batch推理的启动方式可参考上述[多卡推理](#多卡推理)，但是需要增加`predict_batch_size`的入参，并修改`predict_data`的入参。

`input_predict_data.txt`文件的内容和格式是每一行都是一个输入，问题的个数与`predict_batch_size`一致，可以参考以下格式：

```txt
I love Beijing, because
I love Beijing, because
I love Beijing, because
I love Beijing, because
```

以完整权重推理为例，可以参考以下命令启动推理任务：

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

推理结果查看方式，与单卡推理相同。

## 基于 pipeline 接口推理

基于 `pipeline` 接口的自定义文本生成推理任务流程，支持单卡推理和多卡推理。关于如何使用 `pipeline` 接口启动任务并输出结果，可以参考以下实现方式，具体参数说明可以查看 [pipeline 接口的API文档](https://www.mindspore.cn/mindformers/docs/zh-CN/dev/mindformers/mindformers.pipeline.html#mindformers.pipeline)。

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

## 更多信息

更多关于不同模型的推理示例，请访问[MindFormers 已支持模型库](https://www.mindspore.cn/mindformers/docs/zh-CN/dev/start/models.html)。
