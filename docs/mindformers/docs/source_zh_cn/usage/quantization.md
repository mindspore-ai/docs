# 量化

[![查看源文件](https://mindspore-website.obs.cn-north-4.myhuaweicloud.com/website-images/r2.4.10/resource/_static/logo_source.svg)](https://gitee.com/mindspore/docs/blob/r2.4.10/docs/mindformers/docs/source_zh_cn/usage/quantization.md)

## 概述

量化（Quantization）作为一种重要的大模型压缩技术，通过对模型中的浮点参数转为低精度的整数参数，实现对参数的压缩。随着模型的参数和规格不断增大，量化在模型部署中能有效减少模型存储空间和加载时间，提高模型的推理性能。

MindFormers 集成 MindSpore Golden Stick 工具组件，提供统一量化推理流程，方便用户开箱即用。

## 配套安装

使用量化推理功能前请安装MindSpore Golden Stick，参考[安装指南](https://gitee.com/mindspore/golden-stick#%E5%AE%89%E8%A3%85)

下载源码，下载后进入`golden_stick`目录。

```bash
bash build.sh
pip install output/mindspore_gs-0.6.0-py3-none-any.whl
```

执行以下命令，验证安装结果。

```bash
pip show mindspore_gs

# Name: mindspore_gs
# Version: 0.6.0
# Summary: A MindSpore model optimization algorithm set..
# Home-page: https://www.mindspore.cn
# Author: The MindSpore Authors
# Author-email: contact@mindspore.cn
# License: Apache 2.0
```

## 量化基本流程

结合实际操作，可以将量化分解为以下步骤：

1. **选择模型：**
   选择一个语言模型，当前支持量化的模型为Llama2_13B和Llama2_70B。

2. **下载模型权重：**
   从 HuggingFace 模型库中下载相应模型的权重，参考[权重格式转换](https://www.mindspore.cn/mindformers/docs/zh-CN/r1.3.2/function/weight_conversion.html)文档转换为ckpt格式。

3. **量化模型权重转换：**
   运行mindspore_gs库中的转换脚本`quant_ckpt.py`，将步骤2中的原始权重转换为量化权重。

4. **量化配置文件准备：**
   使用mindformers内置的与模型配套的量化推理配置文件，其中量化相关的配置项为`model.model_config.quantization_config`

   以`llama2_13b_rtn`量化模型为例，默认量化配置如下

   ```yaml
     quantization_config:
       quant_method: 'rtn'
       weight_dtype: 'int8'
       activation_dtype: None
       kvcache_dtype: None
       outliers_suppression: None
       modules_to_not_convert: ['lm_head']
       algorithm_args: {}
   ```

   | 参数                   | 属性 | 功能描述                                                             | 参数类型  | 取值范围             |
   | ---------------------- | ---- |:-----------------------------------------------------------------| --------- |------------------|
   | quant_method           | 必选 | 支持的量化算法，目前只支持RTN/Smooth_Quant/PTQ算法                              | str       | rtn/smooth_quant/ptq |
   | weight_dtype           | 必选 | 量化的weight类型，目前只支持int8                                            | str       | int8/None           |
   | activation_dtype       | 必选 | 参数的激活类型，None表示维持网络原计算类型(compute_dtype)不变                         | str       | int8/None        |
   | kvcache_dtype          | 可选 | KVCache量化类型，None和不配置表示维持原KVCache数据类型不变                           | str       | int8/None        |
   | outliers_suppression   | 可选 | 异常值抑制使用的算法类型，目前仅支持smooth平滑抑制                                     | str       | smooth/None        |
   | modules_to_not_convert | 必选 | 配置不进行量化的层                                                        | List[str] | /                |
   | algorithm_args         | 必选 | 对接MindSpore Golden Stick不同的算法类型配置，例如：smooth_quant算法需要配置alpha=0.5 | Dict      | /                |

5. **执行推理任务：**
   基于`generate`接口实现推理脚本，执行脚本即可得到推理结果。

## 基于Llama2_13B模型使用RTN量化算法进行A16W8量化推理实践

### 选择模型

该实践流程选择Llama2-13B模型进行单卡量化推理。

本实践使用`AutoModel.from_pretrained()`通过传参模型配置/权重路径来实例化模型，预先创建存放目录。

```shell
mkdir /data/tutorial/llama2_13b_rtn_a16w8_dir
```

> 注：当前AutoModel.from_pretrained()接口暂不支持通过量化模型名称传参来实例化

单卡目录结构

```shell
llama2_13b_rtn_a16w8_dir
  ├── predict_llama2_13b_rtn.yaml
  └── llama2_13b_rtn_a16w8.ckpt
```

### 下载模型权重

MindFormers提供已经转换完成的预训练权重、词表文件用于预训练、微调和推理，用户也可以下载HuggingFace官方权重经过[模型权重转换](#模型权重转换)后进行使用。

词表下载链接：[tokenizer.model](https://ascend-repo-modelzoo.obs.cn-east-2.myhuaweicloud.com/MindFormers/llama2/tokenizer.model)

| 模型名称            |                                                MindSpore权重                                                 |                      HuggingFace权重                       |
|:----------------|:----------------------------------------------------------------------------------------------------------:|:--------------------------------------------------------:|
| llama2-13b      | [llama2-13b-fp16.ckpt](https://ascend-repo-modelzoo.obs.cn-east-2.myhuaweicloud.com/MindFormers/llama2/llama2-13b-fp16.ckpt) | [Llama-2-13b-hf](https://huggingface.co/meta-llama/Llama-2-13b-hf) |

> 注：Llama2的所有权重都需要通过向Meta[提交申请](https://ai.meta.com/resources/models-and-libraries/llama-downloads)来获取，如有需要请自行申请。

### 模型权重转换

进入mindspore_gs库根目录`golden-stick`，执行量化权重转换脚本

```bash
python example/ptq/quant_ckpt.py -c /path/to/predict_llama2_13b.yaml -s /path/to/boolq/dev.jsonl -t boolq -q rtn-a16w8 > log_rtn_a16w8_quant 2>&
```

其中`predict_llama2_13b.yaml`中的`load_checkpoint`配置为上一步下载的原始权重存放路径。

转换过程中的检验数据集使用`boolq`，下载参考[boolq数据集链接](https://github.com/svinkapeppa/boolq)。下载完成后，在上述脚本中传入`dev.jsonl`存放路径。

执行脚本，将生成的量化权重文件拷贝至`llama2_13b_rtn_a16w8_dir`目录中。

```shell
cp output/rtn-a16w8_ckpt/rank_0/rtn-a16w8.ckpt /data/tutorial/llama2_13b_rtn_a16w8_dir/llama2_13b_rtn_a16w8.ckpt
```

### 量化配置文件准备

MindFormers已提供[predict_llama2_13b_rtn.yaml配置文件](https://gitee.com/mindspore/mindformers/blob/r1.3.0/configs/llama2/predict_llama2_13b_rtn.yaml)，将其拷贝至`llama2_13b_rtn_a16w8_dir`目录中。

```shell
cp configs/llama2/predict_llama2_13b_rtn.yaml /data/tutorial/llama2_13b_rtn_a16w8_dir
```

### 执行推理任务

1. **脚本实例**

   替换MindFormers下的[run_llama2_generate.py](https://gitee.com/mindspore/mindformers/blob/r1.3.0/scripts/examples/llama2/run_llama2_generate.py)脚本为以下代码。

   此实践基于`AutoModel.from_pretrained()`接口实例化量化模型，需调整该接口内的参数为之前创建的目录路径。

   通过调用`generate`接口获取推理结果。具体参数说明可参考[AutoModel](https://www.mindspore.cn/mindformers/docs/zh-CN/r1.3.2/mindformers/mindformers.AutoModel.html#mindformers.AutoModel)和[generate](https://www.mindspore.cn/mindformers/docs/zh-CN/r1.3.2/generation/mindformers.generation.GenerationMixin.html#mindformers.generation.GenerationMixin.generate)接口文档。

   ```python
   """llama2 predict example."""
   import argparse
   import os

   import mindspore as ms
   from mindspore import Tensor, Model
   from mindspore.common import initializer as init

   from mindformers import AutoModel
   from mindformers import MindFormerConfig, logger
   from mindformers.core.context import build_context
   from mindformers.core.parallel_config import build_parallel_config
   from mindformers.models.llama import LlamaTokenizer
   from mindformers.trainer.utils import transform_and_load_checkpoint


   def main(config_path, use_parallel, load_checkpoint):
       # 构造输入
       inputs = ["I love Beijing, because",
                 "LLaMA is a",
                 "Huawei is a company that"]
       batch_size = len(inputs)

       # 根据yaml文件生成模型配置
       config = MindFormerConfig(config_path)
       config.use_parallel = use_parallel
       device_num = os.getenv('MS_WORKER_NUM')
       logger.info(f"Use device number: {device_num}, it will override config.model_parallel.")
       config.parallel_config.model_parallel = int(device_num) if device_num else 1
       config.parallel_config.data_parallel = 1
       config.parallel_config.pipeline_stage = 1
       config.load_checkpoint = load_checkpoint

       # 初始化环境
       build_context(config)
       build_parallel_config(config)
       model_name = config.trainer.model_name

       # 实例化tokenizer
       tokenizer = LlamaTokenizer.from_pretrained(model_name)
       # 实例化模型
       network = AutoModel.from_pretrained("/data/tutorial/llama2_13b_rtn_a16w8_dir",
                                           download_checkpoint=False)
       model = Model(network)

       # 加载权重
       if config.load_checkpoint:
           logger.info("----------------Transform and load checkpoint----------------")
           seq_length = config.model.model_config.seq_length
           input_ids = Tensor(shape=(batch_size, seq_length), dtype=ms.int32, init=init.One())
           infer_data = network.prepare_inputs_for_predict_layout(input_ids)
           transform_and_load_checkpoint(config, model, network, infer_data, do_predict=True)

       inputs_ids = tokenizer(inputs, max_length=config.model.model_config.seq_length, padding="max_length")["input_ids"]

       outputs = network.generate(inputs_ids,
                                  max_length=config.model.model_config.max_decode_length,
                                  do_sample=config.model.model_config.do_sample,
                                  top_k=config.model.model_config.top_k,
                                  top_p=config.model.model_config.top_p)
       for output in outputs:
           print(tokenizer.decode(output))


   if __name__ == "__main__":
       parser = argparse.ArgumentParser()
       parser.add_argument('--config_path', default='predict_llama2_7b.yaml', type=str,
                           help='model config file path.')
       parser.add_argument('--use_parallel', action='store_true',
                           help='if run model prediction in parallel mode.')
       parser.add_argument('--load_checkpoint', type=str,
                           help='load model checkpoint path or directory.')

       args = parser.parse_args()
       main(
           args.config_path,
           args.use_parallel,
           args.load_checkpoint
       )
   ```

2. **执行脚本启动命令**

   MindFormers提供`Llama2`模型的快速推理脚本，支持单卡、多卡以及多batch推理。

   ```shell
   # 脚本使用
   bash scripts/examples/llama2/run_llama2_predict.sh PARALLEL CONFIG_PATH CKPT_PATH DEVICE_NUM
   # 参数说明
   PARALLEL:    是否使用多卡推理, 'single'表示单卡推理, 'parallel'表示多卡推理
   CONFIG_PATH: 模型配置文件路径
   CKPT_PATH:   模型权重文件路径
   DEVICE_NUM:  使用卡数, 仅开启多卡推理时生效
   ```

   单卡推理

   ```bash
   bash scripts/examples/llama2/run_llama2_predict.sh single /data/tutorial/llama2_13b_w8a16_dir/predict_llama2_13b_w8a16.yaml /data/tutorial/llama2_13b_w8a16_dir/llama2_13b_w8a16.ckpt
   ```

   执行以上命令的推理结果如下：

   ```text
   'text_generation_text': [I love Beijing, because it is a city that is constantly constantly changing. I have been living here for ......]
   'text_generation_text': [LLaMA is a large-scale, open-source, multimodal, multilingual, multitask, and multimodal pretrained language model. It is ......]
   'text_generation_text': [Huawei is a company that has been around for a long time. ......]
   ```