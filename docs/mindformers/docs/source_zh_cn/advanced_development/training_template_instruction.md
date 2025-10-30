# 训练配置模板使用说明

[![查看源文件](https://mindspore-website.obs.cn-north-4.myhuaweicloud.com/website-images/r2.7.1/resource/_static/logo_source.svg)](https://gitee.com/mindspore/docs/blob/r2.7.1/docs/mindformers/docs/source_zh_cn/advanced_development/training_template_instruction.md)

## 概述

MindSpore Transformers提供了训练的通用配置文件模板，主要有下面两种使用场景：

1. 用户开发适配的模型，可以基于模板编写训练配置。
2. 对于MindSpore Transformers现有的模型，用户希望使用当前未提供配置的特定规格模型时，可以使用配置模板，并配合Hugging Face或ModelScope的模型配置，来拉起训练任务。

MindSpore Transformers对于不同训练场景提供了对应的配置模板，如下：

进行稠密模型预训练时，请使用[llm_pretrain_dense_template.yaml](https://gitee.com/mindspore/mindformers/blob/r1.7.0/configs/general/llm_pretrain_dense_template.yaml)。

进行MOE模型预训练时，请使用[llm_pretrain_moe_template.yaml](https://gitee.com/mindspore/mindformers/blob/r1.7.0/configs/general/llm_pretrain_moe_template.yaml)。

进行稠密模型微调训练时，请使用[llm_finetune_dense_template.yaml](https://gitee.com/mindspore/mindformers/blob/r1.7.0/configs/general/llm_finetune_dense_template.yaml)。

进行MOE模型微调训练时，请使用[llm_finetune_moe_template.yaml](https://gitee.com/mindspore/mindformers/blob/r1.7.0/configs/general/llm_finetune_moe_template.yaml)。

## 使用说明

### 模块说明

模板主要涵盖以下九个功能模块配置，详细参数配置说明可以参考[配置文件说明](https://www.mindspore.cn/mindformers/docs/zh-CN/r1.7.0/feature/configuration.html)。

| 模块名称     | 模块用途                                                                                                                                                                                                                                                                                                                                                                                                       |
|----------|------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|
| 基础配置     | 基础配置主要用于指定MindSpore随机种子以及加载权重的相关设置。                                                                                                                                                                                                                                                                                                                                                                        |
| 数据集配置    | 数据集配置主要用于MindSpore模型训练时的数据集相关设置。详情可参考[数据集](https://www.mindspore.cn/mindformers/docs/zh-CN/r1.7.0/feature/dataset.html)。                                                                                                                                                                                                                                                                                   |
| 模型配置     | 不同的模型配置参数存在差异，模板中的参数为通用配置。                                                                                                                                                                                                                                                                                                                                                                                 |
| 模型优化配置   | MindSpore Transformers提供重计算相关配置，以降低模型在训练时的内存占用，详情可参考[重计算](https://www.mindspore.cn/mindformers/docs/zh-CN/r1.7.0/advanced_development/performance_optimization.html#%E9%87%8D%E8%AE%A1%E7%AE%97)。                                                                                                                                                                                                          |
| 模型训练配置   | 启动模型训练时相关参数的配置模块，模板中主要包含trainer、runner_config、runner_wrapper、学习率（lr_schedule）以及优化器（optimizer）相关训练所需模块的参数。                                                                                                                                                                                                                                                                                                  |
| 并行配置     | 为了提升模型的性能，在大规模集群的使用场景中通常需要为模型配置并行策略，详情可参考[分布式并行](https://www.mindspore.cn/mindformers/docs/zh-CN/r1.7.0/feature/parallel_training.html)。                                                                                                                                                                                                                                                                   |
| 回调函数配置   | MindSpore Transformers提供封装后的Callbacks函数类，主要实现在模型训练过程中返回模型的训练状态并输出、保存模型权重文件等操作。目前支持以下几个Callbacks函数类：<br>1.MFLossMonitor<br> 该回调函数类主要用于在训练过程中对训练进度、模型Loss、学习率等信息进行打印<br>2.SummaryMonitor<br>该回调函数类主要用于收集Summary数据，详情可参考[mindspore.SummaryCollector](https://www.mindspore.cn/docs/zh-CN/r2.7.1/api_python/mindspore/mindspore.SummaryCollector.html)。<br>3.CheckpointMonitor<br>该回调函数类主要用于在模型训练过程中保存模型权重文件。 |
| context配置 | Context配置主要用于指定[mindspore.set_context](https://www.mindspore.cn/docs/zh-CN/r2.7.1/api_python/mindspore/mindspore.set_context.html)中的相关参数。                                                                                                                                                                                                                                                                  |
| 性能分析工具配置 | MindSpore Transformers提供Profile作为模型性能调优的主要工具，详情可参考[性能调优指南](https://www.mindspore.cn/mindformers/docs/zh-CN/r1.7.0/advanced_development/performance_optimization.html)。                                                                                                                                                                                                                                     |

## 基本配置修改

使用配置模版进行训练时，修改以下基础配置即可快速启动。

配置模板默认使用8卡。

### 数据集配置修改

1. 预训练场景使用Megatron数据集，详情请参考[Megatron数据集](https://www.mindspore.cn/mindformers/docs/zh-CN/r1.7.0/feature/dataset.html#megatron%E6%95%B0%E6%8D%AE%E9%9B%86)。
2. 微调场景使用HuggingFace数据集，详情请参考[HuggingFace数据集](https://www.mindspore.cn/mindformers/docs/zh-CN/r1.7.0/feature/dataset.html#hugging-face%E6%95%B0%E6%8D%AE%E9%9B%86)。

### 模型配置修改

1. 修改模型配置时可以选择下载Huggingface模型后直接修改yaml配置中的pretrained_model_dir来读取模型配置（该功能暂不支持预训练），模型训练时会自动生成tokenizer和model_config，支持模型列表：

   | 模型名称     |
   |----------|
   | Deepseek3 |
   | Qwen3    |
   | Qwen2_5  |

2. 生成的模型配置优先以yaml配置为准，未配置参数则取值pretrained_model_dir路径下的config.json中的参数。如若要修改定制模型配置，则只需要在model_config中添加相关配置即可。
3. 通用配置详情请参考[模型配置](https://www.mindspore.cn/mindformers/docs/zh-CN/r1.7.0/feature/configuration.html#legacy-%E6%A8%A1%E5%9E%8B%E9%85%8D%E7%BD%AE)。

## 进阶配置修改

可以进一步按照下述方式进行修改，以自定义训练。

### 基础配置修改

进行预训练时，可通过load_ckpt_format来修改生成的权重格式，支持safetensors和ckpt，推荐使用safetensors。可通过output_dir来指定训练过程中日志、权重和策略文件的生成路径。

### 训练超参修改

1. recompute_config（重计算）、optimizer（优化器）、lr_schedule（学习率）相关配置修改会影响模型训练结果的精度。
2. 如果在训练过程中出现内存不足而导致模型无法开启训练，可考虑开启重计算从而降低模型在训练时的内存占用。
3. 通过修改学习率配置来达到模型训练时的学习效果。
4. 修改优化器配置能够修改计算模型训练时的梯度。
5. parallel（模型并行）、context相关配置会影响模型训练时的性能。
6. 模型训练时通过开启use_parallel=True来提升训练时的性能，通过调试配置并行策略达到预期的性能效果。详细参数配置请参考[并行配置](https://www.mindspore.cn/mindformers/docs/zh-CN/r1.7.0/feature/configuration.html#%E5%B9%B6%E8%A1%8C%E9%85%8D%E7%BD%AE)。
7. 具体配置详情参考[模型训练配置](https://www.mindspore.cn/mindformers/docs/zh-CN/r1.7.0/feature/configuration.html#%E6%A8%A1%E5%9E%8B%E8%AE%AD%E7%BB%83%E9%85%8D%E7%BD%AE)。

### 回调函数配置修改

1. 模板提供了保存权重相关的回调函数：save_checkpoint_steps可修改权重的保存步数间隔；keep_checkpoint_max可设定最大权重的保存数量，能够有效控制权重保存的磁盘空间。
2. 其他回调函数应用请参考[回调函数配置](https://www.mindspore.cn/mindformers/docs/zh-CN/r1.7.0/feature/configuration.html#callbacks%E9%85%8D%E7%BD%AE)。

### 断点续训

进行断点续训时，需要基于上次训练使用的yaml配置文件，修改load_checkpoint指定到上一次训练任务时保存的权重目录，即output_dir参数指定目录下的checkpoint目录，resume_training设置为True。详情参考[断点续训](https://www.mindspore.cn/mindformers/docs/zh-CN/r1.7.0/feature/resume_training.html)。
