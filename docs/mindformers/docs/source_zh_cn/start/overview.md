# 整体架构

[![查看源文件](https://mindspore-website.obs.cn-north-4.myhuaweicloud.com/website-images/master/resource/_static/logo_source.svg)](https://gitee.com/mindspore/docs/blob/master/docs/mindformers/docs/source_zh_cn/start/overview.md)

MindFormers整体架构可分为如下几个部分：

1. 在硬件层面，MindFormers支持用户在Ascend服务器上运行大模型；
2. 在软件层面，MindFormers通过MindSpore提供的Python接口实现大模型相关代码，并由昇腾AI处理器配套软件包提供的算子库进行数据运算；
3. MindFormers目前支持的基础功能特性如下：
   1. 支持大模型[分布式并行](https://www.mindspore.cn/mindformers/docs/zh-CN/dev/function/distributed_parallel.html)运行训练和推理等任务，并行能力包括数据并行、模型并行、超长序列并行等；
   2. 支持[模型权重转换](https://www.mindspore.cn/mindformers/docs/zh-CN/dev/function/weight_conversion.html)、[分布式权重切分与合并](https://www.mindspore.cn/mindformers/docs/zh-CN/dev/function/transform_weight.html)、不同格式[数据集加载](https://www.mindspore.cn/mindformers/docs/zh-CN/dev/function/dataset.html)以及[断点续训](https://www.mindspore.cn/mindformers/docs/zh-CN/dev/function/resume_training.html)等功能；
   3. 支持20+大模型[预训练](https://www.mindspore.cn/mindformers/docs/zh-CN/dev/usage/pre_training.html)、[微调](https://www.mindspore.cn/mindformers/docs/zh-CN/dev/usage/sft_tuning.html)、[推理](https://www.mindspore.cn/mindformers/docs/zh-CN/dev/usage/inference.html)和[评测](https://www.mindspore.cn/mindformers/docs/zh-CN/dev/usage/evaluation.html)等功能，同时支持对模型参数进行[量化](https://www.mindspore.cn/mindformers/docs/zh-CN/dev/usage/quantization.html)，具体支持模型列表可参考[模型库](https://www.mindspore.cn/mindformers/docs/zh-CN/dev/start/models.html)；
4. MindFormers支持用户通过[MindIE](https://www.mindspore.cn/mindformers/docs/zh-CN/dev/usage/mindie_deployment.html)进行模型服务化部署功能，同时支持使用[MindX](https://www.hiascend.com/software/mindx-dl)实现大规模集群调度；后续将支持更多第三方平台，敬请期待。

![/overall_architecture](./image/overall_architecture.png)
