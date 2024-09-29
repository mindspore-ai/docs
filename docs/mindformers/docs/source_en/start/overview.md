# Overall Structure

[![View Source On Gitee](https://mindspore-website.obs.cn-north-4.myhuaweicloud.com/website-images/master/resource/_static/logo_source_en.svg)](https://gitee.com/mindspore/docs/blob/master/docs/mindformers/docs/source_en/start/overview.md)

The overall architecture of MindFormers can be divided into the following sections:

1. At the hardware level, MindFormers supports users running large models on Ascend servers;
2. At the software level, MindFormers implements the big model-related code through the Python interface provided by MindSpore and performs data computation by the operator libraries provided by the supporting software package of the Ascend AI processor;
3. The basic functionality features currently supported by MindFormers are listed below:
   1. Supports tasks such as running training and inference for large models [distributed parallelism](https://www.mindspore.cn/mindformers/docs/en/dev/function/distributed_parallel.html), with parallel capabilities including data parallelism, model parallelism, ultra-long sequence parallelism;
   2. Supports [model weight conversion](https://www.mindspore.cn/mindformers/docs/en/dev/function/weight_conversion.html), [distributed weight splitting and combination](https://www.mindspore.cn/mindformers/docs/en/dev/function/transform_weight.html), and different format of [dataset loading](https://www.mindspore.cn/mindformers/docs/en/dev/function/dataset.html) and [resumable training after breakpoint](https://www.mindspore.cn/mindformers/docs/en/dev/function/resume_training.html);
   3. Support 20+ large models [pretraining](https://www.mindspore.cn/mindformers/docs/en/dev/usage/pre_training.html), [fine-tuning](https://www.mindspore.cn/mindformers/docs/en/dev/usage/sft_tuning.html), [inference](https://www.mindspore.cn/mindformers/docs/en/dev/usage/inference.html) and [evaluation] (https://www.mindspore.cn/mindformers/docs/en/dev/usage/evaluation.html). Meanwhile, it also supports [quantization](https://www.mindspore.cn/mindformers/docs/en/dev/usage/quantization.html), and the list of supported models can be found in [Model Library](https://www.mindspore.cn/mindformers/docs/en/dev/start/models.html);
4. MindFormers supports users to carry out model service deployment function through [MindIE](https://www.mindspore.cn/mindformers/docs/en/dev/usage/mindie_deployment.html), and also supports the use of [MindX]( https://www.hiascend.com/software/mindx-dl) to realize large-scale cluster scheduling; more third-party platforms will be supported in the future, please look forward to it.
