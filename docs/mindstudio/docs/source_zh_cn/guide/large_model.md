# 大模型调试调优指南

[![查看源文件](https://mindspore-website.obs.cn-north-4.myhuaweicloud.com/website-images/r2.7.1/resource/_static/logo_source.svg)](https://gitee.com/mindspore/docs/blob/r2.7.1/docs/mindstudio/docs/source_zh_cn/guide/large_model.md)

## 基于MindSpore TransFormers大模型套件的调试调优指南

[MindSpore TransFormers](https://www.mindspore.cn/mindformers/docs/zh-CN/master/index.html)是MindSpore提供的包含大模型预训练、微调、推理、部署的全流程开发套件，也是MindSpore当前常用的大模型开发套件。

我们总结了大模型训练过程中常见精度问题、通用的精度问题定位方法、精度基准以及大模型场景工具常见用法，详见[大模型精度调试指南](https://www.mindspore.cn/mindformers/docs/zh-CN/master/advanced_development/precision_optimization.html#)。

为了方便用户进行性能调优，MindSpore TransFormers套件集成了以下功能：

- 集成了Profiler数据采集的功能，并提供了超参可直接通过模型参数配置使用，详见[大模型性能调试指南](https://www.mindspore.cn/mindformers/docs/zh-CN/master/advanced_development/performance_optimization.html#)。
- 集成了精度在线监控功能，详见[训练指标监控](https://www.mindspore.cn/mindformers/docs/zh-CN/master/feature/monitor.html)。

## 基于MindSpeed加速库的调试调优指南

[MindSpeed](https://gitcode.com/Ascend/MindSpeed-Core-MS/blob/r0.3.0/README.md)是MindSpore支持的昇腾训练加速库，提供了丰富的加速算法和模型。

针对MindSpeed加速库，以及[MindSpeed-LLM大模型套件](https://gitcode.com/Ascend/MindSpeed-LLM/blob/2.1.0/README.md)，我们也提供了调试调优指南。

- [MindSpeed精度调试指南](https://gitcode.com/Ascend/MindSpeed-Core-MS/blob/r0.3.0/docs/precision_opt.md)
- [MindSpeed性能调优指南](https://gitcode.com/Ascend/MindSpeed-Core-MS/blob/r0.3.0/docs/performance_opt.md)
