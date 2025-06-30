# 大模型调试调优指南

## 基于MindSpore TransFormers大模型套件的调试调优指南

[MindSpore TransFormers](https://www.mindspore.cn/mindformers/docs/zh-CN/dev/index.html)是MindSpore提供的包含大模型预训练、微调、推理、部署的全流程开发套件，也是MindSpore当前常用的大模型开发套件。

为此，我们总结了大模型训练过程中常见精度问题、通用的精度问题定位方法、精度基准以及大模型场景工具常见用法，详见[大模型精度调试指南](https://www.mindspore.cn/mindformers/docs/zh-CN/dev/advanced_development/precision_optimization.html#)。

同时，为了方便用户进行性能调优，MindSpore TransFormers套件集成了Profiler数据采集的功能，并提供了超参可直接通过模型参数配置使用，详见[大模型性能调试指南](https://www.mindspore.cn/mindformers/docs/zh-CN/dev/advanced_development/performance_optimization.html#)。

## 基于MindSpeed加速库的调试调优指南

[MindSpeed-待补充链接，后续会上线mindspore官网](xxx)是MindSpore支持的昇腾训练加速库，提供了丰富的加速算法和模型。

针对MindSpeed加速库，及[MindSpeed-LLM大模型套件-待补充链接，后续会上线mindspore官网](xxx)，我们也提供了调试调优指南。

- [MindSpeed精度调试指南](https://gitee.com/ascend/MindSpeed-Core-MS/blob/master/docs/precision_opt.md)
- [MindSpeed性能调优指南](https://gitee.com/ascend/MindSpeed-Core-MS/blob/master/docs/performance_opt.md)
- [MindSpeed-LLM精度调试指南  - 待补充](xxxx)
- [MindSpeed-LLM性能调优指南  - 待补充](xxxx)