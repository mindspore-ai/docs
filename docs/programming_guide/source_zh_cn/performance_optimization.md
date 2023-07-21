# 性能优化

[![查看源文件](./_static/logo_source.png)](https://gitee.com/mindspore/docs/blob/r1.2/docs/programming_guide/source_zh_cn/performance_optimization.md)

MindSpore提供了多种性能优化方法，用户可根据实际情况，利用它们来提升训练和推理的性能。

| 优化阶段 | 优化方法 | 支持情况 |
| --- | --- | --- |
| 训练 | [分布式并行训练](https://www.mindspore.cn/tutorial/training/zh-CN/r1.2/advanced_use/distributed_training_tutorials.html) | Ascend、GPU |
| | [混合精度](https://www.mindspore.cn/tutorial/training/zh-CN/r1.2/advanced_use/enable_mixed_precision.html) | Ascend、GPU |
| | [图算融合](https://www.mindspore.cn/tutorial/training/zh-CN/r1.2/advanced_use/enable_graph_kernel_fusion.html) | Ascend、GPU |
| | [梯度累积](https://www.mindspore.cn/tutorial/training/zh-CN/r1.2/advanced_use/apply_gradient_accumulation.html) | GPU |
| 推理 | [训练后量化](https://www.mindspore.cn/tutorial/lite/zh-CN/r1.2/use/post_training_quantization.html) | Lite |
