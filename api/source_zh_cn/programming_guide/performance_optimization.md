# 性能优化

<a href="https://gitee.com/mindspore/docs/blob/master/api/source_zh_cn/programming_guide/performance_optimization.md" target="_blank"><img src="../_static/logo_source.png"></a>

MindSpore提供了多种性能优化方法，用户可根据实际情况，利用它们来提升训练和推理的性能。

| 优化阶段 | 优化方法 | 支持情况 |
| --- | --- | --- |
| 训练 | [分布式并行训练](https://www.mindspore.cn/tutorial/zh-CN/master/advanced_use/distributed_training_tutorials.html) | Ascend、GPU |
| | [混合精度](https://www.mindspore.cn/tutorial/zh-CN/master/advanced_use/mixed_precision.html) | Ascend、GPU |
| | [图算融合](https://www.mindspore.cn/tutorial/zh-CN/master/advanced_use/graph_kernel_fusion.html) | Ascend |
| | [梯度累积](https://www.mindspore.cn/tutorial/zh-CN/master/advanced_use/gradient_accumulation.html) | Ascend、GPU |
| 推理 | [训练后量化](https://www.mindspore.cn/lite/tutorial/zh-CN/master/use/post_training_quantization.html) | Lite |