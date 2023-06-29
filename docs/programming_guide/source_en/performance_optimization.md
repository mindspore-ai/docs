# Performance Optimization

<a href="https://gitee.com/mindspore/docs/blob/r1.2/docs/programming_guide/source_en/performance_optimization.md" target="_blank"><img src="./_static/logo_source.png"></a>

MindSpore provides a variety of performance optimization methods, users can use them to improve the performance of training and inference according to the actual situation.

| Optimization Stage | Optimization Method | Supported |
| --- | --- | --- |
| Training | [Distributed Training](https://www.mindspore.cn/tutorial/training/en/r1.2/advanced_use/distributed_training_tutorials.html) | Ascend, GPU |
| | [Mixed Precision](https://www.mindspore.cn/tutorial/training/en/r1.2/advanced_use/enable_mixed_precision.html) | Ascend, GPU |
| | [Graph Kernel Fusion](https://www.mindspore.cn/tutorial/training/en/r1.2/advanced_use/enable_graph_kernel_fusion.html) | Ascend, GPU |
| | [Gradient Accumulation](https://www.mindspore.cn/tutorial/training/en/r1.2/advanced_use/apply_gradient_accumulation.html) | GPU |
| Inference | [Quantization After Training](https://www.mindspore.cn/tutorial/lite/en/r1.2/use/post_training_quantization.html) | Lite |
