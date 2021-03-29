# Performance Optimization

<!-- TOC -->

- [Performance Optimization](#performance-optimization)

<!-- /TOC -->

<a href="https://gitee.com/mindspore/docs/blob/master/docs/programming_guide/source_en/performance_optimization.md" target="_blank"><img src="https://gitee.com/mindspore/docs/raw/master/resource/_static/logo_source.png"></a>

MindSpore provides a variety of performance optimization methods, users can use them to improve the performance of training and inference according to the actual situation.

| Optimization Stage | Optimization Method | Supported |
| --- | --- | --- |
| Training | [Distributed Training](https://www.mindspore.cn/tutorial/training/en/master/advanced_use/distributed_training_tutorials.html) | Ascend, GPU |
| | [Mixed Precision](https://www.mindspore.cn/tutorial/training/en/master/advanced_use/enable_mixed_precision.html) | Ascend, GPU |
| | [Graph Kernel Fusion](https://www.mindspore.cn/tutorial/training/en/master/advanced_use/enable_graph_kernel_fusion.html) | Ascend, GPU |
| | [Gradient Accumulation](https://www.mindspore.cn/tutorial/training/en/master/advanced_use/apply_gradient_accumulation.html) | GPU |
| Inference | [Quantization After Training](https://www.mindspore.cn/tutorial/lite/en/master/use/post_training_quantization.html) | Lite |
