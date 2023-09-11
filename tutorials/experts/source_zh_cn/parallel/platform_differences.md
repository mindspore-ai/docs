# 不同平台差异

[![查看源文件](https://mindspore-website.obs.cn-north-4.myhuaweicloud.com/website-images/master/resource/_static/logo_source.png)](https://gitee.com/mindspore/docs/blob/master/tutorials/experts/source_zh_cn/parallel/platform_differences.md)

## 概述

在分布式训练中，不同硬件平台（Ascend、CPU或者GPU）支持的特性也有所不同，用户可以根据自己的平台选择对应的分布式启动方式、并行模式和优化方法。

### 启动方式的差异

- Ascend支持动态组网、mpirun以及rank table启动三种启动方式。
- GPU支持动态组网和mpirun两种启动方式。
- CPU仅支持动态组网启动。

详细过程请参考[启动方式](https://www.mindspore.cn/tutorials/experts/zh-CN/master/parallel/startup_method.html)。

### 并行方式的差异

- Ascend和GPU支持所有并行方式，包括数据并行、半自动并行、自动并行等。
- CPU仅支持数据并行。

详细过程请参考[并行模式](https://www.mindspore.cn/tutorials/experts/zh-CN/master/parallel/parallel_mode.html)。

### 优化特性支持的差异

- Ascend支持所有的优化特性。
- GPU支持除了通信子图提取与复用以外的优化特性。
- CPU不支持优化特性。

详细过程请参考[优化方法](https://www.mindspore.cn/tutorials/experts/zh-CN/master/parallel/optimize_technique.html)。
