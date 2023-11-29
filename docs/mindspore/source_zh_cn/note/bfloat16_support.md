# bfloat16 数据类型支持情况

## 概述

bfloat16（BF16）是一种新的浮点格式，可以加速机器学习（尤其是深度学习训练）算法。

FP16 格式有 5 位指数和 10 位尾数，而 BF16 有 8 位指数和 7 位尾数。与 FP32 相比，BF16 在降低精度（仅 7 位尾数）的同时，保留了与 FP32 相似的范围，这使其适用于深度学习训练。

## 支持列表

- 当使用bfloat16数据类型的Tensor进行计算时，所用到的算子也需支持bfloat16数据类型。当前仅在Ascend后端存在适配的算子。
- bfloat16数据类型暂不支持隐式类型转换，即当参数数据类型不一致时，bfloat16精度类型不会被自动转换成较高精度类型。

|API名称|Ascend|说明|
|:----|:---------|:---------|
|[mindspore.Tensor.asnumpy](https://www.mindspore.cn/docs/zh-CN/r2.3/api_python/mindspore/Tensor/mindspore.Tensor.asnumpy.html)|❌|由于numpy不支持bfloat16数据类型，无法将bfloat16类型的Tensor转换为numpy类型。|
|[mindspore.amp.auto_mixed_precision](https://www.mindspore.cn/docs/zh-CN/r2.3/api_python/amp/mindspore.amp.auto_mixed_precision.html)|✔️|使用自动混合精度接口时，支持将低精度的数据类型指定为bfloat16。|
|[mindspore.amp.custom_mixed_precision](https://www.mindspore.cn/docs/zh-CN/r2.3/api_python/amp/mindspore.amp.custom_mixed_precision.html)|✔️|使用自定义混合精度接口时，支持将低精度的数据类型指定为bfloat16。|
|[mindspore.save_mindir](https://www.mindspore.cn/docs/zh-CN/r2.3/api_python/mindspore/mindspore.save_mindir.html)|✔️|保存MindIR文件时，支持网络中存在bfloat16数据类型。|
|[mindspore.load_mindir](https://www.mindspore.cn/docs/zh-CN/r2.3/api_python/mindspore/mindspore.load_mindir.html)|✔️|加载MindIR文件时，支持网络中存在bfloat16数据类型。|
