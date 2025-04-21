# bfloat16 数据类型支持情况

[![查看源文件](https://mindspore-website.obs.cn-north-4.myhuaweicloud.com/website-images/r2.6.0rc1/resource/_static/logo_source.svg)](https://gitee.com/mindspore/docs/blob/r2.6.0rc1/docs/mindspore/source_zh_cn/api_python/bfloat16_support.md)

## 概述

bfloat16（BF16）是一种新的浮点格式，可以加速机器学习（尤其是深度学习训练）算法。

FP16 格式有 5 位指数和 10 位尾数，而 BF16 有 8 位指数和 7 位尾数。与 FP32 相比，BF16 在降低精度（仅 7 位尾数）的同时，保留了与 FP32 相似的范围，这使其适用于深度学习训练。

## 支持列表

- 当使用bfloat16数据类型的Tensor进行计算时，所用到的算子也需支持bfloat16数据类型。当前仅在Ascend后端存在适配的算子。
- bfloat16数据类型暂不支持隐式类型转换，即当参数数据类型不一致时，bfloat16精度类型不会被自动转换成较高精度类型。

|API名称|Ascend|说明|
|:----|:---------|:---------|
|[mindspore.Tensor.asnumpy](https://www.mindspore.cn/docs/zh-CN/r2.6.0rc1/api_python/mindspore/Tensor/mindspore.Tensor.asnumpy.html)|❌|由于numpy不支持bfloat16数据类型，无法将bfloat16类型的Tensor转换为numpy类型。|
|[mindspore.amp.auto_mixed_precision](https://www.mindspore.cn/docs/zh-CN/r2.6.0rc1/api_python/amp/mindspore.amp.auto_mixed_precision.html)|✔️|使用自动混合精度接口时，支持将低精度的数据类型指定为bfloat16。|
|[mindspore.amp.custom_mixed_precision](https://www.mindspore.cn/docs/zh-CN/r2.6.0rc1/api_python/amp/mindspore.amp.custom_mixed_precision.html)|✔️|使用自定义混合精度接口时，支持将低精度的数据类型指定为bfloat16。|
|[mindspore.load_checkpoint](https://www.mindspore.cn/docs/zh-CN/r2.6.0rc1/api_python/mindspore/mindspore.load_checkpoint.html)|✔️||
|[mindspore.save_checkpoint](https://www.mindspore.cn/docs/zh-CN/r2.6.0rc1/api_python/mindspore/mindspore.save_checkpoint.html)|✔️||
|[mindspore.ops.Add](https://www.mindspore.cn/docs/zh-CN/r2.6.0rc1/api_python/ops/mindspore.ops.Add.html)|✔️||
|[mindspore.ops.AddN](https://www.mindspore.cn/docs/zh-CN/r2.6.0rc1/api_python/ops/mindspore.ops.AddN.html)|✔️||
|[mindspore.ops.AllGather](https://www.mindspore.cn/docs/zh-CN/r2.6.0rc1/api_python/ops/mindspore.ops.AllGather.html)|✔️||
|[mindspore.ops.AllReduce](https://www.mindspore.cn/docs/zh-CN/r2.6.0rc1/api_python/ops/mindspore.ops.AllReduce.html)|✔️||
|[mindspore.ops.AssignAdd](https://www.mindspore.cn/docs/zh-CN/r2.6.0rc1/api_python/ops/mindspore.ops.AssignAdd.html)|✔️||
|[mindspore.ops.BatchMatMul](https://www.mindspore.cn/docs/zh-CN/r2.6.0rc1/api_python/ops/mindspore.ops.BatchMatMul.html)|✔️||
|[mindspore.ops.Broadcast](https://www.mindspore.cn/docs/zh-CN/r2.6.0rc1/api_python/ops/mindspore.ops.Broadcast.html)|✔️||
|[mindspore.ops.Cast](https://www.mindspore.cn/docs/zh-CN/r2.6.0rc1/api_python/ops/mindspore.ops.Cast.html)|✔️||
|[mindspore.ops.Equal](https://www.mindspore.cn/docs/zh-CN/r2.6.0rc1/api_python/ops/mindspore.ops.Equal.html)|✔️||
|[mindspore.ops.Exp](https://www.mindspore.cn/docs/zh-CN/r2.6.0rc1/api_python/ops/mindspore.ops.Exp.html)|✔️||
|[mindspore.ops.FastGeLU](https://www.mindspore.cn/docs/zh-CN/r2.6.0rc1/api_python/ops/mindspore.ops.FastGeLU.html)|✔️||
|[mindspore.ops.GreaterEqual](https://www.mindspore.cn/docs/zh-CN/r2.6.0rc1/api_python/ops/mindspore.ops.GreaterEqual.html)|✔️||
|[mindspore.ops.LayerNorm](https://www.mindspore.cn/docs/zh-CN/r2.6.0rc1/api_python/ops/mindspore.ops.LayerNorm.html)|✔️||
|[mindspore.ops.LessEqual](https://www.mindspore.cn/docs/zh-CN/r2.6.0rc1/api_python/ops/mindspore.ops.LessEqual.html)|✔️||
|[mindspore.ops.MatMul](https://www.mindspore.cn/docs/zh-CN/r2.6.0rc1/api_python/ops/mindspore.ops.MatMul.html)|✔️||
|[mindspore.ops.Maximum](https://www.mindspore.cn/docs/zh-CN/r2.6.0rc1/api_python/ops/mindspore.ops.Maximum.html)|✔️||
|[mindspore.ops.Minimum](https://www.mindspore.cn/docs/zh-CN/r2.6.0rc1/api_python/ops/mindspore.ops.Minimum.html)|✔️||
|[mindspore.ops.Mul](https://www.mindspore.cn/docs/zh-CN/r2.6.0rc1/api_python/ops/mindspore.ops.Mul.html)|✔️||
|[mindspore.ops.NotEqual](https://www.mindspore.cn/docs/zh-CN/r2.6.0rc1/api_python/ops/mindspore.ops.NotEqual.html)|✔️||
|[mindspore.ops.RealDiv](https://www.mindspore.cn/docs/zh-CN/r2.6.0rc1/api_python/ops/mindspore.ops.RealDiv.html)|✔️||
|[mindspore.ops.ReduceMean](https://www.mindspore.cn/docs/zh-CN/r2.6.0rc1/api_python/ops/mindspore.ops.ReduceMean.html)|✔️||
|[mindspore.ops.ReduceScatter](https://www.mindspore.cn/docs/zh-CN/r2.6.0rc1/api_python/ops/mindspore.ops.ReduceScatter.html)|✔️||
|[mindspore.ops.ReduceSum](https://www.mindspore.cn/docs/zh-CN/r2.6.0rc1/api_python/ops/mindspore.ops.ReduceSum.html)|✔️||
|[mindspore.ops.Select](https://www.mindspore.cn/docs/zh-CN/r2.6.0rc1/api_python/ops/mindspore.ops.Select.html)|✔️||
|[mindspore.ops.Softmax](https://www.mindspore.cn/docs/zh-CN/r2.6.0rc1/api_python/ops/mindspore.ops.Softmax.html)|✔️||
|[mindspore.ops.Sqrt](https://www.mindspore.cn/docs/zh-CN/r2.6.0rc1/api_python/ops/mindspore.ops.Sqrt.html)|✔️||
|[mindspore.ops.Square](https://www.mindspore.cn/docs/zh-CN/r2.6.0rc1/api_python/ops/mindspore.ops.Square.html)|✔️||
|[mindspore.ops.Sub](https://www.mindspore.cn/docs/zh-CN/r2.6.0rc1/api_python/ops/mindspore.ops.Sub.html)|✔️||
|[mindspore.ops.Tile](https://www.mindspore.cn/docs/zh-CN/r2.6.0rc1/api_python/ops/mindspore.ops.Tile.html)|✔️||
|[mindspore.ops.Transpose](https://www.mindspore.cn/docs/zh-CN/r2.6.0rc1/api_python/ops/mindspore.ops.Transpose.html)|✔️||
