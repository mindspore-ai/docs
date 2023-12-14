# bfloat16 Datatype Support Status

[![View Source On Gitee](https://mindspore-website.obs.cn-north-4.myhuaweicloud.com/website-images/r2.2/resource/_static/logo_source_en.svg)](https://gitee.com/mindspore/docs/blob/r2.2/docs/mindspore/source_en/note/bfloat16_support.md)

## Overview

bfloat16 (BF16) is a new floating-point format that can accelerate machine learning (deep learning training, in particular) algorithms.

FP16 format has 5 bits of exponent and 10 bits of mantissa, while BF16 has 8 bits of exponent and 7 bits of mantissa. Compared to FP32, while reducing the precision (only 7 bits mantissa), BF16 retains a range that is similar to FP32, which makes it appropriate for deep learning training.

## Support List

- When computing with tensors of bfloat16 data type, the operators used must also support bfloat16 data type. Currently, only Ascend backend has adapted operators.
- The bfloat16 data type does not support implicit type conversion, that is, when the data types of parameters are inconsistent, the bfloat16 precision type will not be automatically converted to a higher precision type.

|API Name|Ascend|Descriptions|Version|
|:----|:---------|:----|:----|
|[mindspore.Tensor.asnumpy](https://www.mindspore.cn/docs/en/r2.2/api_python/mindspore/Tensor/mindspore.Tensor.asnumpy.html)|❌|Since numpy does not support bfloat16 data type, it is not possible to convert a tensor of bfloat16 type to numpy type.||
|[mindspore.amp.auto_mixed_precision](https://www.mindspore.cn/docs/en/r2.2/api_python/amp/mindspore.amp.auto_mixed_precision.html)|✔️|When using the auto-mixed-precision interface, you can specify bfloat16 as the low-precision data type.||
|[mindspore.amp.custom_mixed_precision](https://www.mindspore.cn/docs/en/r2.2/api_python/amp/mindspore.amp.custom_mixed_precision.html)|✔️|When using the custom-mixed-precision interface, you can specify bfloat16 as the low-precision data type.||
|[mindspore.ops.AllGather](https://www.mindspore.cn/docs/en/r2.2/api_python/ops/mindspore.ops.AllGather.html)|✔️||2.2.10|
|[mindspore.ops.AllReduce](https://www.mindspore.cn/docs/en/r2.2/api_python/ops/mindspore.ops.AllReduce.html)|✔️||2.2.10|
|[mindspore.ops.BatchMatMul](https://www.mindspore.cn/docs/en/r2.2/api_python/ops/mindspore.ops.BatchMatMul.html)|✔️||2.2.10|
|[mindspore.ops.Broadcast](https://www.mindspore.cn/docs/en/r2.2/api_python/ops/mindspore.ops.Broadcast.html)|✔️||2.2.10|
|[mindspore.ops.Cast](https://www.mindspore.cn/docs/en/r2.2/api_python/ops/mindspore.ops.Cast.html)|✔️|2.2.0|
|[mindspore.ops.LayerNorm](https://www.mindspore.cn/docs/en/r2.2/api_python/ops/mindspore.ops.LayerNorm.html)|✔️|2.2.0|
|[mindspore.ops.Mul](https://www.mindspore.cn/docs/en/r2.2/api_python/ops/mindspore.ops.Mul.html)|✔️|2.2.0|
|[mindspore.ops.ReduceScatter](https://www.mindspore.cn/docs/en/r2.2/api_python/ops/mindspore.ops.ReduceScatter.html)|✔️||2.2.10|
|[mindspore.ops.ReduceSum](https://www.mindspore.cn/docs/en/r2.2/api_python/ops/mindspore.ops.ReduceSum.html)|✔️|2.2.0|
|[mindspore.ops.Sub](https://www.mindspore.cn/docs/en/r2.2/api_python/ops/mindspore.ops.Sub.html)|✔️|2.2.0|
|[mindspore.ops.Softmax](https://www.mindspore.cn/docs/en/r2.2/api_python/ops/mindspore.ops.Softmax.html)|✔️|2.2.0|
|[mindspore.ops.Transpose](https://www.mindspore.cn/docs/en/r2.2/api_python/ops/mindspore.ops.Transpose.html)|✔️|2.2.0|

The overall bfloat16 datatype capability is currently supported, and more operators will be added to support the bfloat16 datatype.
