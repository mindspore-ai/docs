# bfloat16 Datatype Support Status

[![View Source On Gitee](https://mindspore-website.obs.cn-north-4.myhuaweicloud.com/website-images/br_base/resource/_static/logo_source_en.svg)](https://gitee.com/mindspore/docs/blob/br_base/docs/mindspore/source_en/api_python/bfloat16_support.md)

## Overview

bfloat16 (BF16) is a new floating-point format that can accelerate machine learning (deep learning training, in particular) algorithms.

FP16 format has 5 bits of exponent and 10 bits of mantissa, while BF16 has 8 bits of exponent and 7 bits of mantissa. Compared to FP32, while reducing the precision (only 7 bits mantissa), BF16 retains a range that is similar to FP32, which makes it appropriate for deep learning training.

## Support List

- When computing with tensors of bfloat16 data type, the operators used must also support bfloat16 data type. Currently, only Ascend backend has adapted operators.
- The bfloat16 data type does not support implicit type conversion, that is, when the data types of parameters are inconsistent, the bfloat16 precision type will not be automatically converted to a higher precision type.

|API Name|Ascend|Descriptions|
|:----|:---------|:----|
|[mindspore.Tensor.asnumpy](https://www.mindspore.cn/docs/en/br_base/api_python/mindspore/Tensor/mindspore.Tensor.asnumpy.html)|❌|Since numpy does not support bfloat16 data type, it is not possible to convert a tensor of bfloat16 type to numpy type.|
|[mindspore.amp.auto_mixed_precision](https://www.mindspore.cn/docs/en/br_base/api_python/amp/mindspore.amp.auto_mixed_precision.html)|✔️|When using the auto-mixed-precision interface, you can specify bfloat16 as the low-precision data type.|
|[mindspore.amp.custom_mixed_precision](https://www.mindspore.cn/docs/en/br_base/api_python/amp/mindspore.amp.custom_mixed_precision.html)|✔️|When using the custom-mixed-precision interface, you can specify bfloat16 as the low-precision data type.|
|[mindspore.load_checkpoint](https://www.mindspore.cn/docs/en/br_base/api_python/mindspore/mindspore.load_checkpoint.html)|✔️||
|[mindspore.save_checkpoint](https://www.mindspore.cn/docs/en/br_base/api_python/mindspore/mindspore.save_checkpoint.html)|✔️||
|[mindspore.ops.Add](https://www.mindspore.cn/docs/en/br_base/api_python/ops/mindspore.ops.Add.html)|✔️||
|[mindspore.ops.AddN](https://www.mindspore.cn/docs/en/br_base/api_python/ops/mindspore.ops.AddN.html)|✔️||
|[mindspore.ops.AllGather](https://www.mindspore.cn/docs/en/br_base/api_python/ops/mindspore.ops.AllGather.html)|✔️||
|[mindspore.ops.AllReduce](https://www.mindspore.cn/docs/en/br_base/api_python/ops/mindspore.ops.AllReduce.html)|✔️||
|[mindspore.ops.AssignAdd](https://www.mindspore.cn/docs/en/br_base/api_python/ops/mindspore.ops.AssignAdd.html)|✔️||
|[mindspore.ops.BatchMatMul](https://www.mindspore.cn/docs/en/br_base/api_python/ops/mindspore.ops.BatchMatMul.html)|✔️||
|[mindspore.ops.Broadcast](https://www.mindspore.cn/docs/en/br_base/api_python/ops/mindspore.ops.Broadcast.html)|✔️||
|[mindspore.ops.Cast](https://www.mindspore.cn/docs/en/br_base/api_python/ops/mindspore.ops.Cast.html)|✔️||
|[mindspore.ops.Equal](https://www.mindspore.cn/docs/en/br_base/api_python/ops/mindspore.ops.Equal.html)|✔️||
|[mindspore.ops.Exp](https://www.mindspore.cn/docs/en/br_base/api_python/ops/mindspore.ops.Exp.html)|✔️||
|[mindspore.ops.FastGeLU](https://www.mindspore.cn/docs/en/br_base/api_python/ops/mindspore.ops.FastGeLU.html)|✔️||
|[mindspore.ops.GreaterEqual](https://www.mindspore.cn/docs/en/br_base/api_python/ops/mindspore.ops.GreaterEqual.html)|✔️||
|[mindspore.ops.LayerNorm](https://www.mindspore.cn/docs/en/br_base/api_python/ops/mindspore.ops.LayerNorm.html)|✔️||
|[mindspore.ops.LessEqual](https://www.mindspore.cn/docs/en/br_base/api_python/ops/mindspore.ops.LessEqual.html)|✔️||
|[mindspore.ops.MatMul](https://www.mindspore.cn/docs/en/br_base/api_python/ops/mindspore.ops.MatMul.html)|✔️||
|[mindspore.ops.Maximum](https://www.mindspore.cn/docs/en/br_base/api_python/ops/mindspore.ops.Maximum.html)|✔️||
|[mindspore.ops.Minimum](https://www.mindspore.cn/docs/en/br_base/api_python/ops/mindspore.ops.Minimum.html)|✔️||
|[mindspore.ops.Mul](https://www.mindspore.cn/docs/en/br_base/api_python/ops/mindspore.ops.Mul.html)|✔️||
|[mindspore.ops.NotEqual](https://www.mindspore.cn/docs/en/br_base/api_python/ops/mindspore.ops.NotEqual.html)|✔️||
|[mindspore.ops.RealDiv](https://www.mindspore.cn/docs/en/br_base/api_python/ops/mindspore.ops.RealDiv.html)|✔️||
|[mindspore.ops.ReduceMean](https://www.mindspore.cn/docs/en/br_base/api_python/ops/mindspore.ops.ReduceMean.html)|✔️||
|[mindspore.ops.ReduceScatter](https://www.mindspore.cn/docs/en/br_base/api_python/ops/mindspore.ops.ReduceScatter.html)|✔️||
|[mindspore.ops.ReduceSum](https://www.mindspore.cn/docs/en/br_base/api_python/ops/mindspore.ops.ReduceSum.html)|✔️||
|[mindspore.ops.Select](https://www.mindspore.cn/docs/en/br_base/api_python/ops/mindspore.ops.Select.html)|✔️||
|[mindspore.ops.Softmax](https://www.mindspore.cn/docs/en/br_base/api_python/ops/mindspore.ops.Softmax.html)|✔️||
|[mindspore.ops.Sqrt](https://www.mindspore.cn/docs/en/br_base/api_python/ops/mindspore.ops.Sqrt.html)|✔️||
|[mindspore.ops.Square](https://www.mindspore.cn/docs/en/br_base/api_python/ops/mindspore.ops.Square.html)|✔️||
|[mindspore.ops.Sub](https://www.mindspore.cn/docs/en/br_base/api_python/ops/mindspore.ops.Sub.html)|✔️||
|[mindspore.ops.Tile](https://www.mindspore.cn/docs/en/br_base/api_python/ops/mindspore.ops.Tile.html)|✔️||
|[mindspore.ops.Transpose](https://www.mindspore.cn/docs/en/br_base/api_python/ops/mindspore.ops.Transpose.html)|✔️||
