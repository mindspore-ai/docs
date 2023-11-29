# bfloat16 Datatype Support Status

## Overview

bfloat16 (BF16) is a new floating-point format that can accelerate machine learning (deep learning training, in particular) algorithms.

FP16 format has 5 bits of exponent and 10 bits of mantissa, while BF16 has 8 bits of exponent and 7 bits of mantissa. Compared to FP32, while reducing the precision (only 7 bits mantissa), BF16 retains a range that is similar to FP32, which makes it appropriate for deep learning training.

## Support List

- When computing with tensors of bfloat16 data type, the operators used must also support bfloat16 data type. Currently, only Ascend backend has adapted operators.
- The bfloat16 data type does not support implicit type conversion, that is, when the data types of parameters are inconsistent, the bfloat16 precision type will not be automatically converted to a higher precision type.

|API Name|Ascend|Descriptions|
|:----|:---------|:----|
|[mindspore.Tensor.asnumpy](https://www.mindspore.cn/docs/en/r2.3/api_python/mindspore/Tensor/mindspore.Tensor.asnumpy.html)|❌|Since numpy does not support bfloat16 data type, it is not possible to convert a tensor of bfloat16 type to numpy type.|
|[mindspore.amp.auto_mixed_precision](https://www.mindspore.cn/docs/en/r2.3/api_python/amp/mindspore.amp.auto_mixed_precision.html)|✔️|When using the auto-mixed-precision interface, you can specify bfloat16 as the low-precision data type.|
|[mindspore.amp.custom_mixed_precision](https://www.mindspore.cn/docs/en/r2.3/api_python/amp/mindspore.amp.custom_mixed_precision.html)|✔️|When using the custom-mixed-precision interface, you can specify bfloat16 as the low-precision data type.|
|[mindspore.save_mindir](https://www.mindspore.cn/docs/en/r2.3/api_python/mindspore/mindspore.save_mindir.html)|✔️|When saving a MindIR file, the bfloat16 data type is supported in the network.|
|[mindspore.load_mindir](https://www.mindspore.cn/docs/en/r2.3/api_python/mindspore/mindspore.load_mindir.html)|✔️|When loading a MindIR file, the bfloat16 data type is supported in the network.|