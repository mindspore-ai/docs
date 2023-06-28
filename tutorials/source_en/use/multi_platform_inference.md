# Multi-platform Inference

<a href="https://gitee.com/mindspore/docs/blob/r0.5/tutorials/source_en/use/multi_platform_inference.md" target="_blank"><img src="../_static/logo_source.png"></a>

## Overview

Models based on MindSpore training can be used for inference on different hardware platforms. This document introduces the inference process on each platform.

1. Inference on the Ascend 910 AI processor

   MindSpore provides the `model.eval` API for model validation. You only need to import the validation dataset. The processing method of the validation dataset is the same as that of the training dataset. For details about the complete code, see <https://gitee.com/mindspore/mindspore/blob/r0.5/model_zoo/lenet/eval.py>.

   ```python
   res = model.eval(dataset)
   ```
   
   In addition, the` model.predict` interface can be used for inference. For detailed usage, please refer to API description.

2. Inference on the Ascend 310 AI processor

   1. Export the ONNX or GEIR model by referring to the [Export GEIR Model and ONNX Model](https://www.mindspore.cn/tutorial/en/r0.5/use/saving_and_loading_model_parameters.html#geironnx).

   2. For performing inference in the cloud environment, see the [Ascend 910 training and Ascend 310 inference samples](https://support.huaweicloud.com/bestpractice-modelarts/modelarts_10_0026.html). For details about the bare-metal environment (compared with the cloud environment where the Ascend 310 AI processor is deployed locally), see the description document of the Ascend 310 AI processor software package.

3. Inference on a GPU

   1. Export the ONNX model by referring to the [Export GEIR Model and ONNX Model](https://www.mindspore.cn/tutorial/en/r0.5/use/saving_and_loading_model_parameters.html#geironnx).

   2. Perform inference on the NVIDIA GPU by referring to [TensorRT backend for ONNX](https://github.com/onnx/onnx-tensorrt).

## On-Device Inference

The On-Device Inference is based on the MindSpore Predict. Please refer to [On-Device Inference Tutorial](https://www.mindspore.cn/tutorial/en/r0.5/advanced_use/on_device_inference.html) for details.
