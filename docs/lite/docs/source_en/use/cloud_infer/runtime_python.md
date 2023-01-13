# Using Python Interface to Perform Cloud-side Inference

<a href="https://gitee.com/mindspore/docs/blob/master/docs/lite/docs/source_en/use/cloud_infer/runtime_python.md" target="_blank"><img src="https://mindspore-website.obs.cn-north-4.myhuaweicloud.com/website-images/master/resource/_static/logo_source_en.png"></a>

## Overview

This tutorial describes how to perform cloud-side inference with MindSpore Lite by using the [Python interface](https://mindspore.cn/lite/api/en/master/mindspore_lite.html).

MindSpore Lite cloud-side inference is supported to run in Linux environment deployment only. Ascend 310/310P/910, Nvidia GPU and CPU hardware backends are supported.

To experience the MindSpore Lite device-side inference process, please refer to the document [Using the Python interface to perform device-side inference](https://www.mindspore.cn/lite/docs/en/master/quick_start/quick_start_python.html).

Running MindSpore Lite inference framework mainly consists of the following steps:

1. Model reading: Export MindIR model via MindSpore or get MindIR model by [model conversion tool](https://www.mindspore.cn/lite/docs/en/master/use/cloud_infer/converter_tool.html).
2. Create configuration context: Create a configuration context [Context](https://mindspore.cn/lite/api/en/master/mindspore_lite/mindspore_lite.Context.html#mindspore_lite.Context) and save some basic configuration parameters used to guide model compilation and model execution.
3. Model creation and compilation: Before executing inference, you need to call [build_from_file](https://mindspore.cn/lite/api/en/master/mindspore_lite/mindspore_lite.Model.html#mindspore_lite.Model.build_from_file) interface of [Model](https://mindspore.cn/lite/api/en/master/mindspore_lite/mindspore_lite.Model.html#mindspore_lite.Model) for model loading and model compilation. The model loading phase parses the file cache into a runtime model. The model compilation phase can take more time, so it is recommended that the model be created once, compiled once and performed inference about multiple times.
4. Input data: The input data needs to be padded before the model execution.
5. Execute inference: Use [Predict](https://mindspore.cn/lite/api/en/master/mindspore_lite/mindspore_lite.Model.html#mindspore_lite.Model.predict) of [Model](https://mindspore.cn/lite/api/en/master/mindspore_lite/mindspore_lite.Model.html#mindspore_lite.Model) method for model inference.

![img](../../images/lite_runtime.png)

## Preparation

1. The following code samples are taken from [Using Python interface to perform cloud-side inference sample code](https://gitee.com/mindspore/mindspore/tree/master/mindspore/lite/examples/cloud_infer/quick_start_python).

2. Export the MindIR model via MindSpore, or get the MindIR model by converting it with [model conversion tool](https://www.mindspore.cn/lite/docs/en/master/use/cloud_infer/converter_tool.html) and copy it to the `mindspore/lite/examples/cloud_infer/runtime_cpp/model` directory. You can download the MobileNetV2 model file [mobilenetv2.mindir](https://download.mindspore.cn/model_zoo/official/lite/quick_start/mobilenetv2.mindir) and input data [input.bin](https://download.mindspore.cn/model_zoo/official/lite/quick_start/input.bin).

3. Install the MindSpore Lite cloud-side inference Python package for Python version 3.7 via pip.

    ```bash
    python -m pip install https://ms-release.obs.cn-north-4.myhuaweicloud.com/${MINDSPORE_LITE_VERSION}/MindSpore/lite/release/centos_x86/cloud_fusion/mindspore_lite-${MINDSPORE_LITE_VERSION}-cp37-cp37m-linux_x86.whl --trusted-host ms-release.obs.cn-north-4.myhuaweicloud.com -i https://pypi.tuna.tsinghua.edu.cn/simple
    ```

## Creating Configuration Context

Create the configuration context `Context`. Since this tutorial demonstrates a scenario where inference is performed on a CPU device, the created CPU device hardware information needs to be added to the context.

```python
import numpy as np
import mindspore_lite as mslite

# init context, and add CPU device info
cpu_device_info = mslite.CPUDeviceInfo(enable_fp16=False)
context = mslite.Context(thread_num=1, thread_affinity_mode=2)
context.append_device_info(cpu_device_info)
```

If users need to run inference on Ascend or GPU devices, they need to add Ascend or GPU device hardware information.

```python
import numpy as np
import mindspore_lite as mslite

# init context, and add Ascend device info and CPU device info
ascend_device_info = mslite.AscendDeviceInfo(device_id=0)
cpu_device_info = mslite.CPUDeviceInfo(enable_fp16=False)
context = mslite.Context(thread_num=1, thread_affinity_mode=2)
context.append_device_info(ascend_device_info)
context.append_device_info(cpu_device_info)
```

If the user needs to run inference on a GPU device, the GPU device hardware information needs to be added.

```python
import numpy as np
import mindspore_lite as mslite

# init context, and add Ascend device info and CPU device info
gpu_device_info = mslite.GPUDeviceInfo(device_id=0)
cpu_device_info = mslite.CPUDeviceInfo(enable_fp16=False)
context = mslite.Context(thread_num=1, thread_affinity_mode=2)
context.append_device_info(gpu_device_info)
context.append_device_info(cpu_device_info)
```

## Model Loading and Compilation

Model loading and compilation can be done by calling [build_from_file](https://www.mindspore.cn/lite/api/en/master/mindspore_lite/mindspore_lite.Model.html#mindspore_lite.Model.build_from_file) interface of `Model` to load and compile the runtime model directly from the file cache.

```python
# build model from file
MODEL_PATH = "./model/mobilenetv2.mindir"
IN_DATA_PATH = "./model/input.bin"
model = mslite.Model()
model.build_from_file(MODEL_PATH, mslite.ModelType.MINDIR, context)
```

## Inputting the Data

The way that this tutorial sets the input data is importing from a file. For other ways to set the input data, please refer to [predict](https://www.mindspore.cn/lite/api/en/master/mindspore_lite/mindspore_lite.Model.html#mindspore_lite.Model.predict) interface of `Model`.

```python
# set model input
inputs = model.get_inputs()
in_data = np.fromfile(IN_DATA_PATH, dtype=np.float32)
inputs[0].set_data_from_numpy(in_data)
```

## Executing Inference

Call [predict](https://www.mindspore.cn/lite/api/en/master/mindspore_lite/mindspore_lite.Model.html#mindspore_lite.Model.predict) interface of `Model` to perform inference, and the inference result is output to `output`.

```python
# execute inference
outputs = model.get_outputs()
model.predict(inputs, outputs)
```

## Obtaining the Output

Print the output results after performing inference. Iterate through the `outputs` list and print the name, data size, number of elements, and the first 50 data for each output Tensor.

```python
# get output
for output in outputs:
  tensor_name = output.get_tensor_name().rstrip()
  data_size = output.get_data_size()
  element_num = output.get_element_num()
  print("tensor name is:%s tensor size is:%s tensor elements num is:%s" % (tensor_name, data_size, element_num))
  data = output.get_data_to_numpy()
  data = data.flatten()
  print("output data is:", end=" ")
  for i in range(50):
    print(data[i], end=" ")
  print("")
```
