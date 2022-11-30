# Experiencing the Python Simplified Inference Demo

<a href="https://gitee.com/mindspore/docs/blob/master/docs/lite/docs/source_en/quick_start/quick_start_python.md" target="_blank"><img src="https://mindspore-website.obs.cn-north-4.myhuaweicloud.com/website-images/master/resource/_static/logo_source_en.png"></a>

## Overview

This tutorial provides a sample program for MindSpore Lite to perform inference, demonstrating the [Python interface](https://mindspore.cn/lite/api/en/master/mindspore_lite.html) to perform the basic process of device-side inference through file input, inference execution, and inference result printing, and enables users to quickly understand the use of MindSpore Lite APIs related to inference execution. The related files are put in the directory [mindspore/lite/examples/quick_start_python](https://gitee.com/mindspore/mindspore/tree/master/mindspore/lite/examples/quick_start_python).

The following is an example of how to use the Python Simplified Inference Demo on a Linux X86 operating system and a CPU hardware platform, using Ubuntu 18.04 as an example:

- One-click installation of inference-related model files, MindSpore Lite and its required dependencies. See the [One-click installation](#one-click-installation) section for details.

- Execute the Python Simplified Inference Demo. See the [Execute Demo](#executing-demo) section for details.

- For a description of the Python Simplified Inference Demo content, see the [Demo Content Description](#demo-content-description) section for details.

## One-click Installation

This session introduces the installation of MindSpore Lite for Python version 3.7 via pip on a Linux-x86_64 system with a CPU environment, taking the new Ubuntu 18.04 as an example.

Go to the [mindspore/lite/examples/quick_start_python](https://gitee.com/mindspore/mindspore/tree/master/mindspore/lite/examples/quick_start_python) directory, and execute the `lite-cpu-pip.sh` script for a one-click installation, taking installation of MindSpore Lite version 1.9.0 as an example. Script installation needs to download the model required for inference and input data files, the dependencies required for MindSpore_Lite installation, and download and install MindSpore Lite.

Note: This command sets the installed version of MindSpore Lite. Since the Python interface is supported from MindSpore Lite version 1.8.0, the version cannot be set lower than 1.8.0. See the version provided in [Download MindSpore Lite](https://www.mindspore.cn/lite/docs/en/master/use/downloads.html) for details on the versions that can be set.

```bash
MINDSPORE_LITE_VERSION=1.9.0 bash ./lite-cpu-pip.sh
```

> If the MobileNetV2 model download fails, please manually download the relevant model file [mobilenetv2.ms](https://download.mindspore.cn/model_zoo/official/lite/quick_start/mobilenetv2.ms) and copy it to the `mindspore/lite/examples/quick_start_python/model` directory.
>
> If the input.bin input data file download fails, please manually download the relevant input data file [input.bin](https://download.mindspore.cn/model_zoo/official/lite/quick_start/input.bin) and copy it to the ` mindspore/lite/examples/quick_start_python/model` directory.
>
> If MindSpore Lite inference framework by using the script download fails, please manually download [MindSpore Lite model inference framework](https://www.mindspore.cn/lite/docs/en/master/use/downloads.html) corresponding to the hardware platform of CPU and operating system of Linux-x86_64 or Linux-aarch64. Users can use the `uname -m` command to query the operating system in the terminal, and copy it to the `mindspore/lite/examples/quick_start_python` directory.
>
> If you need to use MindSpore Lite corresponding to Python 3.7 or above, please [compile](https://www.mindspore.cn/lite/docs/en/master/use/build.html) locally. Note that the Python API module compilation depends on Python >= 3.7.0, NumPy >= 1.17.0, wheel >= 0.32.0. After successful compilation, copy the Whl installation package generated in the `output/` directory to the `mindspore/lite/examples/quick_start_python` directory.
>
> If the MindSpore Lite installation package does not exist in the `mindspore/lite/examples/quick_start_python` directory, the one-click installation script will uninstall the currently installed MindSpore Lite and then download and install MindSpore Lite from the Huawei image. Otherwise, if the MindSpore Lite installation package exists in the directory, it will be installed first.
>
> After manually downloading and placing the files in the specified location, you need to execute the lite-cpu-pip.sh script again to complete the one-click installation.

A successful execution will show the following results. The model files and input data files can be found in the `mindspore/lite/examples/quick_start_python/model` directory.

```text
Successfully installed mindspore-lite-1.9.0
```

## Executing Demo

After one-click installation, go to the [mindspore/lite/examples/quick_start_python](https://gitee.com/mindspore/mindspore/tree/master/mindspore/lite/examples/quick_start_python) directory and execute the following command to experience MindSpore Lite inference MobileNetV2 models.

```bash
python quick_start_python.py
```

When the execution is completed, the following results will be obtained, printing the name of the output Tensor, the data size of the output Tensor, the number of elements of the output Tensor and the first 50 pieces of data.

```text
tensor name is:Softmax-65 tensor size is:4004 tensor elements num is:1001
output data is: 1.02271215e-05 9.92699e-06 1.6968432e-05 6.8573616e-05 9.731416e-05 0.0011149431 0.00020790889 0.0010379024 8.951246e-06 3.5114933e-06 4.233835e-06 2.8036434e-06 2.6037442e-06 1.8385846e-06 1.1539755e-05 8.275104e-05 9.712361e-06 1.1271673e-05 4.0994237e-06 2.0738518e-05 2.3865257e-06 6.13505e-06 2.2388376e-06 3.8502785e-06 6.7741335e-06 8.045284e-06 7.4303607e-06 3.081847e-06 1.6161586e-05 3.8332796e-06 1.6814663e-05 1.7688351e-05 6.5563186e-06 1.2908386e-06 2.292212e-05 0.00028948952 4.608292e-06 7.4074756e-06 5.352228e-06 1.2963507e-06 3.3694944e-06 6.408071e-06 3.6104643e-06 5.094248e-06 3.1630923e-06 6.4333294e-06 3.2282237e-06 2.03353e-05 2.1681694e-06 4.8566693e-05
```

## Demo Content Description

Performing inference with MindSpore Lite consists of the following main steps:

1. [Create Configuration Context](#creating-configuration-context): Create a configuration [context](https://www.mindspore.cn/lite/api/en/master/mindspore_lite/mindspore_lite.Context.html) and saves some basic configuration parameters used to guide model compilation and model inference.
2. [Model Loading and Compilation](#model-loading-and-compilation): Before executing inference, you need to call [build_from_file](https://www.mindspore.cn/lite/api/en/master/mindspore_lite/mindspore_lite.Model.html#mindspore_lite.Model.build_from_file) interface of `Model` for model loading and model compilation, and to configure the Context obtained in the previous step into the Model. The model loading phase parses the file cache into a runtime model. The model compilation phase mainly carries out the process of operator selection scheduling, subgraph slicing, etc. This phase will consume more time, so it is recommended that `Model` be loaded once, compiled once, and inferenced for several times.
3. [Input Data](#inputting-data): The model needs to fill the `Input Tensor` with data before executing inference.
4. [Execute Inference](#executing-inference): Use the [predict](https://www.mindspore.cn/lite/api/en/master/mindspore_lite/mindspore_lite.Model.html#mindspore_lite.Model.predict) interface of `Model` for model inference.
5. [Get Output](#getting-outputs): After the model finishes performing inference, you can get the inference result by `Output Tensor`.

For more advanced usage and examples of Python interfaces, please refer to the [Python API](https://www.mindspore.cn/lite/api/en/master/mindspore_lite.html).

![img](./images/lite_runtime.png)

### Creating Configuration Context

Create the configuration context `Context`. Since this tutorial demonstrates a scenario where inference is performed on a CPU device, the created CPU device hardware information needs to be added to the context.

```python
import numpy as np
import mindspore_lite as mslite


# init context, and add CPU device info
cpu_device_info = mslite.CPUDeviceInfo(enable_fp16=False)
context = mslite.Context(thread_num=1, thread_affinity_mode=2)
context.append_device_info(cpu_device_info)
```

If the user needs to perform inference on an Ascend device, the Ascend device hardware information needs to be added.

> When performing inference on Ascend devices, in scenarios where the input format of the original model does not match the input format of the model generated by the transformation, in addition to adding Ascend device hardware information, users have the option to add CPU device hardware information before calling the context. Because in this case, the model generated by the transformation on the Ascend device will contain the `Transpose` operator, which currently needs to perform inference on the CPU, and therefore needs to be switched to a context with information about the CPU device hardware.

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

### Model Loading and Compilation

Model loading and compilation can be done by calling [build_from_file](https://www.mindspore.cn/lite/api/en/master/mindspore_lite/mindspore_lite.Model.html#mindspore_lite.Model.build_from_file) interface of `Model` to load and compile the runtime model directly from the file cache.

```python
# build model from file
MODEL_PATH = "./model/mobilenetv2.ms"
IN_DATA_PATH = "./model/input.bin"
model = mslite.Model()
model.build_from_file(MODEL_PATH, mslite.ModelType.MINDIR_LITE, context)
```

### Inputting Data

This tutorial sets the way of inputting data as importing from a file. For other ways to set the input data, please refer to [predict](https://www.mindspore.cn/lite/api/en/master/mindspore_lite/mindspore_lite.Model.html#mindspore_lite.Model.predict) interface of `Model`.

```python
# set model input
inputs = model.get_inputs()
in_data = np.fromfile(IN_DATA_PATH, dtype=np.float32)
inputs[0].set_data_from_numpy(in_data)
```

### Executing Inference

Call [predict](https://www.mindspore.cn/lite/api/en/master/mindspore_lite/mindspore_lite.Model.html#mindspore_lite.Model.predict) interface of `model` to perform inference, and the inference result is output to `output`.

```python
# execute inference
outputs = model.get_outputs()
model.predict(inputs, outputs)
```

### Getting Outputs

Print the output results after performing inference. Traverse through the `outputs` list and print the name, data size, number of elements, and the first 50 pieces of data for each output Tensor.

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
