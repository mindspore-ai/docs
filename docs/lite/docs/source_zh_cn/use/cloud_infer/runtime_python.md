# 体验Python极简推理Demo

<a href="https://gitee.com/mindspore/docs/blob/master/docs/lite/docs/source_zh_cn/use/cloud_infer/runtime_python.md" target="_blank"><img src="https://mindspore-website.obs.cn-north-4.myhuaweicloud.com/website-images/master/resource/_static/logo_source.png"></a>

## 概述

本教程介绍如何使用[Python接口](https://mindspore.cn/lite/api/zh-CN/master/mindspore_lite.html)执行MindSpore Lite云侧推理。

MindSpore Lite云侧推理仅支持在Linux环境部署运行。支持Ascend310、Ascend310P、Nvidia GPU和CPU硬件后端。

如需体验MindSpore Lite端侧推理流程，请参考文档[使用Python接口执行端侧推理](https://www.mindspore.cn/lite/docs/zh-CN/master/use/runtime_python.html)。

使用MindSpore Lite推理框架主要包括以下步骤：

1. 模型读取：通过MindSpore导出MindIR模型，或者由[模型转换工具](https://www.mindspore.cn/lite/docs/zh-CN/master/use/cloud_infer/converter_tool.html)转换获得MindIR模型。
2. 创建配置上下文：创建配置上下文[Context](https://mindspore.cn/lite/api/zh-CN/master/mindspore_lite/mindspore_lite.Context.html#mindspore_lite.Context)，保存需要的一些基本配置参数，用于指导模型编译和模型执行。
3. 模型加载与编译：执行推理之前，需要调用[Model](https://mindspore.cn/lite/api/zh-CN/master/mindspore_lite/mindspore_lite.Model.html#mindspore_lite.Model)的[build_from_file](https://mindspore.cn/lite/api/zh-CN/master/mindspore_lite/mindspore_lite.Model.html#mindspore_lite.Model.build_from_file)接口进行模型加载和模型编译。模型加载阶段将文件缓存解析成运行时的模型。模型编译阶段会耗费较多时间所以建议Model创建一次，编译一次，多次推理。
4. 输入数据：模型执行之前需要填充输入数据。
5. 执行推理：使用[Model](https://mindspore.cn/lite/api/zh-CN/master/mindspore_lite/mindspore_lite.Model.html#mindspore_lite.Model)的[Predict](https://mindspore.cn/lite/api/zh-CN/master/mindspore_lite/mindspore_lite.Model.html#mindspore_lite.Model.predict)进行模型推理。

![img](../../images/lite_runtime.png)

## 准备工作

1. 以下代码样例来自于[使用C++接口执行云侧推理示例代码](https://gitee.com/mindspore/mindspore/blob/master/mindspore/lite/examples/cloud_infer/quick_start_parallel_python)。

2. 通过MindSpore导出MindIR模型，或者由[模型转换工具](https://www.mindspore.cn/lite/docs/zh-CN/master/use/converter_tool.html)转换获得MindIR模型，并将其拷贝到`mindspore/lite/examples/cloud_infer/quick_start_parallel_python`目录，可以下载MobileNetV2模型文件[mobilenetv2.mindir](https://download.mindspore.cn/model_zoo/official/lite/quick_start/mobilenetv2.mindir)。

3. 通过pip安装Python3.7版本的MindSpore Lite云侧推理Python包。

    ```bash
    python -m pip install https://ms-release.obs.cn-north-4.myhuaweicloud.com/${MINDSPORE_LITE_VERSION}/MindSpore/lite/release/centos_x86/cloud_fusion/mindspore_lite-${MINDSPORE_LITE_VERSION}-cp37-cp37m-linux_x86.whl --trusted-host ms-release.obs.cn-north-4.myhuaweicloud.com -i https://pypi.tuna.tsinghua.edu.cn/simple
    ```

## 创建配置上下文

创建配置上下文`Context`。由于本教程演示的是在CPU设备上执行推理的场景，因此需要将创建的CPU设备硬件信息加入上下文。

```python
import numpy as np
import mindspore_lite as mslite

# init context, and add CPU device info
cpu_device_info = mslite.CPUDeviceInfo(enable_fp16=False)
context = mslite.Context(thread_num=1, thread_affinity_mode=2)
context.append_device_info(cpu_device_info)
```

如果用户需要在Ascend或者GPU设备上运行推理时，需要添加Ascend或者GPU设备硬件信息。

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

如果用户需要在GPU设备上运行推理时，需要添加GPU设备硬件信息。

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

## 模型加载与编译

模型加载与编译可以调用`Model`的[build_from_file](https://www.mindspore.cn/lite/api/zh-CN/master/mindspore_lite/mindspore_lite.Model.html#mindspore_lite.Model.build_from_file)接口，直接从文件缓存加载、编译得到运行时的模型。

```python
# build model from file
MODEL_PATH = "./model/mobilenetv2.mindir"
IN_DATA_PATH = "./model/input.bin"
model = mslite.Model()
model.build_from_file(MODEL_PATH, mslite.ModelType.MINDIR, context)
```

## 输入数据

本教程设置输入数据的方式是从文件导入。其他设置输入数据的方式，请参考`Model`的[predict](https://www.mindspore.cn/lite/api/zh-CN/master/mindspore_lite/mindspore_lite.Model.html#mindspore_lite.Model.predict)接口。

```python
# set model input
inputs = model.get_inputs()
in_data = np.fromfile(IN_DATA_PATH, dtype=np.float32)
inputs[0].set_data_from_numpy(in_data)
```

## 执行推理

调用`Model`的[predict](https://www.mindspore.cn/lite/api/zh-CN/master/mindspore_lite/mindspore_lite.Model.html#mindspore_lite.Model.predict)接口执行推理，推理结果输出给`output`。

```python
# execute inference
outputs = model.get_outputs()
model.predict(inputs, outputs)
```

## 获得输出

打印执行推理后的输出结果。遍历`outputs`列表，打印每个输出Tensor的名字、数据大小、元素数量、以及前50个数据。

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
