# 体验Python极简推理Demo

<a href="https://gitee.com/mindspore/docs/blob/master/docs/lite/docs/source_zh_cn/quick_start/quick_start_python.md" target="_blank"><img src="https://mindspore-website.obs.cn-north-4.myhuaweicloud.com/website-images/master/resource/_static/logo_source.png"></a>

## 概述

本教程提供了MindSpore Lite执行推理的示例程序，通过文件输入、执行推理、打印推理结果的方式，演示了Python接口进行端侧推理的基本流程，用户能够快速了解MindSpore Lite执行推理相关API的使用。相关代码放置在[mindspore/lite/examples/quick_start_python](https://gitee.com/mindspore/mindspore/tree/master/mindspore/lite/examples/quick_start_python)目录。

下面以Ubuntu 18.04为例，介绍了在Linux X86操作系统配合CPU硬件平台下如何使用Python极简推理Demo：

- 一键安装推理相关模型文件、MindSpore Lite及其所需的依赖，详情参见[一键安装](#一键安装)小节。

- 执行Python极简推理Demo，详情参见[执行极简推理Demo](#执行极简推理demo)小节。

- Python极简推理Demo内容说明，详情参见[极简推理Demo内容说明](#极简推理demo内容说明)小节。

### 一键安装

本环节以全新的Ubuntu 18.04为例，介绍在CPU环境的Linux-x86_64系统上，通过pip安装MindSpore Lite。

进入到[mindspore/lite/examples/quick_start_python](https://gitee.com/mindspore/mindspore/tree/master/mindspore/lite/examples/quick_start_python)目录下，以安装1.9.0版本的MindSpore Lite为例，执行`lite-cpu-pip.sh`脚本进行一键式安装。安装脚本会下载推理所需的模型和输入数据文件、安装MindSpore_Lite所需的依赖，以及下载并安装MindSpore Lite。

注：此命令可设置安装的MindSpore Lite版本，由于1.8.0版本开始支持Python接口，因此版本不能低于1.8.0。

```bash
MINDSPORE_LITE_VERSION=1.9.0 bash ./lite-cpu-pip.sh
```

> 若MobileNetV2模型下载失败，请手动下载相关模型文件[mobilenetv2.ms](https://download.mindspore.cn/model_zoo/official/lite/quick_start/mobilenetv2.ms)，并将其拷贝到`mindspore/lite/examples/quick_start_python/model`目录。
>
> 若input.bin输入数据文件下载失败，请手动下载相关输入数据文件[input.bin](https://download.mindspore.cn/model_zoo/official/lite/quick_start/input.bin)，并将其拷贝到`mindspore/lite/examples/quick_start_python/model`目录。
>
> 若使用脚本下载MindSpore Lite推理框架失败，请手动下载硬件平台为CPU、操作系统为Linux-x86_64或Linux-aarch64的MindSpore Lite 模型推理框架[mindspore_lite-{version}-cp37-cp37m-linux_x86_64.whl](https://www.mindspore.cn/lite/docs/zh-CN/master/use/downloads.html)，用户可以使用`uname -m`命令在终端上查询操作系统。并将其拷贝到`mindspore/lite/examples/quick_start_python`目录下。
>
> 若需要使用Python3.7以上版本对应的MindSpore Lite，请在本地[编译](https://www.mindspore.cn/lite/docs/zh-CN/master/use/build.html)，注意Python API模块编译依赖：Python >= 3.7.0、NumPy >= 1.17.0、wheel >= 0.32.0。编译成功后，将`output/`目录下生成的Whl安装包拷贝到`mindspore/lite/examples/quick_start_python`目录下。
>
> 若`mindspore/lite/examples/quick_start_python`目录下不存在MindSpore Lite安装包，则一键安装脚本将会卸载当前的MindSpore Lite安装包后，从华为镜像下载并安装MindSpore Lite。否则，若目录下存在MindSpore Lite安装包，则会优先安装该安装包。
>
> 通过手动下载并且将文件放到指定位置后，需要再次执行lite-cpu-pip.sh脚本才能完成一键安装。

执行成功将会显示MindSpore Lite安装成功如下，模型文件和输入数据文件可在`mindspore/lite/examples/quick_start_python/model`目录下找到。

```text
Successfully installed mindspore-lite-1.9.0
```

### 执行极简推理Demo

一键安装后，进入[mindspore/lite/examples/quick_start_python](https://gitee.com/mindspore/mindspore/tree/master/mindspore/lite/examples/quick_start_python)目录，并执行以下命令，体验MindSpore Lite推理MobileNetV2模型。

```bash
python quick_start_python.py
```

执行完成后将能得到如下结果，打印输出Tensor的名称、输出Tensor的大小，输出Tensor的数量以及前50个数据。

```text
tensor name is:Softmax-65 tensor size is:4004 tensor elements num is:1001
output data is: 1.02271215e-05 9.92699e-06 1.6968432e-05 6.8573616e-05 9.731416e-05 0.0011149431 0.00020790889 0.0010379024 8.951246e-06 3.5114933e-06 4.233835e-06 2.8036434e-06 2.6037442e-06 1.8385846e-06 1.1539755e-05 8.275104e-05 9.712361e-06 1.1271673e-05 4.0994237e-06 2.0738518e-05 2.3865257e-06 6.13505e-06 2.2388376e-06 3.8502785e-06 6.7741335e-06 8.045284e-06 7.4303607e-06 3.081847e-06 1.6161586e-05 3.8332796e-06 1.6814663e-05 1.7688351e-05 6.5563186e-06 1.2908386e-06 2.292212e-05 0.00028948952 4.608292e-06 7.4074756e-06 5.352228e-06 1.2963507e-06 3.3694944e-06 6.408071e-06 3.6104643e-06 5.094248e-06 3.1630923e-06 6.4333294e-06 3.2282237e-06 2.03353e-05 2.1681694e-06 4.8566693e-05
```

### 极简推理Demo内容说明

使用MindSpore Lite执行推理主要包括以下步骤：

1. [创建配置上下文](#创建配置上下文)：创建配置上下文[Context](https://www.mindspore.cn/lite/api/zh-CN/master/mindspore_lite/mindspore_lite.Context.html)，保存需要的一些基本配置参数，用于指导模型编译和模型执行。
2. [模型创建、加载与编译](#模型创建加载与编译)：执行推理之前，需要调用`Model`的[build_from_file](https://www.mindspore.cn/lite/api/zh-CN/master/mindspore_lite/mindspore_lite.Model.html#mindspore_lite.Model.build_from_file)接口进行模型加载和模型编译，并将上一步得到的Context配置到Model中。模型加载阶段将文件缓存解析成运行时的模型。模型编译阶段主要进行算子选型调度、子图切分等过程，该阶段会耗费较多时间，所以建议Model创建一次，编译一次，多次推理。
3. [输入数据](#输入数据)：模型执行之前需要向`输入Tensor`中填充数据。
4. [执行推理](#执行推理)：使用`Model`的[predict](https://www.mindspore.cn/lite/api/zh-CN/master/mindspore_lite/mindspore_lite.Model.html#mindspore_lite.Model.predict)接口进行模型推理。
5. [获得输出](#获得输出)：模型执行结束之后，可以通过`输出Tensor`得到推理结果。

更多Python接口的高级用法与示例，请参考[Python API](https://www.mindspore.cn/lite/api/zh-CN/master/mindspore_lite.html)。

![img](../images/lite_runtime.png)

#### 创建配置上下文

创建配置上下文Context。由于本用例演示的是在CPU设备上执行推理的场景，因此需要将创建的CPU设备硬件信息加入上下文。

```python
import numpy as np
import mindspore_lite as mslite


# init context, and add CPU device info
cpu_device_info = mslite.CPUDeviceInfo(enable_fp16=False)
context = mslite.Context(thread_num=1, thread_affinity_mode=2)
context.append_device_info(cpu_device_info)
```

如果用户需要在Ascend设备上运行推理时，添加Ascend设备硬件信息后，必须在调用上下文之前添加CPU设备硬件信息。因为当在Ascend上不支持算子时，系统将尝试CPU是否支持算子。此时，需要切换至带有CPU设备信息的上下文中。

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

#### 模型创建加载与编译

模型加载与编译可以调用`Model`的[build_from_file](https://www.mindspore.cn/lite/api/zh-CN/master/mindspore_lite/mindspore_lite.Model.html#mindspore_lite.Model.build_from_file)接口，直接从文件缓存加载、编译得到运行时的模型。

```python
# build model from file
MODEL_PATH = "./model/mobilenetv2.ms"
IN_DATA_PATH = "./model/input.bin"
model = mslite.Model()
model.build_from_file(MODEL_PATH, mslite.ModelType.MINDIR_LITE, context)
```

#### 输入数据

本用例使用的设置输入数据的方式是从文件导入。其他设置输入数据的方式，请参考`Model`的[predict](https://www.mindspore.cn/lite/api/zh-CN/master/mindspore_lite/mindspore_lite.Model.html#mindspore_lite.Model.predict)接口。

```python
# set model input
inputs = model.get_inputs()
in_data = np.fromfile(IN_DATA_PATH, dtype=np.float32)
inputs[0].set_data_from_numpy(in_data)
```

#### 执行推理

调用`Model`的[predict](https://www.mindspore.cn/lite/api/zh-CN/master/mindspore_lite/mindspore_lite.Model.html#mindspore_lite.Model.predict)接口执行推理，推理结果输出给`output`。

```python
# execute inference
outputs = model.get_outputs()
model.predict(inputs, outputs)
```

#### 获得输出

打印执行推理后的输出结果。遍历`output`列表，打印每个输出Tensor的名字、数据大小、元素数量、shape、以及前50个数据。

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
