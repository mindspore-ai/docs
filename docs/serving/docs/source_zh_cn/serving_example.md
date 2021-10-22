# 基于MindSpore Serving部署推理服务

<!-- TOC -->

- [基于MindSpore Serving部署推理服务](#基于mindspore-serving部署推理服务)
    - [概述](#概述)
        - [环境准备](#环境准备)
        - [下载样例](#下载样例)
        - [导出模型](#导出模型)
        - [部署Serving推理服务](#部署serving推理服务)
            - [配置服务](#配置服务)
            - [启动服务](#启动服务)
        - [执行推理](#执行推理)

<!-- /TOC -->

<a href="https://gitee.com/mindspore/docs/blob/r1.5/docs/serving/docs/source_zh_cn/serving_example.md" target="_blank"><img src="https://gitee.com/mindspore/docs/raw/r1.5/resource/_static/logo_source.png"></a>

## 概述

MindSpore Serving是一个轻量级、高性能的服务模块，旨在帮助MindSpore开发者在生产环境中高效部署在线推理服务。当用户使用MindSpore完成模型训练后，导出MindSpore模型，即可使用MindSpore Serving创建该模型的推理服务。  

本文以一个简单的Add网络为例，演示MindSpore Serving如何使用。

### 环境准备

运行示例前，需确保已经正确安装了MindSpore Serving，并配置了环境变量。MindSpore Serving和安装和配置可以参考[MindSpore Serving安装页面](https://www.mindspore.cn/serving/docs/zh-CN/r1.5/serving_install.html)。

### 下载样例

请先[下载样例](https://gitee.com/mindspore/serving/blob/r1.5/example/tensor_add/)。

### 导出模型

在`export_model`目录下，使用[add_model.py](https://gitee.com/mindspore/serving/blob/r1.5/example/tensor_add/export_model/add_model.py)，构造一个只有Add算子的网络，并导出MindSpore推理部署模型。

```python
import os
from shutil import copyfile
import numpy as np

import mindspore.context as context
import mindspore.nn as nn
import mindspore.ops as ops
import mindspore as ms

context.set_context(mode=context.GRAPH_MODE, device_target="Ascend")


class Net(nn.Cell):
    """Define Net of add"""

    def __init__(self):
        super(Net, self).__init__()
        self.add = ops.Add()

    def construct(self, x_, y_):
        """construct add net"""
        return self.add(x_, y_)


def export_net():
    """Export add net of 2x2 + 2x2, and copy output model `tensor_add.mindir` to directory ../add/1"""
    x = np.ones([2, 2]).astype(np.float32)
    y = np.ones([2, 2]).astype(np.float32)
    add = Net()
    ms.export(add, ms.Tensor(x), ms.Tensor(y), file_name='tensor_add', file_format='MINDIR')
    dst_dir = '../add/1'
    try:
        os.mkdir(dst_dir)
    except OSError:
        pass

    dst_file = os.path.join(dst_dir, 'tensor_add.mindir')
    copyfile('tensor_add.mindir', dst_file)
    print("copy tensor_add.mindir to " + dst_dir + " success")


if __name__ == "__main__":
    export_net()
```

使用MindSpore定义神经网络需要继承`mindspore.nn.Cell`。Cell是所有神经网络的基类。神经网络的各层需要预先在`__init__`方法中定义，然后通过定义`construct`方法来完成神经网络的前向构造。使用`mindspore`模块的`export`即可导出模型文件。
更为详细完整的示例可以参考[初学入门](https://www.mindspore.cn/tutorials/zh-CN/r1.5/quick_start.html)。

执行`add_model.py`脚本，生成`tensor_add.mindir`文件，该模型的输入为两个shape为[2,2]的二维Tensor，输出结果是两个输入Tensor之和。

### 部署Serving推理服务

#### 配置服务

启动Serving服务，以Add用例为例，需要如下文件列表：

```text
tensor_add
├── add/
│   │── servable_config.py
│   └── 1/
│       └── tensor_add.mindir
└── serving_server.py
```

- `serving_server.py`为启动服务脚本文件。
- `add`为模型文件夹，文件夹名即为模型名。
- `tensor_add.mindir`为上一步网络生成的模型文件，放置在文件夹1下，1为版本号，不同的版本放置在不同的文件夹下，版本号需以纯数字串命名，默认配置下启动最大数值的版本号的模型文件。
- [servable_config.py](https://gitee.com/mindspore/serving/blob/r1.5/example/tensor_add/add/servable_config.py)为[模型配置文件](https://www.mindspore.cn/serving/docs/zh-CN/r1.5/serving_model.html)，其定义了模型的处理函数，包括`add_common`和`add_cast`两个方法，`add_common`定义了输入为两个普通float32类型的加法操作，`add_cast`定义输入类型为其他类型，经过输入类型转换float32后的加法操作。

模型配置文件内容如下：

```python
import numpy as np
from mindspore_serving.server import register


def add_trans_datatype(x1, x2):
    """define preprocess, this example has one input and two outputs"""
    return x1.astype(np.float32), x2.astype(np.float32)


# when with_batch_dim is set to False, only 2x2 add is supported
# when with_batch_dim is set to True(default), Nx2 add is supported, while N is viewed as batch
# float32 inputs/outputs
model = register.declare_model(model_file="tensor_add.mindir", model_format="MindIR", with_batch_dim=False)


# register add_common method in add
@register.register_method(output_names=["y"])
def add_common(x1, x2):  # only support float32 inputs
    """method add_common data flow definition, only call model"""
    y = register.add_stage(model, x1, x2, outputs_count=1)
    return y


# register add_cast method in add
@register.register_method(output_names=["y"])
def add_cast(x1, x2):
    """method add_cast data flow definition, only preprocessing and call model"""
    x1, x2 = register.add_stage(add_trans_datatype, x1, x2, outputs_count=2)  # cast input to float32
    y = register.add_stage(model, x1, x2, outputs_count=1)
    return y
```

#### 启动服务

执行[serving_server.py](https://gitee.com/mindspore/serving/blob/r1.5/example/tensor_add/serving_server.py)，完成服务启动：

```python
import os
import sys
from mindspore_serving import server


def start():
    servable_dir = os.path.dirname(os.path.realpath(sys.argv[0]))

    servable_config = server.ServableStartConfig(servable_directory=servable_dir, servable_name="add",
                                                 device_ids=(0, 1))
    server.start_servables(servable_configs=servable_config)

    server.start_grpc_server(address="127.0.0.1:5500")
    server.start_restful_server(address="127.0.0.1:1500")


if __name__ == "__main__":
    start()
```

上述启动脚本将在设备0和1上共加载和运行两个`add`推理副本，来自客户端的推理请求将被切割分流到两个推理副本。

当服务端打印日志`Serving RESTful server start success, listening on 127.0.0.1:1500`时，表示Serving RESTful服务启动成功，推理模型已成功加载。

### 执行推理

客户端提供两种方式访问推理服务，一种是通过[gRPC方式](https://www.mindspore.cn/serving/docs/zh-CN/r1.5/serving_grpc.html)，一种是通过[RESTful方式](https://www.mindspore.cn/serving/docs/zh-CN/r1.5/serving_restful.html)，本文以gRPC方式为例。
使用[serving_client.py](https://gitee.com/mindspore/serving/blob/r1.5/example/tensor_add/serving_client.py)，启动Python客户端。

```python
import numpy as np
from mindspore_serving.client import Client


def run_add_common():
    """invoke servable add method add_common"""
    client = Client("127.0.0.1:5500", "add", "add_common")
    instances = []

    # instance 1
    x1 = np.asarray([[1, 1], [1, 1]]).astype(np.float32)
    x2 = np.asarray([[1, 1], [1, 1]]).astype(np.float32)
    instances.append({"x1": x1, "x2": x2})

    # instance 2
    x1 = np.asarray([[2, 2], [2, 2]]).astype(np.float32)
    x2 = np.asarray([[2, 2], [2, 2]]).astype(np.float32)
    instances.append({"x1": x1, "x2": x2})

    # instance 3
    x1 = np.asarray([[3, 3], [3, 3]]).astype(np.float32)
    x2 = np.asarray([[3, 3], [3, 3]]).astype(np.float32)
    instances.append({"x1": x1, "x2": x2})

    result = client.infer(instances)
    print(result)


def run_add_cast():
    """invoke servable add method add_cast"""
    client = Client("127.0.0.1:5500", "add", "add_cast")
    instances = []
    x1 = np.ones((2, 2), np.int32)
    x2 = np.ones((2, 2), np.int32)
    instances.append({"x1": x1, "x2": x2})
    result = client.infer(instances)
    print(result)


if __name__ == '__main__':
    run_add_common()
    run_add_cast()
```

使用`mindspore_serving.client`定义的`Client`类，客户端定义两个用例，分别调用模型的两个方法，`run_add_common`用例为三对float32类型数组相加操作，`run_add_cast`用例计算两个int32数组相加操作。执行后显示如下返回值，三对float32类型相加结果合集和一对int32类型的相加结果，说明Serving服务已正确执行Add网络的推理。

```text
[{'y': array([[2. , 2.],
        [2.,  2.]], dtype=float32)},{'y': array([[4. , 4.],
        [4.,  4.]], dtype=float32)},{'y': array([[6. , 6.],
        [6.,  6.]], dtype=float32)}]
[{'y': array([[2. , 2.],
        [2.,  2.]], dtype=float32)}]
```
