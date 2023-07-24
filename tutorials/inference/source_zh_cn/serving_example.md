# 基于MindSpore Serving部署推理服务

`Linux` `Ascend` `GPU` `Serving` `初级` `中级` `高级`

[![查看源文件](_static/logo_source.png)](https://gitee.com/mindspore/docs/blob/r1.2/tutorials/inference/source_zh_cn/serving_example.md)

## 概述

MindSpore Serving是一个轻量级、高性能的服务模块，旨在帮助MindSpore开发者在生产环境中高效部署在线推理服务。当用户使用MindSpore完成模型训练后，导出MindSpore模型，即可使用MindSpore Serving创建该模型的推理服务。  

本文以一个简单的Add网络为例，演示MindSpore Serving如何使用。

### 环境准备

运行示例前，需确保已经正确安装了MindSpore Serving。如果没有，可以通过[MindSpore Serving安装页面](https://gitee.com/mindspore/serving/blob/r1.2/README_CN.md#%E5%AE%89%E8%A3%85)，将MindSpore Serving正确地安装到你的电脑当中，同时通过[MindSpore Serving环境配置页面](https://gitee.com/mindspore/serving/blob/r1.2/README_CN.md#%E9%85%8D%E7%BD%AE%E7%8E%AF%E5%A2%83%E5%8F%98%E9%87%8F)完成环境变量配置。

### 导出模型

使用[add_model.py](https://gitee.com/mindspore/serving/blob/r1.2/example/add/export_model/add_model.py)，构造一个只有Add算子的网络，并导出MindSpore推理部署模型。

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
    output = add(ms.Tensor(x), ms.Tensor(y))
    ms.export(add, ms.Tensor(x), ms.Tensor(y), file_name='tensor_add', file_format='MINDIR')
    dst_dir = '../add/1'
    try:
        os.mkdir(dst_dir)
    except OSError:
        pass

    dst_file = os.path.join(dst_dir, 'tensor_add.mindir')
    copyfile('tensor_add.mindir', dst_file)
    print("copy tensor_add.mindir to " + dst_dir + " success")

    print(x)
    print(y)
    print(output.asnumpy())


if __name__ == "__main__":
    export_net()
```

使用MindSpore定义神经网络需要继承`mindspore.nn.Cell`。Cell是所有神经网络的基类。神经网络的各层需要预先在`__init__`方法中定义，然后通过定义`construct`方法来完成神经网络的前向构造。使用`mindspore`模块的`export`即可导出模型文件。
更为详细完整的示例可以参考[实现一个图片分类应用](https://www.mindspore.cn/tutorial/training/zh-CN/r1.2/quick_start/quick_start.html)。

执行`add_model.py`脚本，生成`tensor_add.mindir`文件，该模型的输入为两个shape为[2,2]的二维Tensor，输出结果是两个输入Tensor之和。

### 部署Serving推理服务

启动Serving服务，以Add用例为例，需要如下文件列表：

```shell
test_dir
├── add/
│    └── servable_config.py
│    └── 1/
│        └── tensor_add.mindir
└── master_with_worker.py
```

- `master_with_worker.py`为启动服务脚本文件。
- `add`为模型文件夹，文件夹名即为模型名。
- `tensor_add.mindir`为上一步网络生成的模型文件，放置在文件夹1下，1为版本号，不同的版本放置在不同的文件夹下，版本号需以纯数字串命名，默认配置下启动最大数值的版本号的模型文件。
- [servable_config.py](https://gitee.com/mindspore/serving/blob/r1.2/example/add/add/servable_config.py)为[模型配置文件](https://www.mindspore.cn/tutorial/inference/zh-CN/r1.2/serving_model.html)，其定义了模型的处理函数，包括`add_common`和`add_cast`两个方法，`add_common`定义了输入为两个普通float32类型的加法操作，`add_cast`定义输入类型为其他类型，经过输入类型转换float32后的加法操作。

模型配置文件内容如下：

```python
import numpy as np
from mindspore_serving.worker import register


def add_trans_datatype(x1, x2):
    """define preprocess, this example has two input and two output"""
    return x1.astype(np.float32), x2.astype(np.float32)


# when with_batch_dim is set to False, only 2x2 add is supported
# when with_batch_dim is set to True(default), Nx2 add is supported, while N is viewed as batch
# float32 inputs/outputs
register.declare_servable(servable_file="tensor_add.mindir", model_format="MindIR", with_batch_dim=False)


# register add_common method in add
@register.register_method(output_names=["y"])
def add_common(x1, x2):  # only support float32 inputs
    """method add_common data flow definition, only call model inference"""
    y = register.call_servable(x1, x2)
    return y


# register add_cast method in add
@register.register_method(output_names=["y"])
def add_cast(x1, x2):
    """method add_cast data flow definition, only call preprocess and model inference"""
    x1, x2 = register.call_preprocess(add_trans_datatype, x1, x2)  # cast input to float32
    y = register.call_servable(x1, x2)
    return y
```

MindSpore Serving提供两种部署方式，轻量级部署和集群部署。轻量级部署master和worker在一个进程中，集群部署方式master和worker部署在不同的进程中。当只有一个worker节点时，用户可以考虑轻量级部署，即将master部署在这个worker所在进程中；当worker节点有多个，为了充分利用资源，可以考虑集群部署方式，选取一台机器作为master，管理所有的worker节点。用户可根据需要进行选择部署。

#### 轻量级部署

服务端调用Python接口直接启动推理进程（master和worker共进程），客户端直接连接推理服务后下发推理任务。
执行[master_with_worker.py](https://gitee.com/mindspore/serving/blob/r1.2/example/add/master_with_worker.py)，完成轻量级部署服务如下：

```python
import os
from mindspore_serving import master
from mindspore_serving import worker

def start():
    servable_dir = os.path.abspath(".")
    worker.start_servable_in_master(servable_dir, "add", device_id=0)
    master.start_grpc_server("127.0.0.1", 5500)

if __name__ == "__main__":
    start()
```

当服务端打印日志`Serving gRPC start success, listening on 0.0.0.0:5500`时，表示Serving服务已加载推理模型完毕。

#### 集群部署

服务端由master进程和worker进程组成，master用来管理集群内所有的worker节点，并进行推理任务的分发。部署方式如下：

部署master：

```python
import os
from mindspore_serving import master

def start():
    servable_dir = os.path.abspath(".")
    master.start_grpc_server("127.0.0.1", 5500)
    master.start_master_server("127.0.0.1", 6500)
if __name__ == "__main__":
    start()
```

部署worker：

```python
import os
from mindspore_serving import worker

def start():
    servable_dir = os.path.abspath(".")
    worker.start_servable(servable_dir, "add", device_id=0,
                          master_ip="127.0.0.1", master_port=6500,
                          worker_ip="127.0.0.1", worker_port=6600)

if __name__ == "__main__":
    start()
```

轻量级部署和集群部署启动worker所使用的接口存在差异，其中，轻量级部署使用`start_servable_in_master`接口启动worker，集群部署使用`start_servable`接口启动worker。

### 执行推理

客户端提供两种方式访问推理服务，一种是通过[gRPC方式](https://www.mindspore.cn/tutorial/inference/zh-CN/r1.2/serving_grpc.html)，一种是通过[RESTful方式](https://www.mindspore.cn/tutorial/inference/zh-CN/r1.2/serving_restful.html)，本文以gRPC方式为例。
使用[client.py](https://gitee.com/mindspore/serving/blob/r1.2/example/add/client.py)，启动Python客户端。

```python
import numpy as np
from mindspore_serving.client import Client


def run_add_common():
    """invoke servable add method add_common"""
    client = Client("localhost", 5500, "add", "add_common")
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
    client = Client("localhost", 5500, "add", "add_cast")
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

```shell
[{'y': array([[2. , 2.],
        [2.,  2.]], dtype=float32)},{'y': array([[4. , 4.],
        [4.,  4.]], dtype=float32)},{'y': array([[6. , 6.],
        [6.,  6.]], dtype=float32)}]
[{'y': array([[2. , 2.],
        [2.,  2.]], dtype=float32)}]
```
