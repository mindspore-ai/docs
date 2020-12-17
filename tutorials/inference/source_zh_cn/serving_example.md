# 实现一个Add网络示例

`Linux` `Ascend` `Serving` `初级` `中级` `高级`

<!-- TOC -->

- [实现一个Add网络示例](#实现一个add网络示例)
    - [概述](#概述)
        - [导出模型](#导出模型)
        - [部署Serving推理服务](#部署serving推理服务)
            - [轻量级部署](#轻量级部署)
            - [集群部署](#集群部署)
        - [执行推理](#执行推理)

<!-- /TOC -->

<a href="https://gitee.com/mindspore/docs/tree/master/tutorials/inference/source_zh_cn/serving_example.md" target="_blank"><img src="_static/logo_source.png"></a>

## 概述

MindSpore Serving是一个轻量级、高性能的服务模块，旨在帮助MindSpore开发者在生产环境中高效部署在线推理服务。当用户使用MindSpore完成模型训练后，导出MindSpore模型，即可使用MindSpore Serving创建该模型的推理服务。  

本文以一个简单的Add网络为例，演示MindSpore Serving如何使用。

### 导出模型

使用[add_model.py](https://gitee.com/mindspore/serving/blob/master/mindspore_serving/example/add/export_model/add_model.py)，构造一个只有Add算子的网络，并导出MindSpore推理部署模型。

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
    def __init__(self):
        super(Net, self).__init__()
        self.add = ops.TensorAdd()

    def construct(self, x_, y_):
        return self.add(x_, y_)


def export_net():
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
更为详细完整的示例可以参考[实现一个图片分类应用](https://www.mindspore.cn/tutorial/training/zh-CN/master/quick_start/quick_start.html)。

执行`add_model.py`脚本，生成`tensor_add.mindir`文件，该模型的输入为两个shape为[2,2]的二维Tensor，输出结果是两个输入Tensor之和。

### 部署Serving推理服务

启动Serving服务，当前目录下需要有模型文件夹，如`add`，文件夹下放置版本模型文件和配置文件，文件目录结果如下图所示：

```shell
test_dir
├── add/
│    └── servable_config.py
│    └── 1/
│        └── tensor_add.mindir
└── master_with_worker.py
```

其中，模型文件为上一步网络生成的，即`tensor_add.mindir`文件，放置在文件夹1下，1为版本号，不同的版本放置在不同的文件夹下。
配置文件为[servable_config.py](https://gitee.com/mindspore/serving/blob/master/mindspore_serving/example/add/add/servable_config.py)，其定义了模型的处理函数。

```python
from mindspore_serving.worker import register
import numpy as np

# define preprocess pipeline, the function arg is multi instances, every instance is tuple of inputs
# this example has one input and one output
def add_trans_datatype(instances):
    """preprocess python implement"""
    for instance in instances:
        x1 = instance[0]
        x2 = instance[1]
        yield x1.astype(np.float32), x2.astype(np.float32)


# when with_batch_dim set to False, only support 2x2 add
# when with_batch_dim set to True(default), support Nx2 add, while N is view as batch
# float32 inputs/outputs
register.declare_servable(servable_file="tensor_add.mindir", model_format="MindIR", with_batch_dim=False)


# register add_common method in add
@register.register_method(output_names=["y"])
def add_common(x1, x2):  # only support float32 inputs
    """method add_common data flow definition, only call model servable"""
    y = register.call_servable(x1, x2)
    return y


# register add_cast method in add
@register.register_method(output_names=["y"])
def add_cast(x1, x2):
    """method add_cast data flow definition, only call preprocess and model servable"""
    x1, x2 = register.call_preprocess(add_trans_datatype, x1, x2)  # cast input to float32
    y = register.call_servable(x1, x2)
    return y
```

该文件定义了`add_common`和`add_cast`两个方法。

MindSpore Serving提供两种部署方式，轻量级部署和集群部署，用户可根据需要进行选择部署。

#### 轻量级部署

服务端调用Python接口直接启动推理进程（master和worker共进程），客户端直接连接推理服务后下发推理任务。
执行[master_with_worker.py](https://gitee.com/mindspore/serving/blob/master/mindspore_serving/example/add/master_with_worker.py)，完成轻量级部署服务如下：

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
from mindspore_serving import master

def start():
    servable_dir = os.path.abspath(".")
    worker.start_servable(servable_dir, "add", device_id=0,
                          master_ip="127.0.0.1", master_port=6500,
                          host_ip="127.0.0.1", host_port=6600)

if __name__ == "__main__":
    start()
```

轻量级部署和集群部署除了master和woker进程是否隔离，worker使用的接口也不同，轻量级部署使用worker的`start_servable_in_master`接口，集群部署使用worker的`start_servable`接口。

### 执行推理

使用[client.py](https://gitee.com/mindspore/serving/blob/master/mindspore_serving/example/add/client.py)，启动Python客户端。

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

使用`mindspore_serving.client`定义的`Client`类，分别调用模型的两个方法，显示如下返回值说明Serving服务已正确执行Add网络的推理。

```shell
[{'y': array([[2. , 2.],
        [2.,  2.]], dtype=float32)},{'y': array([[4. , 4.],
        [4.,  4.]], dtype=float32)},{'y': array([[6. , 6.],
        [6.,  6.]], dtype=float32)}]
[{'y': array([[2. , 2.],
        [2.,  2.]], dtype=float32)}]
```
