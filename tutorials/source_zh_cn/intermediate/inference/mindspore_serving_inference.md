# 基于MindSpore Serving部署推理服务

<!-- TOC -->

- [基于MindSpore Serving部署推理服务](#基于mindspore-serving部署推理服务)
    - [环境准备](#环境准备)
    - [部署Serving推理服务](#部署serving推理服务)
        - [配置服务](#配置服务)
        - [启动服务](#启动服务)
    - [执行推理](#执行推理)

<!-- /TOC -->

<a href="https://gitee.com/mindspore/docs/blob/master/tutorials/source_zh_cn/intermediate/inference/mindspore_serving_inference.md" target="_blank"><img src="https://gitee.com/mindspore/docs/raw/master/resource/_static/logo_source.png"></a>

MindSpore Serving是一个轻量级、高性能的推理服务模块，旨在帮助MindSpore开发者在生产环境中高效部署在线推理服务。当用户使用MindSpore完成模型训练后，导出MindSpore模型，即可使用MindSpore Serving创建该模型的推理服务。

MindSpore Serving提供如下功能：

- 加载模型文件生成推理引擎，提供推理功能；
- 预测请求和处理结果的消息交互，支持gPRC和RESTful两种请求方式；
- 预测接口调用，执行预测，返回预测结果；
- 模型的生命周期管理；
- 服务的生命周期管理；
- 多模型多版本的管理。

本文以一个简单的Add网络为例，演示MindSpore Serving的基础使用方法。可通过链接查看[Add网络推理源码](https://gitee.com/mindspore/serving/tree/master/example/tensor_add)。

## 环境准备

用户在使用MindSpore Serving前，可参考[安装指南](https://gitee.com/mindspore/serving/blob/master/README_CN.md#%E5%AE%89%E8%A3%85)，与[MindSpore Serving环境变量配置](https://gitee.com/mindspore/serving/blob/master/README_CN.md#%E9%85%8D%E7%BD%AE%E7%8E%AF%E5%A2%83%E5%8F%98%E9%87%8F)部署MindSpore Serving环境。

## 部署Serving推理服务

### 配置服务

以Add用例为例，启动Serving服务需要如下文件：

```shell
tensor_add
├── add/
│    └── servable_config.py
│    └── 1/
│        └── tensor_add.mindir
└── serving_server.py
```

- `tensor_add.mindir`为模型文件，放置在文件夹1下，1为版本号。不同的版本放置在不同的文件夹下，版本号需以纯数字串命名，默认配置下启动最大数值的版本号的模型文件。
- `servable_config.py`为模型配置文件，定义了模型的处理函数，包括`add_common`和`add_cast`两个方法，`add_common`定义了输入为两个普通float32类型的加法操作，`add_cast`定义输入类型为其他类型，经过输入类型转换float32后的加法操作。

模型配置文件内容如下：

```python
import numpy as np
from mindspore_serving.server import register


def add_trans_datatype(x1, x2):
    """预处理定义，本例中有两个输入和输出"""
    return x1.astype(np.float32), x2.astype(np.float32)


# 进行模型声明，其中declare_model入参model_file指示模型的文件名称，model_format指示模型的模型类别
# 当with_batch_dim设定为False时, 仅支持2x2的Tensor
# 当with_batch_dim设定为True时, 可支持Nx2的Tensor, N的值由batch决定
model = register.declare_model(model_file="tensor_add.mindir", model_format="MindIR", with_batch_dim=False)

# add_common方法定义
# Servable方法的入参由Python方法的入参指定，Servable方法的出参由register_method的output_names指定
@register.register_method(output_names=["y"])
def add_common(x1, x2):  # 仅支持float32类型的输入
    """add_common数据流定义，只调用模型推理"""
    y = register.add_stage(model, x1, x2, outputs_count=1)
    return y


# add_cast方法定义
@register.register_method(output_names=["y"])
def add_cast(x1, x2):
    """add_cast数据流定义，调用预处理和模型推理"""
    x1, x2 = register.add_stage(add_trans_datatype, x1, x2, outputs_count=2)  # 将输入转换为 float32
    y = register.add_stage(model, x1, x2, outputs_count=1)
    return y
```

### 启动服务

执行[serving_server.py](https://gitee.com/mindspore/serving/blob/master/example/tensor_add/serving_server.py)，完成服务启动：

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

## 执行推理

客户端提供两种方式访问推理服务，一种是通过[gRPC方式](https://www.mindspore.cn/serving/docs/en/master/serving_grpc.html)，一种是通过[RESTful方式](https://www.mindspore.cn/serving/docs/zh-CN/master/serving_restful.html)。本文以gRPC方式为例，通过`client.py`执行推理。

```python
import numpy as np
from mindspore_serving.client import Client


def run_add_common():
    """调用add add_common"""
    client = Client("localhost", 5500, "add", "add_common")
    instances = []

    # 例1
    x1 = np.asarray([[1, 1], [1, 1]]).astype(np.float32)
    x2 = np.asarray([[1, 1], [1, 1]]).astype(np.float32)
    instances.append({"x1": x1, "x2": x2})

    # 例2
    x1 = np.asarray([[2, 2], [2, 2]]).astype(np.float32)
    x2 = np.asarray([[2, 2], [2, 2]]).astype(np.float32)
    instances.append({"x1": x1, "x2": x2})

    # 例3
    x1 = np.asarray([[3, 3], [3, 3]]).astype(np.float32)
    x2 = np.asarray([[3, 3], [3, 3]]).astype(np.float32)
    instances.append({"x1": x1, "x2": x2})

    result = client.infer(instances)
    print(result)


def run_add_cast():
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
