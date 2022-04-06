# 实现多子图和有状态模型的服务部署

<a href="https://gitee.com/mindspore/docs/blob/master/docs/serving/docs/source_zh_cn/serving_multi_subgraphs.md" target="_blank"><img src="https://mindspore-website.obs.cn-north-4.myhuaweicloud.com/website-images/master/resource/_static/logo_source.png"></a>

## 概述

MindSpore支持一个模型导出生成多张子图，拥有多个子图的模型一般也是有状态的模型，多个子图之间共享权重，通过多个子图配合实现性能优化等目标。例如，在鹏程·盘古模型网络场景，基于一段语句，经过多次推理产生一段语句，其中每次推理产生一个词。不同输入长度将会产生两个图，第一为输入长度为1024的全量输入图，处理首次长度不定文本，只需执行一次，第二图为输入长度为1的增量输入图，处理上一次新增的字，第二个图将执行多次。相对于优化之前仅有全量图执行多次，可实现推理服务性能的5-6倍提升。为此，MindSpore Serving提供了多子图功能，实现多张图之间的调度。

下面以一个简单的单卡模型场景为例，演示多子图模型部署流程，分布式场景可以参考[鹏程·盘古模型模型Serving部署](https://gitee.com/mindspore/models/tree/master/official/nlp/pangu_alpha#serving)。

### 环境准备

运行示例前，需确保已经正确安装了MindSpore Serving，并配置了环境变量。MindSpore Serving安装和配置可以参考[MindSpore Serving安装页面](https://www.mindspore.cn/serving/docs/zh-CN/master/serving_install.html)。

### 下载样例

请先[下载样例](https://gitee.com/mindspore/serving/blob/master/example/matmul_multi_subgraphs/)。

### 导出多图模型

在`export_model`目录下，使用[export_matmul.py](https://gitee.com/mindspore/serving/blob/master/example/matmul_multi_subgraphs/export_model/export_matmul.py)，构造一个包含Matmul和ReduceSum的网络，基于两个不同的输入导出MindSpore推理部署模型。

```python
import os
from shutil import copyfile
import numpy as np
import mindspore.context as context
from mindspore import Tensor, Parameter, ops, export
from mindspore.nn import Cell


class Net(Cell):
    """Net"""

    def __init__(self, matmul_size, init_val, transpose_a=False, transpose_b=False):
        """init"""
        super().__init__()
        matmul_np = np.full(matmul_size, init_val, dtype=np.float32)
        self.matmul_weight = Parameter(Tensor(matmul_np))
        self.matmul = ops.MatMul(transpose_a=transpose_a, transpose_b=transpose_b)
        self.sum = ops.ReduceSum()

    def construct(self, inputs):
        """construct"""
        x = self.matmul(inputs, self.matmul_weight)
        x = self.sum(x, 0)
        return x


def export_net():
    """Export matmul net , and copy output model `matmul_0.mindir` and `matmul_1.mindir` to directory ../matmul/1"""
    context.set_context(mode=context.GRAPH_MODE)
    network = Net(matmul_size=(96, 16), init_val=0.5)
    # subgraph 0: 128,96 matmul 16,96 -> 128,16 reduce sum axis 0-> 16
    predict_data = np.random.randn(128, 96).astype(np.float32)
    # pylint: disable=protected-access
    export(network, Tensor(predict_data), file_name="matmul_0", file_format="MINDIR")

    # subgraph 1: 8,96 matmul 16,96 -> 8,16 reduce sum axis 0-> 16
    predict_data = np.random.randn(8, 96).astype(np.float32)
    # pylint: disable=protected-access
    export(network, Tensor(predict_data), file_name="matmul_1", file_format="MINDIR")

    dst_dir = '../matmul/1'
    try:
        os.mkdir(dst_dir)
    except OSError:
        pass

    dst_file = os.path.join(dst_dir, 'matmul_0.mindir')
    copyfile('matmul_0.mindir', dst_file)
    print("copy matmul_0.mindir to " + dst_dir + " success")

    dst_file = os.path.join(dst_dir, 'matmul_1.mindir')
    copyfile('matmul_1.mindir', dst_file)
    print("copy matmul_1.mindir to " + dst_dir + " success")


if __name__ == "__main__":
    export_net()
```

使用MindSpore定义神经网络需要继承`mindspore.nn.Cell`。`Cell`是所有神经网络的基类。神经网络的各层需要预先在`__init__`方法中定义，然后通过定义`construct`方法来完成神经网络的前向构造。使用`mindspore`模块的`export`即可导出模型文件。
更为详细完整的示例可以参考[初学入门](https://www.mindspore.cn/tutorials/zh-CN/master/beginner/quick_start.html)。

执行`export_matmul.py`脚本，生成`matmul_0.mindir`和`matmul_1.mindir`文件，输入shape分别为[128,96]和[8,96]。

### 部署推理服务

#### 配置服务

启动推理服务，可以参考[matmul_multi_subgraphs](https://gitee.com/mindspore/serving/tree/master/example/matmul_multi_subgraphs)，需要如下文件列表：

```text
matmul_multi_subgraphs
├── matmul/
│   │── servable_config.py
│   └── 1/
│       │── matmul_0.mindir
│       └── matmul_1.mindir
└── serving_server.py
```

- `serving_server.py`为启动服务脚本文件。
- `matmul`为模型文件夹，文件夹名即为模型名。
- `matmul_0.mindir`和`matmul_1.mindir`为上一步网络生成的模型文件，放置在文件夹1下，1为版本号，不同的版本放置在不同的文件夹下，版本号需以纯数字串命名，默认配置下启动最大数值的版本号的模型文件。
- [servable_config.py](https://gitee.com/mindspore/serving/blob/master/example/matmul_multi_subgraphs/matmul/servable_config.py)为[模型配置文件](https://www.mindspore.cn/serving/docs/zh-CN/master/serving_model.html)，其定义了Servable的方法`predict`。

模型配置文件内容如下：

```python
from mindspore_serving.server import distributed
from mindspore_serving.server import register

model = register.declare_model(model_file=["matmul_0.mindir", "matmul_1.mindir"], model_format="MindIR",
                               with_batch_dim=False)

def process(x, y):
    z1 = model.call(x, subgraph=0)  # 128,96 matmul 16,96 -> reduce sum axis 0-> 16
    z2 = model.call(y, subgraph=1)  # 8,96 matmul 16,96 -> reduce sum axis 0-> 16
    return z1 + z2


@register.register_method(output_names=["z"])
def predict(x, y):
    z = register.add_stage(process, x, y, outputs_count=1)
    return z
```

如果模型是有状态的，则需要在一个Python 函数Stage中完成对这个模型的所需要的多次调用，避免多个实例的干扰，多子图的模型一般也是有状态的模型。

例子中，`process`函数中通过`Model.call`接口分别调用两个子图，其中的每个子图也可以调用多次，`subgraph`参数指定图的标号，从0开始，此编号在为图加载的序号，单机场景与`declare_model`接口的`model_file`的参数列表序号对应，分布式场景与`startup_agents`接口的`model_files`的参数列表序号对应。

#### 启动Serving服务器

使用[serving_server.py](https://gitee.com/mindspore/serving/blob/master/example/matmul_multi_subgraphs/serving_server.py)启动Serving服务器。

```python
import os
import sys
from mindspore_serving import server


def start():
    servable_dir = os.path.dirname(os.path.realpath(sys.argv[0]))

    servable_config = server.ServableStartConfig(servable_directory=servable_dir, servable_name="matmul",
                                                 device_ids=(0, 1))
    server.start_servables(servable_config)

    server.start_grpc_server("127.0.0.1:5500")
    server.start_restful_server("127.0.0.1:1500")


if __name__ == "__main__":
    start()
```

上述启动脚本将在设备0和1上共加载和运行两个`matmul`推理副本，来自客户端的推理请求将被切割分流到两个推理副本。

当服务端打印日志`Serving gRPC server start success, listening on 127.0.0.1:1500`时，表示Serving gRPC服务启动成功，推理模型已成功加载。

### 执行推理

通过gRPC访问推理服务，client需要指定gRPC服务器的网络地址。运行[serving_client.py](https://gitee.com/mindspore/serving/blob/master/example/matmul_multi_subgraphs/serving_client.py)，调用matmul Servable的`predict`方法，执行推理。

```python
import numpy as np
from mindspore_serving.client import Client


def run_matmul():
    """Run client of distributed matmul"""
    client = Client("localhost:5500", "matmul", "predict")
    instance = {"x": np.ones((128, 96), np.float32), "y": np.ones((8, 96), np.float32)}
    result = client.infer(instance)
    print("result:\n", result)
    assert "z" in result[0]


if __name__ == '__main__':
    run_matmul()
```

执行后显示如下返回值，说明Serving推理服务已正确执行推理任务：

```text
result:
 [{'z': array([6528.， 6528.， 6528.， 6528.， 6528.， 6528.， 6528.， 6528.， 6528.，
       6528.， 6528.， 6528.， 6528.， 6528.， 6528.， 6528.], }]
```
