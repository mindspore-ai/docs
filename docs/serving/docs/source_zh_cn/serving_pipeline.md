# 基于Pipeline实现多图调度

`Linux` `Ascend` `Serving` `中级` `高级`

[![查看源文件](https://gitee.com/mindspore/docs/raw/r1.3/resource/_static/logo_source.png)](https://gitee.com/mindspore/docs/blob/r1.3/docs/serving/docs/source_zh_cn/serving_pipeline.md)

## 概述

MindSpore支持一个模型可以生成多张子图，通过调度多张子图实现性能的提升。例如，在GPT3场景，会将模型拆成两个阶段的图，第一阶段为初始化图，只需执行一次，第二阶段的推理图，根据输入句子长度N决定需要执行N次。这样，相对于优化之前两张图合并在一起都需要执行N次，实现了推理服务性能的5-6倍提升。为此，MindSpore Serving提供了Pineline功能，实现多张图之间的调度，提升特定场景的推理服务性能。

当前对Pipeline使用有以下限制：

- 当前仅支持batchsize为1场景。
- 有Pipeline存在场景，服务仅是以Pineline的形式对外体现，即客户端调用的模型方法需为注册的Pipeline方法。

下面以一个简单的分布式场景为例，演示Pipeline部署流程。

### 环境准备

运行示例前，需确保已经正确安装了MindSpore Serving，并配置了环境变量。MindSpore Serving和安装和配置可以参考[MindSpore Serving安装页面](https://www.mindspore.cn/serving/docs/zh-CN/r1.3/serving_install.html)。

### 导出多图模型

导出分布式模型需要的文件可以参考[export_model目录](https://gitee.com/mindspore/serving/tree/r1.3/example/pipeline_distributed/export_model)，需要如下文件列表：

```text
export_model
├── distributed_inference.py
├── export_model.sh
├── net.py
└── rank_table_8pcs.json
```

- `net.py`为MatMul网络定义。
- `distributed_inference.py`配置分布式相关的参数。
- `export_model.sh`在当前机器上创建`device`目录并且导出每个`device`对应的模型文件。
- `rank_table_8pcs.json`为配置当前多卡环境的组网信息的json文件，可以参考[rank_table](https://gitee.com/mindspore/mindspore/tree/r1.3/model_zoo/utils/hccl_tools)。

使用[net.py](https://gitee.com/mindspore/serving/blob/r1.3/example/matmul_distributed/export_model/net.py)，构造一个包含MatMul、Neg算子的网络。

```python
import numpy as np
from mindspore import Tensor, Parameter, ops
from mindspore.nn import Cell


class Net(Cell):
    def __init__(self, matmul_size, init_val, transpose_a=False, transpose_b=False, strategy=None):
        super().__init__()
        matmul_np = np.full(matmul_size, init_val, dtype=np.float32)
        self.matmul_weight = Parameter(Tensor(matmul_np))
        self.matmul = ops.MatMul(transpose_a=transpose_a, transpose_b=transpose_b)
        self.neg = ops.Neg()
        if strategy is not None:
            self.matmul.shard(strategy)

    def construct(self, inputs):
        x = self.matmul(inputs, self.matmul_weight)
        x = self.neg(x)
        return x
```

使用[distributed_inference.py](https://gitee.com/mindspore/serving/blob/r1.3/example/pipeline_distributed/export_model/distributed_inference.py)，生成多图模型。可以参考[分布式推理](https://www.mindspore.cn/docs/programming_guide/zh-CN/r1.3/distributed_inference.html)。

```python
import numpy as np
from net import Net
from mindspore import context, Model, Tensor, export
from mindspore.communication import init


def test_inference():
    """distributed inference after distributed training"""
    context.set_context(mode=context.GRAPH_MODE)
    init(backend_name="hccl")
    context.set_auto_parallel_context(full_batch=True, parallel_mode="semi_auto_parallel",
                                      device_num=8, group_ckpt_save_file="./group_config.pb")

    predict_data = create_predict_data()
    network = Net(matmul_size=(96, 16), init_val = 0.5)
    model = Model(network)
    model.infer_predict_layout(Tensor(predict_data))
    export(model._predict_network, Tensor(predict_data), file_name="matmul_0", file_format="MINDIR")

    network_1 = Net(matmul_size=(96, 16), init_val = 1.5)
    model_1 = Model(network)
    model_1.infer_predict_layout(Tensor(predict_data))
    export(model_1._predict_network, Tensor(predict_data), file_name="matmul_1", file_format="MINDIR")


def create_predict_data():
    """user-defined predict data"""
    inputs_np = np.random.randn(128, 96).astype(np.float32)
    return Tensor(inputs_np)
```

使用[export_model.sh](https://gitee.com/mindspore/serving/blob/r1.3/example/matmul_distributed/export_model/export_model.sh)，导出多图模型。执行成功后会在上一级目录创建`model`目录，结构如下：

```text
model
├── device0
│   ├── group_config.pb
│   └── matmul.mindir
├── device1
├── device2
├── device3
├── device4
├── device5
├── device6
└── device7
```

每个`device`目录都包含两个文件`group_config.pb`、`matmul_0.mindir`和`matmul_1.mindir`，分别表示模型分组配置文件与两张图对应的模型文件。

### 部署分布式推理服务

启动分布式推理服务，可以参考[pipeline_distributed](https://gitee.com/mindspore/serving/tree/r1.3/example/pipeline_distributed)，需要如下文件列表：

```text
matmul_distributed
├── serving_agent.py
├── serving_server.py
├── matmul
│   └── servable_config.py
├── model
└── rank_table_8pcs.json
```

- `model`为存放模型文件的目录。
- `serving_server.py`为启动服务脚本，包括`Main`和`Distributed Worker`进程。
- `serving_agent.py`为启动`Agent`脚本。
- `servable_config.py`为[模型配置文件](https://www.mindspore.cn/serving/docs/zh-CN/r1.3/serving_model.html)，通过`distributed.declare_servable`声明了一个rank_size为8、stage_size为1的分布式模型，同时定义了一个Pipeline的方法`predict`。

模型配置文件内容如下：

```python
import numpy as np
from mindspore_serving.server import distributed
from mindspore_serving.server import register
from mindspore_serving.server.register import PipelineServable

distributed.declare_servable(rank_size=8, stage_size=1, with_batch_dim=False)

def add_preprocess(x):
    """define preprocess, this example has one input and one output"""
    x = np.add(x, x)
    return x

@register.register_method(output_names=["y"])
def fun1(x):
    x = register.call_preprocess(add_preprocess, x)
    y = register.call_servable(x, subgraph=0)
    return y

@register.register_method(output_names=["y"])
def fun2(x):
    y = register.call_servable(x, subgraph=1)
    return y

servable1 = PipelineServable(servable_name="matmul", method="fun1", version_number=0)
servable2 = PipelineServable(servable_name="matmul", method="fun2", version_number=0)

@register.register_pipeline(output_names=["x", "z"])
def predict(x, y):
    x = servable1.run(x)
    for i in range(10):
        print(i)
        z = servable2.run(y)
    return x, z

```

其中，`call_servable`方法的`subgraph`参数指定图的标号，从0开始，此编号在为图加载的序号，单机场景与`declare_servable`接口的`servable_file`的参数列表序号对应，分布式场景与`startup_agents`接口的`model_files`的参数列表序号对应。
`PipelineServable`类声明模型的服务函数，`servable_name`指定模型名, `method`指定函数方法, `version_number`指定版本号，`register_pipeline`实现对Pipeline函数的注册，入参output_names指定输出列表。

#### 启动Serving服务器

使用[serving_server.py](https://gitee.com/mindspore/serving/blob/r1.3/example/pipeline_distributed/serving_server.py)，调用`distributed.start_servable`方法部署分布式Serving服务器。

```python
import os
import sys
from mindspore_serving import server
from mindspore_serving.server import distributed


def start():
    servable_dir = os.path.dirname(os.path.realpath(sys.argv[0]))
    distributed.start_servable(servable_dir, "matmul",
                               rank_table_json_file="rank_table_8pcs.json",
                               version_number=1,
                               distributed_address="127.0.0.1:6200")

    server.start_grpc_server("127.0.0.1:5500")
    server.start_restful_server("127.0.0.1:1500")


if __name__ == "__main__":
    start()
```

- `servable_dir`为servable存放的目录。
- `servable_name`为servable的名称，对应一个存放模型配置文件的目录。
- `rank_table_json_file`为配置当前多卡环境的组网信息的json文件。
- `distributed_address`为`Distributed Worker`的地址。
- `wait_agents_time_in_seconds`设置等待所有`Agent`注册完成的时限，默认为0表示会一直等待。

#### 启动Agent

使用[serving_agent.py](https://gitee.com/mindspore/serving/blob/r1.3/example/pipeline_distributed/serving_agent.py)，调用`startup_agents`方法会在当前机器上启动的8个`Agent`进程。`Agent`会从`Distributed Worker`获取rank_table，这样`Agent`之间才能利用HCCL进行通信。

```python
from mindspore_serving.server import distributed


def start_agents():
    """Start all the agents in current machine"""
    model_files = []
    group_configs = []
    for i in range(8):
        model_files.append([f"model/device{i}/matmul_0.mindir", f"model/device{i}/matmul_1.mindir"])
        group_configs.append([f"model/device{i}/group_config.pb"])

    distributed.startup_agents(distributed_address="127.0.0.1:6200", model_files=model_files,
                               group_config_files=group_configs)

if __name__ == '__main__':
    start_agents()

```

- `distributed_address`为`Distributed Worker`的地址。
- `model_files`为模型文件路径的列表，传入多个模型文件表示支持多图，文件传输顺序号决定`call_servable`方法的`subgraph`参数对应的图号。
- `group_config_files`为模型分组配置文件路径的列表。
- `agent_start_port`表示`Agent`占用的起始端口，默认为7000。
- `agent_ip`为`Agent`的ip地址，默认为None。`Agent`与`Distributed Worker`通信的ip默认会从rank_table获取，如果该ip地址不可用，则需要同时设置`agent_ip`与`rank_start`。
- `rank_start`为当前机器起始的rank_id，默认为None。

### 执行推理

通过gRPC访问推理服务，client需要指定gRPC服务器的ip地址和port。运行[serving_client.py](https://gitee.com/mindspore/serving/blob/r1.3/example/pipeline_distributed/serving_client.py)，调用matmul分布式模型的`predict`方法，该方法对应注册的pipeline方法，执行推理。

```python
import numpy as np
from mindspore_serving.client import Client


def run_matmul():
    """Run client of distributed matmul"""
    client = Client("localhost:5500", "matmul", "predict")
    instance = {"x": np.ones((128, 96), np.float32), "y": np.ones((128, 96), np.float32)}
    result = client.infer(instance)
    print("result:\n", result)

if __name__ == '__main__':
    run_matmul()
```

执行后显示如下返回值，说明Serving分布式推理服务已正确执行Pipeline的多图推理：

```text
result:
[{'x': array([[-96., -96., -96., ..., -96., -96., -96.],
      [-96., -96., -96., ..., -96., -96., -96.],
      [-96., -96., -96., ..., -96., -96., -96.],
      ...,
      [-96., -96., -96., ..., -96., -96., -96.],
      [-96., -96., -96., ..., -96., -96., -96.],
      [-96., -96., -96., ..., -96., -96., -96.]], dtype=float32)，'z': array([[-48., -48., -48., ..., -48., -48., -48.],
      [-48., -48., -48., ..., -48., -48., -48.],
      [-48., -48., -48., ..., -48., -48., -48.],
      ...,
      [-48., -48., -48., ..., -48., -48., -48.],
      [-48., -48., -48., ..., -48., -48., -48.],
      [-48., -48., -48., ..., -48., -48., -48.]], }]
```
