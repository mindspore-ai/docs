# 基于MindSpore Serving部署分布式推理服务

`Linux` `Ascend` `Serving` `中级` `高级`

<a href="https://gitee.com/mindspore/docs/blob/r1.2/tutorials/inference/source_zh_cn/serving_distributed_example.md" target="_blank"><img src="_static/logo_source.png"></a>

## 概述

分布式推理是指推理阶段采用多卡进行推理，针对超大规模神经网络模型参数个数过多、模型无法完全加载至单卡中进行推理的问题，可利用多卡进行分布式推理。本文介绍部署分布式推理服务的流程，与[单卡推理服务](https://www.mindspore.cn/tutorial/inference/zh-CN/r1.2/serving_example.html)部署流程大致相同，可以相互参考。

分布式推理服务的架构如图所示：

![image](images/distributed_servable.png)

master提供客户端访问的接口，管理分布式worker并进行任务管理与分发；分布式worker根据模型配置自动调度agent完成分布式推理；每一个agent包含一个分布式模型的切片，占用一个device，加载模型执行推理。

上图展示了rank_size为16，stage_size为2的场景，每个stage包含8个agent，占用8个device。rank_size表示推理使用的device的个数，stage表示流水线的一段，stage_size表示流水线的段数。分布式worker向agent发送推理请求并从agent获取推理结果。agent之间使用HCCL通信。

当前对分布式模型有以下限制：

- 第一个stage的模型接收相同的输入数据。
- 其他的stage的模型不接收数据。
- 最后一个stage的所有模型都返回相同的数据。
- 仅支持Ascend 910推理。

下面以一个简单的分布式网络MatMul为例，演示部署流程。

### 环境准备

运行示例前，需确保已经正确安装了MindSpore Serving。如果没有，可以参考[MindSpore Serving安装页面](https://gitee.com/mindspore/serving/blob/r1.2/README_CN.md#%E5%AE%89%E8%A3%85)，将MindSpore Serving正确地安装到你的电脑当中，同时参考[MindSpore Serving环境配置页面](https://gitee.com/mindspore/serving/blob/r1.2/README_CN.md#%E9%85%8D%E7%BD%AE%E7%8E%AF%E5%A2%83%E5%8F%98%E9%87%8F)完成环境变量配置。

### 导出分布式模型

导出分布式模型需要的文件可以参考[export_model目录](https://gitee.com/mindspore/serving/tree/r1.2/example/matmul_distributed/export_model)，需要如下文件列表：

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
- `rank_table_8pcs.json`为配置当前多卡环境的组网信息的json文件，可以参考[rank_table](https://www.mindspore.cn/tutorial/training/zh-CN/r1.2/advanced_use/distributed_training_ascend.html#id4)。

使用[net.py](https://gitee.com/mindspore/serving/blob/r1.2/example/matmul_distributed/export_model/net.py)，构造一个包含MatMul、Neg算子的网络。

```python
import numpy as np
from mindspore import Tensor, Parameter, ops
from mindspore.nn import Cell


class Net(Cell):
    def __init__(self, matmul_size, transpose_a=False, transpose_b=False, strategy=None):
        super().__init__()
        matmul_np = np.full(matmul_size, 0.5, dtype=np.float32)
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

使用[distributed_inference.py](https://gitee.com/mindspore/serving/blob/r1.2/example/matmul_distributed/export_model/distributed_inference.py)， 配置分布式模型。可以参考[分布式推理](https://www.mindspore.cn/tutorial/inference/zh-CN/r1.2/multi_platform_inference_ascend_910.html#id1)。

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
    network = Net(matmul_size=(96, 16))
    model = Model(network)
    model.infer_predict_layout(Tensor(predict_data))
    export(model._predict_network, Tensor(predict_data), file_name="matmul", file_format="MINDIR")


def create_predict_data():
    """user-defined predict data"""
    inputs_np = np.random.randn(128, 96).astype(np.float32)
    return Tensor(inputs_np)
```

使用[export_model.sh](https://gitee.com/mindspore/serving/blob/r1.2/example/matmul_distributed/export_model/export_model.sh)，导出分布式模型。执行成功后会在上一级目录创建`model`目录，结构如下：

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

每个`device`目录都包含两个文件`group_config.pb`和`matmul.mindir`，分别表示模型分组配置文件与模型文件。

### 部署分布式推理服务

启动分布式推理服务，可以参考[matmul_distributed](https://gitee.com/mindspore/serving/tree/r1.2/example/matmul_distributed)，需要如下文件列表：

```text
matmul_distributed
├── agent.py
├── master_with_worker.py
├── matmul
│   └── servable_config.py
├── model
└── rank_table_8pcs.json
```

- `model`为存放模型文件的目录。
- `master_with_worker.py`为启动服务脚本。
- `agent.py`为启动agent脚本。
- `servable_config.py`为[模型配置文件](https://www.mindspore.cn/tutorial/inference/zh-CN/r1.2/serving_model.html)，通过`declare_distributed_servable`声明了一个rank_size为8、stage_size为1的分布式模型，同时定义了一个分布式servable的方法`predict`。

模型配置文件内容如下：

```python
from mindspore_serving.worker import distributed
from mindspore_serving.worker import register

distributed.declare_distributed_servable(rank_size=8, stage_size=1, with_batch_dim=False)


@register.register_method(output_names=["y"])
def predict(x):
    y = register.call_servable(x)
    return y
```

#### 启动master与分布式worker

使用[master_with_worker.py](https://gitee.com/mindspore/serving/blob/r1.2/example/matmul_distributed/master_with_worker.py)，调用`start_distributed_servable_in_master`方法部署共进程的master和分布式worker。

```python
import os
import sys
from mindspore_serving import master
from mindspore_serving.worker import distributed


def start():
    servable_dir = os.path.dirname(os.path.realpath(sys.argv[0]))
    distributed.start_distributed_servable_in_master(servable_dir, "matmul",
                                                     rank_table_json_file="rank_table_8pcs.json",
                                                     version_number=1,
                                                     worker_ip="127.0.0.1", worker_port=6200,
                                                     wait_agents_time_in_seconds=0)
    master.start_grpc_server("127.0.0.1", 5500)
    master.start_restful_server("127.0.0.1", 1500)


if __name__ == "__main__":
    start()
```

- `servable_dir`为servable存放的目录。
- `servable_name`为servable的名称，对应一个存放模型配置文件的目录。
- `rank_table_json_file`为配置当前多卡环境的组网信息的json文件。
- `worker_ip` 为分布式worker的ip地址。
- `worker_port`为分布式worker的port。
- `wait_agents_time_in_seconds`设置等待所有agent注册完成的时限，默认为0表示会一直等待。

#### 启动Agent

使用[agent.py](https://gitee.com/mindspore/serving/blob/r1.2/example/matmul_distributed/agent.py)，调用`startup_worker_agents`方法会在当前机器上启动的8个agent进程。agent会从分布式worker获取rank_table，这样agent之间才能利用HCCL进行通信。

```python
from mindspore_serving.worker import distributed


def start_agents():
    """Start all the worker agents in current machine"""
    model_files = []
    group_configs = []
    for i in range(8):
        model_files.append(f"model/device{i}/matmul.mindir")
        group_configs.append(f"model/device{i}/group_config.pb")

    distributed.startup_worker_agents(worker_ip="127.0.0.1", worker_port=6200, model_files=model_files,
                                      group_config_files=group_configs, agent_start_port=7000, agent_ip=None,
                                      rank_start=None)


if __name__ == '__main__':
    start_agents()
```

- `worker_ip`为分布式worker的ip地址。
- `worker_port`为分布式worker的port。
- `model_files`为模型文件路径的列表。
- `group_config_files`为模型分组配置文件路径的列表。
- `agent_start_port`表示agent占用的起始端口，默认为7000。
- `agent_ip`为agent的ip地址，默认为None。agent与分布式worker通信的ip默认会从rank_table获取，如果该ip地址不可用，则需要同时设置`agent_ip`与`rank_start`。
- `rank_start`为当前机器起始的rank_id，默认为None。

### 执行推理

通过gRPC访问推理服务，client需要指定gRPC服务器的ip地址和port。运行[client.py](https://gitee.com/mindspore/serving/blob/r1.2/example/matmul_distributed/client.py)，调用matmul分布式模型的`predict`方法，执行推理。

```python
import numpy as np
from mindspore_serving.client import Client


def run_matmul():
    """Run client of distributed matmul"""
    client = Client("localhost", 5500, "matmul", "predict")
    instance = {"x": np.ones((128, 96), np.float32)}
    result = client.infer(instance)
    print("result:\n", result)


if __name__ == '__main__':
    run_matmul()
```

执行后显示如下返回值，说明Serving分布式推理服务已正确执行MatMul网络的推理：

```text
result:
[{'y': array([[-48., -48., -48., ..., -48., -48., -48.],
      [-48., -48., -48., ..., -48., -48., -48.],
      [-48., -48., -48., ..., -48., -48., -48.],
      ...,
      [-48., -48., -48., ..., -48., -48., -48.],
      [-48., -48., -48., ..., -48., -48., -48.],
      [-48., -48., -48., ..., -48., -48., -48.]], dtype=float32)}]
```

