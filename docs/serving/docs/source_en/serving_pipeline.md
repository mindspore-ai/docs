# Scheduling Multi-Graph Based on the Pipeline

`Linux` `Ascend` `Serving` `Intermediate` `Expert`

[![View Source On Gitee](https://gitee.com/mindspore/docs/raw/r1.3/resource/_static/logo_source.png)](https://gitee.com/mindspore/docs/blob/r1.3/docs/serving/docs/source_en/serving_pipeline.md)

## Overview

MindSpore allows a model to generate multiple subgraphs and scheduling these subgraphs can improve the performance. For example, graphs corresponding to two phases are split in the GPT3 scenario. The initialization graph is at the first phase, which needs to be executed only once. The inference graph at the second phase needs to be executed multiple times based on the input sentence length N. Before the optimization, the two graphs were combined N times. Now, the performance of the inference service is improved by 5 to 6 times. MindSpore Serving provides the pineline function to schedule multiple graphs, improving the inference service performance in specific scenarios.

Currently, the pipeline has the following restrictions:

- Only the scenario where batchsize is set to 1 is supported.
- If a pipeline exists, the service is presented as a pipeline. That is, the model method called by the client must be the registered pipeline method.

The following uses a distributed scenario as an example to describe the pipeline deployment process.

### Environment Preparation

Before running the sample network, ensure that MindSpore Serving has been properly installed and the environment variables are configured. To install and configure MindSpore Serving on your PC, go to the [MindSpore Serving installation page](https://www.mindspore.cn/serving/docs/en/r1.3/serving_install.html).

### Exporting a Multi-Graph Model

For details about the files required for exporting a distributed model, see [export_model directory](https://gitee.com/mindspore/serving/tree/r1.3/example/pipeline_distributed/export_model). The following files are required:

```text
export_model
├── distributed_inference.py
├── export_model.sh
├── net.py
└── rank_table_8pcs.json
```

- `net.py` is the definition of the MatMul network.
- `distributed_inference.py` is used to configure distributed parameters.
- `export_model.sh` creates the `device` directory on the current machine and exports the model file corresponding to each `device`.
- `rank_table_8pcs.json` is the JSON file for configuring the networking information of the current multi-device environment. For details, see [rank_table](https://gitee.com/mindspore/mindspore/tree/r1.3/model_zoo/utils/hccl_tools).

Use [net.py](https://gitee.com/mindspore/serving/blob/r1.3/example/matmul_distributed/export_model/net.py) to build a network that contains MatMul and Neg operators.

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

Use [distributed_inference.py](https://gitee.com/mindspore/serving/blob/r1.3/example/pipeline_distributed/export_model/distributed_inference.py) to generate a multi-graph model. For details, see [Distributed Inference](https://www.mindspore.cn/docs/programming_guide/en/r1.3/distributed_inference.html).

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

Use [export_model.sh](https://gitee.com/mindspore/serving/blob/r1.3/example/matmul_distributed/export_model/export_model.sh) to export a multi-graph model. After the script is executed successfully, the `model` directory is created in the upper-level directory. The structure is as follows:

```text
model
├── device0
│   ├── group_config.pb
│   └── matmul.mindir
├── device1
├── device2
├── device3
├── device4
├── device5
├── device6
└── device7
```

Each `device` directory contains two files `group_config.pb` (the model group configuration file) and `matmul_0.mindir` or `matmul_1.mindir` (the model files corresponding to the two graphs, respectively).

### Deploying the Distributed Inference Service

Start the distributed inference service. For details, see [pipeline_distributed](https://gitee.com/mindspore/serving/tree/r1.3/example/pipeline_distributed). The following files are required:

```text
matmul_distributed
├── serving_agent.py
├── serving_server.py
├── matmul
│   └── servable_config.py
├── model
└── rank_table_8pcs.json
```

- `model` is the directory for storing model files.
- `serving_server.py` is used to start service processes, including the `Main` and `Distributed Worker` processes.
- `serving_agent.py` is used to start the `Agent`.
- `servable_config.py` is the [model configuration file](https://www.mindspore.cn/serving/docs/en/r1.3/serving_model.html). It uses `distributed.declare_servable` to declare a distributed model whose rank_size is 8 and stage_size is 1, and defines a pipeline method `predict`.

Content of the configuration file:

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

The `subgraph` parameter of the `call_servable` method specifies the graph number, which starts from 0. The number is the sequence number for loading graphs. In a standalone system, the number corresponds to the sequence number in the `servable_file` parameter list of the `declare_servable` interface. In a distributed system, this number corresponds to the sequence number in the `model_files` parameter list of the `startup_agents` interface.
The `PipelineServable` class declares the service function of the model, `servable_name` specifies the model name, `method` specifies the function method, `version_number` specifies the version number, `register_pipeline` registers the pipeline function, and the input parameter `output_names` specifies the output list.

#### Starting the Serving Server

Use [serving_server.py](https://gitee.com/mindspore/serving/blob/r1.3/example/pipeline_distributed/serving_server.py) to call the `distributed.start_servable` method to deploy the distributed Serving server.

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

- `servable_dir` is the servable directory.
- `servable_name` indicates the servable name, which corresponds to a directory for storing the model configuration file.
- `rank_table_json_file` is the JSON file for configuring the network information in the multi-device environment.
- `distributed_address` is the `Distributed Worker` address.
- `wait_agents_time_in_seconds` specifies the time limit for waiting for the completion of all `Agent` registrations. The default value is 0, indicating that the system keeps waiting for the completion of all `Agent` registrations.

#### Starting the Agent

Use [serving_agent.py](https://gitee.com/mindspore/serving/blob/r1.3/example/pipeline_distributed/serving_agent.py) to call the `startup_agents` method to start the eight `Agent` processes on the current machine. `Agent` obtains rank_table from `Distributed Worker` so that `Agents` can communicate with each other using HCCL.

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

- `distributed_address` is the `Distributed Worker` address.
- `model_files` is a list of model file paths. Inputting multiple model files indicates that multiple graphs are supported. The file transfer sequence number determines the graph number corresponding to the `subgraph` parameter of the `call_servable` method.
- `group_config_files` is the list of model group configuration file paths.
- `agent_start_port` is the start port occupied by the `Agent`. The default value is 7000.
- `agent_ip` is the IP address of the `Agent`. The default value is None. By default, the IP address used for the communication between the `Agent` and `Distributed Worker` is obtained from the rank_table. If the IP address is unavailable, you need to set both `agent_ip` and `rank_start`.
- `rank_start` is the start rank_id of the current machine. The default value is None.

### Executing Inference

To access the inference service through gRPC, you need to specify the IP address and port number of the gRPC server on the client. Execute [serving_client.py](https://gitee.com/mindspore/serving/blob/r1.3/example/pipeline_distributed/serving_client.py) to call the `predict` method of the MatMul distributed model. This method corresponds to the registered pipeline method and is used to perform inference.

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

If the following information is displayed, the Serving distributed inference service has correctly executed the multi-graph inference of the pipeline:

```text
result:
[{'x': array([[-96., -96., -96., ..., -96., -96., -96.],
      [-96., -96., -96., ..., -96., -96., -96.],
      [-96., -96., -96., ..., -96., -96., -96.],
      ...,
      [-96., -96., -96., ..., -96., -96., -96.],
      [-96., -96., -96., ..., -96., -96., -96.],
      [-96., -96., -96., ..., -96., -96., -96.]], dtype=float32), 'z': array([[-48., -48., -48., ..., -48., -48., -48.],
      [-48., -48., -48., ..., -48., -48., -48.],
      [-48., -48., -48., ..., -48., -48., -48.],
      ...,
      [-48., -48., -48., ..., -48., -48., -48.],
      [-48., -48., -48., ..., -48., -48., -48.],
      [-48., -48., -48., ..., -48., -48., -48.]], }]
```
