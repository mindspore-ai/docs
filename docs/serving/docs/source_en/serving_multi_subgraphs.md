# Service Deployment with Multiple Subgraphs and Stateful Model

<a href="https://gitee.com/mindspore/docs/blob/r1.6/docs/serving/docs/source_en/serving_multi_subgraphs.md" target="_blank"><img src="https://gitee.com/mindspore/docs/raw/r1.6/resource/_static/logo_source_en.png"></a>

## Overview

MindSpore allows a model to generate multiple subgraphs. Such a model is generally a stateful model. Multiple subgraps can share weights and cooperate with each other to achieve performance optimization. For example, in the Pengcheng·Pangu Model network scenario,  based on a sentence, a sentence is generated through multiple inferences, where each inference generates one word. Two graphs are generated for different input lengths. The first graph processes the text with variable length for the first time and only needs to be executed one time. The second graph processes the word generated last time and needs to be executed multiple time. Compared with that before optimization, only the first graph is executed multiple times, the inference service performance can be improved by 5 to 6 times.

MindSpore Serving supports scheduling multi-subgraph model, improving the inference service performance in specific scenarios.

The following uses a simple single-chip model scenario as an example to describe the multi-subgraph model deployment process. For the detail about the distributed scenario, see [Pengcheng·Pangu Model model Serving deployment](https://gitee.com/mindspore/models/tree/r1.6/official/nlp/pangu_alpha#serving)

### Environment Preparation

Before running the sample network, ensure that MindSpore Serving has been properly installed and the environment variables are configured. To install and configure MindSpore Serving on your PC, go to the [MindSpore Serving installation page](https://www.mindspore.cn/serving/docs/en/r1.6/serving_install.html).

### Downloading the Example

Please download the [example](https://gitee.com/mindspore/serving/blob/r1.6/example/matmul_multi_subgraphs/) first.

### Exporting a Multi-Graph Model

In the directory `export_model`, use [export_matmul.py](https://gitee.com/mindspore/serving/blob/r1.6/example/matmul_multi_subgraphs/export_model/export_matmul.py) to build a network with the Matmul and ReduceSum operator and export the MindSpore inference deployment model based on two different inputs.

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

To use MindSpore for neural network definition, inherit `mindspore.nn.Cell`. (A `Cell` is a base class of all neural networks.) Define each layer of a neural network in the `__init__` method in advance, and then define the `construct` method to complete the forward construction of the neural network. Use `export` of the `mindspore` module to export the model file.
For more detailed examples, see [Quick Start for Beginners](https://www.mindspore.cn/tutorials/en/r1.6/quick_start.html).

Execute the `export_matmul.py` script to generate the `matmul_0.mindir` and `matmul_1.mindir` files. The inputs shapes of these subgraphs are [128,96] and [8,96].

#### Configuring the Service

Start Serving with the following files:

```text
matmul_multi_subgraphs
├── matmul/
│   │── servable_config.py
│   └── 1/
│       │── matmul_0.mindir
│       └── matmul_1.mindir
└── serving_server.py
```

- `serving_server.py`: Script file for starting the service.
- `matmul`: Model folder, which is named after the model name.
- `matmul_0.mindir` and `matmul_1.mindir`: Model files generated by the network in the previous step, which is stored in folder 1 (the number indicates the version number). Different versions are stored in different folders. The version number must be a string of digits. By default, the latest model file is started.
- [servable_config.py](https://gitee.com/mindspore/serving/blob/r1.6/example/tensor_add/add/servable_config.py): [Model configuration file](https://www.mindspore.cn/serving/docs/en/r1.6/serving_model.html), which defines the model processing functions, in which method `predict` is defined.

Content of the configuration file:

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

If a model is stateful, multiple calls to the model need to performed in a Python function stage to avoid mutual interference between multiple instances. A mutli-subgraph model generally a stateful model.

In this example, the `process` function invokes the two subgraphs through the interface `Model.call`, and the `subgraph` parameter of `Model.call` specifies the graph index, which starts from 0. The number is the sequence number for loading graphs. In a standalone system, the number corresponds to the sequence number in the `model_file` parameter list of the `declare_model` interface. In a distributed system, this number corresponds to the sequence number in the `model_files` parameter list of the `startup_agents` interface.

#### Starting the Service

Run the [serving_server.py](https://gitee.com/mindspore/serving/blob/r1.6/example/matmul_multi_subgraphs/serving_server.py) script to start the Serving server:

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


if __name__ == "__main__":
    start()
```

The above startup script will load and run two inference copies of `matmul` on devices 0 and 1, and the inference requests from the client will be split to the two inference copies.

If the server prints the `Serving gRPC server start success, listening on 127.0.0.1:5500` log, the Serving gRPC service has started successfully and the inference model has already loaded successfully.

### Executing Inference

To access the inference service through gRPC, you need to specify the network address of the gRPC server on the client. Execute [serving_client.py](https://gitee.com/mindspore/serving/blob/r1.6/example/matmul_multi_subgraphs/serving_client.py) to call the `predict` method of the MatMul Servable.

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

If the following information is displayed, the Serving inference service has correctly executed the multi-subgraph inference:

```text
result:
 [{'z': array([6528.， 6528.， 6528.， 6528.， 6528.， 6528.， 6528.， 6528.， 6528.，
       6528.， 6528.， 6528.， 6528.， 6528.， 6528.， 6528.], }]
```
