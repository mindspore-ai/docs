# Parameter Server

[![View Source On Gitee](https://mindspore-website.obs.cn-north-4.myhuaweicloud.com/website-images/master/resource/_static/logo_source_en.svg)](https://gitee.com/mindspore/docs/blob/master/tutorials/experts/source_en/parallel/parameter_server_training.md)

## Overview

A parameter server is a widely used architecture in distributed training. Compared with the synchronous AllReduce training method, a parameter server has better flexibility, and scalability. Specifically, the parameter server supports both synchronous and asynchronous SGD(Stochastic Gradient Descent) training algorithms. In terms of scalability, model computing and update are separately deployed in the worker and server processes, so that resources of the worker and server can be independently scaled out in horizontally (add or delete resources of the worker and server). In addition, in an environment of a large-scale data center, various failures often occur in a computing device, a network, and a storage device, and consequently some nodes are abnormal. However, in an architecture of a parameter server, such a failure can be relatively easily handled without affecting a training job.

> Hardware platforms supported by the parameter server include Ascend, GPU, and `PyNative` mode is not supported.

Related interfaces:

1. `mindspore.set_ps_context(enable_ps=True)` enables Parameter Server training mode.

    - This interface needs to be called before `mindspore.communication.init()`.
    - If this interface is not called, the following environment variable settings will not take effect.
    - Parameter Server training mode can be turned off by calling `mindspore.reset_ps_context()`.

2. In this training mode, there are two ways to call the interface in order to control whether the training parameters are updated through the Parameter Server, and to control the parameter initialization location:

    - Weights recursive settings in `nn.Cell` via `mindspore.nn.Cell.set_param_ps()`.
    - Setting `mindspore.Parameter` weights via `mindspore.Parameter.set_param_ps()`.
    - The size of individual weights that are set to be updated via Parameter Server must not exceed INT_MAX(2^31 - 1) bytes.
    -The interface `set_param_ps` can take a `bool` type parameter: `init_in_server`, which indicates whether the training parameter is initialized on the Server side. The default value of `init_in_server` is `False`, which means that the training parameter is initialized on the Worker. Currently, only the training parameter `embedding_table` of `EmbeddingLookup` operator is supported to be initialized on the Server side, in order to solve the problem that the initialization of `embedding_table` of very large shape on the Worker leads to insufficient memory, and the `target` attribute of this operator needs to be set to 'CPU'. The training parameters initialized on the Server side will no longer be synchronized to the Worker, and if multi-Server training is involved and CheckPoints are saved, a CheckPoint will be saved for each Server at the end of training.

3. [Optional Configuration] For `embedding_table` of very large shape, since the full amount of `embedding_table` cannot be stored on the device, the [EmbeddingLookup operator](https://www.mindspore.cn/docs/en/master/api_python/nn/mindspore.nn.EmbeddingLookup.html) with `vocab_cache_size` can be configured, which is used to turn on the cache function of `EmbeddingLookup` in Parameter Server training mode. This function uses the `vocab_cache_size` size of `embedding_table` to train on the device, the full amount of `embedding_table` is stored in Server, the `embedding_table` used for the next batch of training is swapped into the cache in advance, and when the cache can't be put into the Server, then the outdated `embedding_table` will be put back to the Server, in order to achieve the purpose of improving the performance of training. After training, CheckPoint can be exported on Server to save the full amount of `embedding_table` after training. Embedding cache supports sparse mode, and you need to set the `sparse` parameter to True for all `EmbeddingLookup` operators that have the cache turned on. The sparse mode will de-emphasize the feature ids of the input in this operator to reduce the amount of computation and communication. Refer to <https://gitee.com/mindspore/models/tree/master/official/recommend/Wide_and_Deep> for detailed network training scripts.

> The `Parameter Server` model does not support control flow for now, so in `train.py`, you need to modify `model = Model(network, loss_fn, optimizer, metrics={"Accuracy": Accuracy()}, amp_level="O2")` to `model = Model(network, loss_fn, optimizer, metrics={"Accuracy": Accuracy()})`, and turn off the mixed precision `amp_level` option to eliminate the effect of control flow.

Relevant environment variable configuration:

MindSpore controls Parameter Server training by reading environment variables, which include the following options (`MS_SCHED_HOST` and `MS_SCHED_PORT` values need to be consistent in all scripts):

```text
export MS_SERVER_NUM=1                # Server number
export MS_WORKER_NUM=1                # Worker number
export MS_SCHED_HOST=XXX.XXX.XXX.XXX  # Scheduler IP address
export MS_SCHED_PORT=XXXX             # Scheduler port
export MS_ROLE=MS_SCHED               # The role of this process: MS_SCHED represents the scheduler, MS_WORKER represents the worker, MS_PSERVER represents the Server
```

For more detailed instructions, see [dynamic cluster environment variables](https://www.mindspore.cn/docs/en/master/note/env_var_list.html#dynamic-networking).

## Basic Principle

In the parameter server implementation of MindSpore, the self-developed communication framework (core) is used as the basic architecture. Based on the remote communication capability provided by the core and abstract Send/Broadcast primitives, the distributed training algorithm of the synchronous SGD is implemented. In addition, with the high-performance collective communication library in Ascend and GPU(HCCL and NCCL), MindSpore also provides the hybrid training mode of parameter server and AllReduce. Some weights can be stored and updated through the parameter server, and other weights are still trained through the AllReduce algorithm.

The ps-lite architecture consists of three independent components: Server, Worker, and Scheduler. Their functions are as follows:

- Server: saves model weights and backward computation gradients, and updates the model using gradients pushed by Workers.

- Worker: performs forward and backward computation on the network. The gradient value for backward computation is uploaded to a server through the `Push` API, and the model updated by the server is downloaded to the worker through the `Pull` API.

- Scheduler: establishes the communication relationship between the Server and Worker.

## Practice

Parameter server supports GPU and Ascend, the following Ascend as an example for operation description:

### Example Code Description

> Download the complete example code: [parameter_server](https://gitee.com/mindspore/docs/tree/master/docs/sample_code/parameter_server).

The directory structure is as follows:

```text
└─ sample_code
    ├─ parameter_server
       ├── train.py
       └── run.sh
    ...
```

`train.py` is the script that defines the network structure and the training process. `run.sh` is the execution script.

### Configuring a Distributed Environment

Specify the run mode, run device, run card number via the context interface. Unlike single-card scripts, parallel scripts also need to specify the parallel mode `parallel_mode`, enable `enable_ps` to turn on the parameter server training mode, and initialize HCCL or NCCL communication via init. The `device_target` is automatically specified as the backend hardware device corresponding to the MindSpore package.

```python
import mindspore as ms
from mindspore.communication import init

ms.set_context(mode=ms.GRAPH_MODE)
ms.set_auto_parallel_context(full_batch=True, parallel_mode=ms.ParallelMode.AUTO_PARALLEL)
ms.set_ps_context(enable_ps=True)
init()
ms.set_seed(1)
```

- `full_batch`: Whether to import the dataset in full. A value of `True` indicates full import with the same data for each card, and must be set to `True` in multi-Worker scenarios.
- `parallel_mode`: Parallel mode. Multi-Worker scenarios need to enable auto-parallel mode, set `parallel_mode` = `ParallelMode.AUTO_PARALLEL`.

### Network Definition

The network definition for parameter server mode is to configure net.set_param_ps() based on single card mode:

```python
from mindspore import nn

class Network(nn.Cell):
    def __init__(self):
        super().__init__()
        self.flatten = nn.Flatten()
        self.fc1 = nn.Dense(28*28, 10, weight_init="normal", bias_init="zeros")
        self.relu = nn.ReLU()
        self.fc2 = nn.Dense(10, 1, weight_init="normal", bias_init="zeros")

    def construct(self, x):
        x = self.flatten(x)
        logits = self.fc2(self.relu(self.fc1(x)))
        return logits

net = Network()
net.set_param_ps()
```

### Loading the Dataset

The dataset is loaded in the same way as the single card model, with the following code:

```python
import os
import mindspore.dataset as ds

def create_dataset(batch_size):
    dataset_path = os.getenv("DATA_PATH")
    dataset = ds.MnistDataset(dataset_path)
    image_transforms = [
        ds.vision.Rescale(1.0 / 255.0, 0),
        ds.vision.Normalize(mean=(0.1307,), std=(0.3081,)),
        ds.vision.HWC2CHW()
    ]
    label_transform = ds.transforms.TypeCast(ms.int32)
    dataset = dataset.map(image_transforms, 'image')
    dataset = dataset.map(label_transform, 'label')
    dataset = dataset.batch(batch_size)
    return dataset

data_set = create_dataset(32)
```

### Training the Network

In this section, the optimizer, loss function and training network are defined. Here the network is defined using a functional writing style and the code is consistent with the single card model:

```python
import mindspore as ms
from mindspore import nn, ops

optimizer = nn.SGD(net.trainable_params(), 1e-2)
loss_fn = nn.MSELoss()

def forward_fn(data, target):
    logits = net(data)
    loss = loss_fn(logits, target)
    return loss, logits

grad_fn = ops.value_and_grad(forward_fn, None, net.trainable_params(), has_aux=True)

@ms.jit
def train_step(inputs, targets):
    (loss_value, _), grads = grad_fn(inputs, targets)
    optimizer(grads)
    return loss_value

for epoch in range(10):
    i = 0
    for image, label in data_set:
        loss_output = train_step(image, label)
        if i % 10 == 0:
            print("epoch: %s, step: %s, loss is %s" % (epoch, i, loss_output))
        i += 1
```

### Running Stand-alone 8-card Script

Next, the corresponding scripts are called by commands to carry out distributed training using the 8-card distributed training script as an example, and the three roles of Scheduler, Server, and Worker start the corresponding number of processes respectively. The commands are as follows:

```bash
EXEC_PATH=$(pwd)

if [ ! -d "${EXEC_PATH}/MNIST_Data" ]; then
    if [ ! -f "${EXEC_PATH}/MNIST_Data.zip" ]; then
        wget http://mindspore-website.obs.cn-north-4.myhuaweicloud.com/notebook/datasets/MNIST_Data.zip
    fi
    unzip MNIST_Data.zip
fi
export DATA_PATH=${EXEC_PATH}/MNIST_Data/train/

rm -rf output
mkdir output

# run Scheduler process
export MS_SERVER_NUM=8
export MS_WORKER_NUM=8
export MS_SCHED_HOST=127.0.0.1
export MS_SCHED_PORT=8118
export MS_ROLE=MS_SCHED
python train.py > output/scheduler.log 2>&1 &

# run Server processes
export MS_SERVER_NUM=8
export MS_WORKER_NUM=8
export MS_SCHED_HOST=127.0.0.1
export MS_SCHED_PORT=8118
export MS_ROLE=MS_PSERVER
for((server_id=0;server_id<${MS_SERVER_NUM};server_id++))
do
    python train.py > output/server_${server_id}.log 2>&1 &
done

# run Wroker processes
export MS_SERVER_NUM=8
export MS_WORKER_NUM=8
export MS_SCHED_HOST=127.0.0.1
export MS_SCHED_PORT=8118
export MS_ROLE=MS_WORKER
for((worker_id=0;worker_id<${MS_WORKER_NUM};worker_id++))
do
    python train.py > output/worker_${worker_id}.log 2>&1 &
done
```

Or execute directly:

```bash
bash run.sh
```

The output of each process is saved in the `output` folder, and you can view the log of Server and Worker communication in `output/scheduler.log`:

```text
...
Assign rank id of node id: 2fa9d1ab-10b8-4a61-9acf-217a04439287, role: MS_WORKER, with host ip: 127.0.0.1, old rank id: 6, new rank id: 0
...
Assign rank id of node id: 02fb1169-edc3-465e-b307-ccaf62d1f0b3, role: MS_PSERVER, with host ip: 127.0.0.1, old rank id: 4, new rank id: 0
...
Cluster is successfully initialized.
```

The training results are saved in `output/worker_0.log` as shown in the example below:

```text
epoch: 0, step: 0, loss is 26.743706
epoch: 0, step: 10, loss is 17.507723
epoch: 0, step: 20, loss is 9.616591
epoch: 0, step: 30, loss is 8.589715
epoch: 0, step: 40, loss is 8.23479
epoch: 0, step: 50, loss is 10.431321
epoch: 0, step: 60, loss is 7.7080607
epoch: 0, step: 70, loss is 8.599786
epoch: 0, step: 80, loss is 7.669814
epoch: 0, step: 90, loss is 8.584343
epoch: 0, step: 100, loss is 8.803712
```
