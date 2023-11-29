# Model Loading

[![View Source On Gitee](https://mindspore-website.obs.cn-north-4.myhuaweicloud.com/website-images/r2.3/resource/_static/logo_source_en.svg)](https://gitee.com/mindspore/docs/blob/r2.3/tutorials/experts/source_en/parallel/model_loading.md)

## Overview

Model loading under distribution mainly refers to distributed inference, i.e., the inference phase is performed using multi-card. If data parallel is used for training or model parameters are saved in merge, then each card holds full weights and each card infers about its own input data in the same way as [single-card inference](https://www.mindspore.cn/tutorials/experts/en/r2.3/infer/inference.html#model-eval-model-validation). It should be noted that each card loads the same CheckPoint file for inference.
This tutorial focuses on the process of saving slices of the model on each card during the multi-card training process, and reloading the model for inference according to the inference strategy in the inference phase using the multi-card format. For the problem that the number of parameters in the ultra-large-scale neural network model is too large and the model cannot be fully loaded into single card for inference, distributed inference can be performed using multiple cards.

> - When the model is very large and the [load_distributed_checkpoint](https://www.mindspore.cn/docs/en/r2.3/api_python/mindspore/mindspore.load_distributed_checkpoint.html) interface that is used in this tutorials has insufficient host memory, refer to the [model transformation](https://www.mindspore.cn/tutorials/experts/en/r2.3/parallel/model_transformation.html) section to use the way that each card loads its own corresponding sliced checkpoint.
> - If pipeline distributed inference is used, pipeline parallel training must also be used during training, and the `device_num` and `pipeline_stages` used for pipeline parallel training and inference must be the same. For pipeline parallel inference, `micro_batch` is 1, there is no need to call `PipelineCell`, and each `stage` only needs to load the Checkpoint file for this `stage`. Refer to the [Pipeline Parallel](https://www.mindspore.cn/tutorials/experts/en/r2.3/parallel/pipeline_parallel.html) training tutorial.

Related interfaces:

1. `mindspore.set_auto_parallel_context(full_batch=True, parallel_mode=ParallelMode.SEMI_AUTO_PARALLEL)`: Set the parallel configuration, where `full_batch` indicates whether the dataset is imported in full, and `True` indicates a full import with the same data for each card, which must be set to `True` in this scenario. `parallel_mode` is the parallel mode, which must be set to auto-parallel or semi-auto-parallel in this scenario.

2. `mindspore.set_auto_parallel_context(strategy_ckpt_config=strategy_ckpt_dict)`: The configuration used to set the parallel strategy file. `strategy_ckpt_dict` is used to set the configuration of the parallel strategy file and is of dictionary type. strategy_ckpt_dict = {"load_file": ". /stra0.ckpt", "save_file": ". /stra1.ckpt", "only_trainable_params": False}, where:
    - `load_file(str)`: Path to load the parallel sharding strategy. File address of the strategy file generated in the training phase. This parameter must be set in distributed inference scenarios. Default: "".
    - `save_file(str)`: The path where the parallel sharding strategy is saved. Default: "".
    - `only_trainable_params(bool)`: Save/load strategy information for trainable parameters only. Default: `True`.

3. `model.infer_predict_layout(predict_data)`: Generate inference strategies based on inference data.

4. `model.infer_train_layout(data_set)`: Generate training strategies based on training data.

5. `mindspore.load_distributed_checkpoint(network, checkpoint_filenames, predict_strategy)`: This interface merges the model slices and then slices them according to the inference strategy and loads them into the network, where `network` is the network structure and `checkpoint_filenames` is a list consisting of the names of the checkpoint files, in rank id order. `predict_strategy` is the inference strategy exported by `model.infer_predict_layout`.

## Operation Practice

The following is an illustration of distributed inference, using a single-machine 8-card as an example.

### Example Code Description

> Download the complete example code: [model_saving_loading](https://gitee.com/mindspore/docs/tree/r2.3/docs/sample_code/model_saving_loading).

The directory structure is as follows:

```text
└─ sample_code
    ├─ model_saving_loading
       ├── test_loading.py
       ├── run_loading.sh
       ...
    ...
```

`test_loading.py` is the script that defines the network structure and inference. `run_loading.sh` is the execution script.

The user first needs to follow the [Model Saving](https://www.mindspore.cn/tutorials/experts/en/r2.3/parallel/model_saving.html) tutorial to perform the 8-card distributed training, which will generate Checkpoint file directory as well as the cut-point strategy file in the current path at the end of training:

```text
src_checkpoints/
src_strategy.ckpt
```

### Configuring a Distributed Environment

Specify the run mode, run device, run card number via the context interface. Unlike single card scripts, parallel scripts also need to specify the parallel mode `parallel_mode` as semi-parallel mode. Configure the path to the distributed strategy file to be loaded via `strategy_ckpt_config` and initialize HCCL or NCCL communication via init. The `device_target` is automatically specified as the backend hardware device corresponding to the MindSpore package.

```python
import mindspore as ms
from mindspore.communication import init

ms.set_context(mode=ms.GRAPH_MODE)
ms.set_auto_parallel_context(parallel_mode=ms.ParallelMode.SEMI_AUTO_PARALLEL)
ms.set_auto_parallel_context(strategy_ckpt_config={"load_file": "./src_strategy.ckpt"})
init()
ms.set_seed(1)
```

### Defining the Network

The inference network definition needs to be the same as the training network:

```python
from mindspore import nn, ops
from mindspore.common.initializer import initializer

class Dense(nn.Cell):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.weight = ms.Parameter(initializer("normal", [in_channels, out_channels], ms.float32))
        self.bias = ms.Parameter(initializer("normal", [out_channels], ms.float32))
        self.matmul = ops.MatMul()
        self.add = ops.Add()

    def construct(self, x):
        x = self.matmul(x, self.weight)
        x = self.add(x, self.bias)
        return x

class Network(nn.Cell):
    def __init__(self):
        super().__init__()
        self.flatten = ops.Flatten()
        self.layer1 = Dense(28*28, 512)
        self.relu1 = ops.ReLU()
        self.layer2 = Dense(512, 512)
        self.relu2 = ops.ReLU()
        self.layer3 = Dense(512, 10)

    def construct(self, x):
        x = self.flatten(x)
        x = self.layer1(x)
        x = self.relu1(x)
        x = self.layer2(x)
        x = self.relu2(x)
        logits = self.layer3(x)
        return logits

net = Network()
net.layer1.matmul.shard(((2, 1), (1, 2)))
net.layer3.matmul.shard(((2, 2), (2, 1)))
```

### Inference

The difference between distributed inference and inference in the stand-alone mode is that distributed inference needs to call the `model.infer_predict_layout` interface and the `ms.load_distributed_checkpoint` interfaces to load the distributed parameters and the input type must be a Tensor. The code is as follows:

```python
import numpy as np
import mindspore as ms

# The input type in distributed inference scenario must be Tensor
predict_data = ms.Tensor(np.random.randn(1, 28, 28).astype(np.float32))
model = ms.Model(net)
# Get the inference strategy file
predict_layout = model.infer_predict_layout(predict_data)
# Create checkpoint list
ckpt_file_list = ["./src_checkpoints/rank_{}/checkpoint-10_1875.ckpt".format(i) for i in range(0, get_group_size())]
# Load the distributed parameter
ms.load_distributed_checkpoint(net, ckpt_file_list, predict_layout)
predict_result = model.predict(predict_data)
print(predict_result)
```

### Exporting MindIR Files in the Distributed Scenario

In the scenario of ultra-large scale neural network model, a distributed inference scheme can be used to address the problem that the model is unable to perform single-card inference because of the large amount of parameters. At this time, before running the inference task, you need to export multiple MindIR files. The core code is as follows:

```python
import mindspore as ms
from mindspore.communication import get_rank

# Export the distributed MindIR files
ms.export(net, predict_data, file_name='./mindir/net_rank_' + str(get_rank()), file_format='MINDIR')
```

For the case of multi-card training and single-card infereence, exporting MindIR is used in the same way as the stand-alone.

### Running Stand-alone 8-card Script

Next, the corresponding script is called by the command. Take the `mpirun` startup method, the 8-card distributed inference script as an example, and perform the distributed inference:

```bash
bash run_loading.sh
```

The inference results are saved in `log_output/1/rank.*/stdout`, as below:

```text
[[ 0.0504479  -0.94413316  0.84689146 -0.28818333  0.66444737  1.0564338
  -0.04191194  0.25590336 -0.69010115 -0.6532427 ]]
```

MindIR files are stored in the `mindir` directory, and the directory structure is as follows:

```text
├─ mindir
|   ├─ net_rank_0.mindir
|   ├─ net_rank_1.mindir
|   ├─ net_rank_2.mindir
|   ├─ net_rank_3.mindir
|   ├─ net_rank_4.mindir
|   ├─ net_rank_5.mindir
|   ├─ net_rank_6.mindir
|   └─ net_rank_7.mindir
...
```
