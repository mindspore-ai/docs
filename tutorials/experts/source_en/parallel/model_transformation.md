# Model Transformation

[![View Source On Gitee](https://mindspore-website.obs.cn-north-4.myhuaweicloud.com/website-images/master/resource/_static/logo_source_en.svg)](https://gitee.com/mindspore/docs/blob/master/tutorials/experts/source_en/parallel/model_transformation.md)

## Overview

### Background

When using MindSpore for distributed training, it is often necessary to transform the distributed Checkpoint obtained from training to carry out the next steps, such as inference, fine-tuning, and multi-stage training. In this tutorial, we will introduce how to transform the Checkpoint obtained from distributed training to carry out resilient training and inference with distributed strategies and cluster card changes.

> This function only supports SEMI_AUTO_PARALLEL and AUTO_PARALLEL modes.

### Usage Scenarios

If you encounter the following scenario, refer to this tutorial operation for resilience training and inference:

- Scenario 1: Using M cards for training, and using N cards for fine-tuning training, where M and N can have no multiplicative relationship.
- Scenario 2: Training is divided into multiple phases, each with a different cluster size.
- Scenario 3: Using M cards for training, and using N cards for inference, where M and N can have no multiplicative relationship.
- Scenario 4: Changes need to be made to the network sharding strategy.

Related interfaces:

1. `mindspore.transform_checkpoints(src_checkpoints_dir, dst_checkpoints_dir, src_strategy_file, dst_strategy_file)`: Transform Checkpoint of the distributed network from a source sharding strategy to a target sharding strategy, where `src_checkpoints_dir` is the directory where the source Checkpoint file is located, and its subdirectories are required to be stored in the format `rank_x/checkpoint_x.ckpt`, with x being the corresponding rank id. `dst_checkpoints_dir` is the directory where the target checkpoint file is stored, `src_strategy_file` is the name of the source sharding strategy file, and `dst_strategy_file` is the name of the target sharding strategy file.

2. `mindspore.rank_list_for_transform(rank_id, src_strategy_file, dst_strategy_file)`: Get the rank list of source Checkpoint file required for the Checkpoint file of the target rank during the transformation of a distributed Checkpoint.

3. `mindspore.transform_checkpoint_by_rank(rank_id, checkpoint_files_map, save_checkpoint_file_name, src_strategy_file, dst_strategy_file)`: Transform a Checkpoint of a distributed network from a source sharding strategy to a target sharding strategy for a specific rank, where `rank_id` is the rank number of the Checkpoint to be transformed. `checkpoint_files_map` is the source Checkpoint dictionary whose key is the rank number and the value is the path to the Checkpoint file corresponding to that rank number. `save_checkpoint_file_name` is the path and name of the target Checkpoint for the current rank.

## Operation Practice

As an example of training on an Ascend 8-card and fine-tuning on 4-card, the overall procedure is as follows:

1. Perform training, configure the storage location of the model parameter sharding strategy file, and automatically generate the Checkpoint file and the model parameter sharding strategy file.

2. Compile the fine-tuned network, configure the location of the distributed strategy file storage, and automatically generate the model parameter slice and sharding strategy file.

3. The user transforms the saved Checkpoint file based on the strategy file involved in training and inference.

4. After compiling the fine-tuned network, load the distributed Checkpoint file obtained from the transformation.

5. Execute the fine-tuned network.

It should be noted that loading distributed Checkpoint requires [compiling the network](#performing-compilation-on-the-target-network) first.

### Example Code Description

> Download the complete example code: [model_saving_loading](https://gitee.com/mindspore/docs/tree/master/docs/sample_code/model_saving_loading).

The directory structure is as follows:

```text
└─ sample_code
    ├─ model_saving_loading
       ├── model_transformation_infer.py
       ├── model_transformation_retrain.py
       ├── pipeline_train.py
       ├── pipeline_transformation_retrain.py
       ├── run_infer_convert.sh
       ├── run_infer.sh
       ├── run_retrain_convert.sh
       ├── run_retrain.sh
       ├── run_pipeline_train.sh
       ├── run_retrain_pipeline_convert.sh
       ├── run_retrain_pipeline.sh
       ...
    ...
```

The functions of each file are as follows:

- `model_transformation_infer.py`: Scripts for inference after model transformation.
- `model_transformation_retrain.py`: Scripts for second-stage training after model transformation.
- `pipeline_transformation_retrain.py`: Scripts for second-stage training after pipeline parallel model transformation.
- `pipeline_train.py`: Scripts for pipeline parallel training of networks.
- `run_infer_convert.sh`: Scripts that perform model transformation.
- `run_retrain_convert.sh`: Scripts that perform model transformation.
- `run_retrain_pipeline_convert.sh`: Scripts that perform pipeline parallel model transformation.
- `run_infer.sh`: Scripts that perform model inference.
- `run_retrain.sh`: Scripts that perform second-stage training.
- `run_retrain_pipeline.sh`: Scripts for executing second-stage training pipeline parallel model.

### Saving the Distributed Model

First, follow the [Model Saving](https://www.mindspore.cn/tutorials/experts/en/master/parallel/model_saving.html) tutorial to perform 8-card distributed training with a parallel mode of `SEMI_AUTO_ PARALLEL` or `AUTO_PARALLEL`, while customizing the `strategy_ckpt_config` parameter by calling the `set_auto_parallel_context` interface to configure the model sharding strategy file storage path. After training for a period of time, call the `train.ModelCheckpoint` function of storage Checkpoint to store the distributed checkpoint.

At the end of the training, the source Checkpoint file directory as well as the source sharding strategy file will be generated at the current path:

```text
src_checkpoints/
src_strategy.ckpt
```

> Subdirectory within src_checkpoints are required to be stored in the `rank_x/checkpoint_x.ckpt` format.

That is, the directory structure of src_checkpoints is changed to the following:

```text
src_checkpoints
 ├─ rank_0
 |   └─ checkpoint_0.ckpt
 ├─ rank_1
 |   └─ checkpoint_1.ckpt
 ├─ rank_2
 |   └─ checkpoint_2.ckpt
 ├─ rank_3
 |   └─ checkpoint_3.ckpt
 ├─ rank_4
 |   └─ checkpoint_4.ckpt
 ├─ rank_5
 |   └─ checkpoint_5.ckpt
 ├─ rank_6
 |   └─ checkpoint_6.ckpt
 └─ rank_7
     └─ checkpoint_7.ckpt
...
```

### Generating Target Strategy Files

Then the network under the new card or sharding strategy needs to be compiled to generate the model sharding strategy file for the target network. In this example, the original strategy is trained with 8 cards, the `ops.MatMul()` operator parallel strategy of layer1 is ((2, 1), (1, 2)), the optimizer parallel is not turned on, and the strategy file is named as src_strategy.ckpt. The target strategy is trained with 4 cards, the `ops.MatMul()` operator parallel strategy of layer1 is ((2, 2), (2, 1)) and optimizer parallel is turned on, the strategy file is named as dst_strategy.ckpt.

#### Configuring Distributed Environment

Specify the run mode, run device, run card number via the context interface. Configure and save the distributed strategy file via `strategy_ckpt_config`, enable optimizer parallel and initialize HCCL or NCCL communication via init.

```python
import mindspore as ms
from mindspore.communication import init

ms.set_context(mode=ms.GRAPH_MODE)
ms.set_auto_parallel_context(parallel_mode=ms.ParallelMode.SEMI_AUTO_PARALLEL)
ms.set_auto_parallel_context(strategy_ckpt_config={"save_file": args_opt.dst_strategy_file})
ms.set_auto_parallel_context(enable_parallel_optimizer=True)
init()
ms.set_seed(1)
```

#### Network Definition and Loading the Dataset

The network definition modifies the `ops.MatMul()` operator parallel strategy for layer1 in the original network:

```python
import os
from mindspore import nn, ops
import mindspore.dataset as ds
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
net.layer1.matmul.shard(((1, 4), (4, 1)))
net.layer3.matmul.shard(((2, 2), (2, 1)))

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

#### Performing Compilation on the Target Network

The distributed Checkpoint transformation depends on the original distributed strategy file and the target distributed strategy file. When performing the training of the network under the original strategy, the distributed strategy file is stored, so it is necessary to obtain the distributed strategy file under the target strategy separately. The distributed strategy file for the target strategy network can be obtained by performing compilation of the network with the target strategy. Compilation of the network can be performed separately through the `model.infer_train_layout` interface.

```python
import mindspore as ms
from mindspore import nn, ops

optimizer = nn.SGD(net.trainable_params(), 1e-2)
loss_fn = nn.CrossEntropyLoss()
model = ms.Model(net, loss_fn=loss_fn, optimizer=optimizer)
model.infer_train_layout(data_set)
```

When the target network is to perform inference, `model.infer_train_layout` is replaced with `model.infer_predict_layout` to perform compilation:

```python
import numpy as np
import mindspore as ms

predict_data = ms.Tensor(np.random.randn(1, 28, 28).astype(np.float32))
model = ms.Model(net)
model.infer_predict_layout(predict_data)
```

After compilation, you can get the target sharding strategy file `dst_strategy.ckpt`.

### Executing Distributed Checkpoint Transformation

In this step, you need to call the distributed Checkpoint transformation interface for distributed Checkpoint transformation. Distributed Checkpoint provides two interfaces for Checkpoint transformation.

The first interface, `transform_checkpoints`, requires the user to place all checkpoints in a single directory, and the subdirectories must be named in the format "rank_0, rank_1, rank_2, ...". The user calls this interface to transform the entire directory directly, which is easier to use, but the transformation requires a slightly higher memory overhead.

The second interface, `transform_checkpoint_by_rank`, is used to get the checkpoints for a particular rank, which has more flexibility and lower memory overhead, and needs to be used in conjunction with the `rank_list_for_transform` interface to determine original Checkpoints are needed to get the target checkpoints for this rank.

1. Use the interface `transform_checkpoints`.

    ```python
    import mindspore as ms

    ms.transform_checkpoints(args_opt.src_checkpoints_dir, args_opt.dst_checkpoints_dir, "checkpoint_", args_opt.src_strategy_file, args_opt.dst_strategy_file)
    ```

    The method is used in the example code [`model_transformation_retrain.py`](https://gitee.com/mindspore/docs/blob/master/docs/sample_code/model_saving_loading/model_transformation_retrain.py).

2. Call the `transform_checkpoint_by_rank` interface to perform a parameter merge on the original Checkpoint corresponding to the current rank.

    > Ensure that the subdirectory "rank_x" exists in dst_checkpoints_dir.

    ```python
    import os
    import mindspore as ms
    from mindspore.communication import get_rank

    rank_list = ms.rank_list_for_transform(get_rank(), args_opt.src_strategy_file, args_opt.dst_strategy_file)
    checkpoint_file_map = {}
    for rank_id in rank_list:
        checkpoint_file_map[rank_id] = os.path.join(args_opt.src_checkpoints_dir, "rank_{}".format(rank_id), "checkpoint_{}.ckpt".format(rank_id))
    save_checkpoint_path = os.path.join(args_opt.dst_checkpoints_dir, "rank_{}".format(get_rank()), "checkpoint_{}.ckpt".format(get_rank()))
    ms.transform_checkpoint_by_rank(get_rank(), checkpoint_file_map, save_checkpoint_path, args_opt.src_strategy_file, args_opt.dst_strategy_file)
    ```

    The method is used in the example code [`model_transformation_infer.py`](https://gitee.com/mindspore/docs/blob/master/docs/sample_code/model_saving_loading/model_transformation_infer.py).

After execution, a directory of transformed target Checkpoint files will be generated:

```text
dst_checkpoints/
```

### Loading the Transformed Checkpoint Files

The network for the target strategy is compiled and the `load_checkpoint` interface is called to load the model parameter data from the transformed Checkpoint file.

Compile the network using the `model.infer_train_layout` (for training) or `model.infer_predict_layout` (for inference) interfaces, at which point the weight Shape is sliced in the compilation process. Call the `load_checkpoint` interface to load the model parameter data for each card from the Checkpoint file.

The target network is the training scenario:

```python
import os
import mindspore as ms
from mindspore import nn, train

save_checkpoint_path = os.path.join(args_opt.dst_checkpoints_dir, "rank_{}".format(get_rank()), "checkpoint_{}.ckpt".format(get_rank()))
loss_cb = train.LossMonitor(20)
model.infer_train_layout(data_set)
param_dict = ms.load_checkpoint(save_checkpoint_path)
ms.load_param_into_net(net, param_dict)
model.train(2, data_set, callbacks=[loss_cb])
```

- `save_checkpoint_path`: The name of the Checkpoint model parameter file corresponding to the current rank that needs to be loaded.

The target network is the inference scenario:

```python
import os
import mindspore as ms

save_checkpoint_path = os.path.join(args_opt.dst_checkpoints_dir, "rank_{}".format(get_rank()), "checkpoint_{}.ckpt".format(get_rank()))
param_dict = ms.load_checkpoint(save_checkpoint_path)
model.infer_predict_layout(predict_data)
ms.load_param_into_net(net, param_dict)
predict_result = model.predict(predict_data)
print(predict_result)
```

- `predict_data`: Tensor data for inference.

### Running Stand-alone 4-card Script

Next, the corresponding scripts are called by commands to perform second-stage fine-tuning training after model transformation, using the `mpirun` startup method with a 4-card distributed script as an example:

```bash
bash run_retrain_convert.sh
bash run_retrain.sh
```

Or infer after the model transformation:

```bash
bash run_infer_convert.sh
bash run_infer.sh
```

After the execution is completed, the log file is saved to the `log_output` directory, the target Checkpoint file is saved in the `dst_checkpoints` folder, and the target strategy file is saved in `dst_strategy.ckpt`, with the following directory structure of the files:

```text
├─ src_strategy.ckpt
├─ dst_strategy.ckpt
├─ log_output
|   └─ 1
|       ├─ rank.0
|       |   └─ stdout
|       ├─ rank.1
|       |   └─ stdout
|       ...
├─ dst_checkpoints
|   ├─ rank_0
|   |   └─ checkpoint_0.ckpt
|   ├─ rank_1
|   |   └─ checkpoint_1.ckpt
|   |   ...
|   ...
...
```

The part of results of the Loss after the second-stage fine-tuned training are saved in `log_output/1/rank.*/stdout`, as exemplified below:

```text
epoch: 1, step: 20, loss is 0.10617774
epoch: 1, step: 40, loss is 0.06953259
epoch: 1, step: 60, loss is 0.08409108
epoch: 1, step: 80, loss is 0.08699021
epoch: 1, step: 100, loss is 0.07113413
...
```

In the case of an inference task, the results are saved in `log_output/1/rank.*/stdout`, as exemplified below:

```text
[[ 0.05044775 -0.94413316  0.84689134 -0.2881832   0.66444755  1.0564336
  -0.04191193  0.25590348 -0.690101   -0.6532427 ]]
```

### Pipeline Parallel Model Transformation

[Pipelining Parallel](https://www.mindspore.cn/tutorials/experts/en/master/parallel/pipeline_parallel.html) is to slice a linear network to get multiple sub-networks, which are pipelined among multiple cards. Therefore, the sharding strategy file stored for each subgraph is inconsistent, and all the sharding strategies are aggregated together to get the complete slicing information of the network.
Therefore, for the dimension of pipeline parallel, compared to the transformation of other dimensions, it is necessary to perform an operation of aggregating the sharding strategy file before getting the aggregated sharding strategy file, and use this file as the strategy file on which the distributed Checkpoint transformation depends. In addition, there is no difference with the previous [Executing Distributed Checkpoint Transformation](https://www.mindspore.cn/tutorials/experts/en/master/parallel/model_transformation.html#executing-distributed-checkpoint-transformation).

Related interfaces:

`mindspore.merge_pipeline_strategys(src_strategy_dirs, dst_strategy_file)`: the sharding strategy file that aggregates the subgraphs of all pipeline parallels in pipeline parallel mode. `src_strategy_dirs` is the directory containing the sharding strategy files for all pipeline-parallel subgraphs, and the sharding strategy files are obtained by storing them by the `mindspore.set_auto_parallel_context(strategy_ckpt_config)` interface. `dst_strategy_file` is the path to the file where the converged sharding strategy is stored.

First, 8-card pipeline parallel training is executed, where pipeline parallel dimension is 2 and optimizer parallelism is turned on.

The training code is in [pipeline_train.py](https://gitee.com/mindspore/docs/blob/master/docs/sample_code/model_saving_loading/pipeline_train.py). The network structure adds a pipeline parallel configuration based on the chapter [Model Saving](https://www.mindspore.cn/tutorials/experts/en/master/parallel/model_saving.html) with parallel dimension 2.

The core code is:

```python
import mindspore as ms
from mindspore import nn, train
from mindspore.communication import init, get_rank
...
ms.set_context(mode=ms.GRAPH_MODE)
ms.set_auto_parallel_context(parallel_mode=ms.ParallelMode.SEMI_AUTO_PARALLEL)
ms.set_auto_parallel_context(pipeline_stages=2, enable_parallel_optimizer=True)
init()
ms.set_auto_parallel_context(strategy_ckpt_config={"save_file": "./src_pipeline_strategys/src_strategy_{}.ckpt".format(get_rank())})
ms.set_seed(1)
...
ckpt_config = train.CheckpointConfig(save_checkpoint_steps=1000, keep_checkpoint_max=1, integrated_save=False)
ckpoint_cb = train.ModelCheckpoint(prefix="checkpoint",
                                   directory="./src_checkpoints_pipeline/rank_{}".format(get_rank()),
                                   config=ckpt_config)
net_with_grads = nn.PipelineCell(nn.WithLossCell(net, loss_fn), 4)
model = ms.Model(net_with_grads, optimizer=optimizer)
model.train(3, data_set, callbacks=[loss_cb, ckpoint_cb])
```

> The sharding strategy file is inconsistent for each card, so it needs to be saved separately and stored in "src_pipeline_strategys/src_strategy_x.ckpt" format.

Execute the 8-card training script execution command as:

```bash
bash run_pipeline_train.sh
```

After execution, the source Checkpoint file directory and the source sharding strategy file will be generated with the file directory structure:

```text
├─ src_checkpoints_pipeline
|   ├─ rank_0
|   |   ├─ checkpoint-3_1875.ckpt
|   |   └─ checkpoint-graph.meta
|   ├─ rank_1
|   |   ├─ checkpoint-3_1875.ckpt
|   |   ...
|   ...
├─ src_pipeline_strategys
|   ├─ src_strategy_0.ckpt
|   ├─ src_strategy_1.ckpt
|   ├─ src_strategy_2.ckpt
|   ├─ src_strategy_3.ckpt
|   ├─ src_strategy_4.ckpt
|   ├─ src_strategy_5.ckpt
|   ├─ src_strategy_6.ckpt
|   └─ src_strategy_7.ckpt
...
```

Refer to [Performing a compilation of the target network](https://www.mindspore.cn/tutorials/experts/en/master/parallel/model_transformation.html#performing-compilation-on-the-target-network) section, and similarly compile the target network in order to obtain the sharding strategy file for the target network.

The next step unfolds the distributed Checkpoint dimension transformation containing pipeline parallel dimensions, first merging the sharding strategy files obtained from pipline training using interface `merge_pipeline_strategys`, and then performing the distributed Checkpoint transformation using interface `transform_checkpoints` or `transform_checkpoint_by_rank`.

The example introduces an interface that uses `transform_checkpoints`, and the interface that uses `transform_checkpoint_by_rank`. Refer to [Executing Distributed Checkpoint Transformation](https://www.mindspore.cn/tutorials/experts/en/master/parallel/model_transformation.html#executing-distributed-checkpoint-transformation).

```python
import mindspore as ms

ms.merge_pipeline_strategys(args_opt.src_strategy_dir, args_opt.src_strategy_file)
ms.transform_checkpoints(args_opt.src_checkpoints_dir, args_opt.dst_checkpoints_dir, "checkpoint_", args_opt.src_strategy_file, args_opt.dst_strategy_file)
```

> Subdirectories within src_checkpoints_dir are required to be stored in the format "rank_x/checkpoint_x.ckpt".

The example script execution command to transform the entire Checkpoint catalog is:

```bash
bash run_retrain_pipeline_convert.sh
```

After the transformation is completed, refer to [Loading the Transformed Checkpoint Files](https://www.mindspore.cn/tutorials/experts/en/master/parallel/model_transformation.html#loading-the-transformed-checkpoint-files) section to execute the distributed network without pipeline dimension.

In the example, the script execution command for loading the transformed Checkpoint for second-stage fine-tuning training is:

```bash
bash run_retrain_pipeline.sh
```

After the execution is complete, you can see that the loss is decreasing from 0.15:

```text
epoch: 1, step: 20, loss is 0.15090162
epoch: 1, step: 40, loss is 0.13296325
epoch: 1, step: 60, loss is 0.14676111
epoch: 1, step: 80, loss is 0.11930083
epoch: 1, step: 100, loss is 0.0784434
epoch: 1, step: 120, loss is 0.10741685
```
