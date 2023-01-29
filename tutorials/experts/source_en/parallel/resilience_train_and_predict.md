# Distributed Resilience Training and Inference

<a href="https://gitee.com/mindspore/docs/blob/r2.0.0-alpha/tutorials/experts/source_en/parallel/resilience_train_and_predict.md" target="_blank"><img src="https://mindspore-website.obs.cn-north-4.myhuaweicloud.com/website-images/r2.0.0-alpha/resource/_static/logo_source_en.png"></a>

## Overview

### Background

When using MindSpore for distributed training, it is often necessary to convert the distributed Checkpoint obtained from training for the next step, such as inference, fine-tuning, multi-stage training. This tutorial will introduce how to convert the Checkpoint obtained from distributed training to carry out resilient training and inference with distributed strategies and changed number of cards in the cluster.
This function only supports semi_auto_parallel/auto_parallel mode, and temporarily does not support pipeline parallel dimension conversion.

### Usage Scenarios

If you encounter the following scenarios, you need to refer to this tutorial for resilience training and inference.

Scenario1: M number of cards perform training, while N number of cards perform fine-tuning training. M and N can have no multiplicative relationship.
Scenario2: The training is divided into multiple phases, each with a different cluster size.
Scenario3: M number of cards perform training, while N number of cards perform inference. M and N can have no multiplicative relationship.
Scenario4: Changes need to be made to the network's sharding strategy.

Using the example of training on 8 cards and fine-tuning on 4 cards, the overall procedure is as follows:

1. Execute training, configure the storage location of model parameter sharding strategy files, and automatically generate Checkpoint files and model parameter sharding strategy files.

2. Compile fine-tuned networks, configure distributed strategy file storage locations, and automatically generate model parameter sharding strategy files.

3. The user converts the saved Checkpoint file based on the strategy file involved in the training and inference.

4. After compiling the fine-tuned network, load the distributed Checkpoint file obtained by the conversion.

5. Perform fine-tuning of the network.

Note that loading a distributed Checkpoint requires that the network be compiled before it can be loaded.

> For dataset download, please refer to the [Preparation](https://www.mindspore.cn/tutorials/experts/en/r2.0.0-alpha/parallel/transformer.html#preparation) in the Distributed Parallel Training Transformer Model tutorial.
>
> Download the complete sample code: [Distributed Resilience Training](https://gitee.com/mindspore/docs/tree/r2.0.0-alpha/docs/sample_code/distributed_resilience_training).

## Converting Distributed Checkpoint Files

### Overall Process

First, perform distributed training with parallel mode set to `semi_auto_parallel`/`auto_parallel`, and also custom `strategy_ckpt_save_file` parameter and configure the model sharding strategy file storage path by calling the `set_auto_parallel_context` interface.
After a period of training, the callback function that stores Checkpoint is called to store the distributed Checkpoint. And then compile the network under the new number of cards/sharding strategy, generate the model sharding strategy file of the target network, and call the distributed checkpoint conversion interface for distributed checkpoint conversion.

### Executing the Distributed Training

Define the network, perform distributed initialization, and get the number of devices and card numbers. For the non-pipelined parallel case, the content of the sharding strategy file is the same for each card, so just call `set_auto_parallel_context(strategy_ckpt_save_file=". /src_strategy.ckpt")` on card 0 to save the strategy file.

Add a callback function to save Checkpoint, first define configuration object `CheckpointConfig` related to the Checkpoint storage. Note that `integrated_save` is configured to `False`, which means that aggregated saving is not performed on distributed training weights to accommodate the memory overhead under large models.
And then define the callback function `ModelCheckpoint` that saves the checkpoint. Finally, call `model.train` to perform the training.
For the basic usage of distributed training, please refer to [Distributed Training Ascend](https://www.mindspore.cn/tutorials/experts/en/r2.0.0-alpha/parallel/train_ascend.html) or [Distributed Training GPU](https://www.mindspore.cn/tutorials/experts/en/r2.0.0-alpha/parallel/train_gpu.html).

```python
import mindspore as ms
from mindspore.train import Model, ModelCheckpoint, CheckpointConfig, TimeMonitor, LossMonitor
import mindspore.communication as D
D.init()
device_num = D.get_group_size()
rank_id = D.get_rank()
net = Net()
ms.set_auto_parallel_context(parallel_mode=ms.ParallelMode.SEMI_AUTO_PARALLEL)
if rank_id == 0:
    ms.set_auto_parallel_context(strategy_ckpt_save_file="../src_strategy.ckpt")
opt = Momentum(learning_rate=0.01, momentum=0.9, params=net.get_parameters())
model = Model(net, optimizer=opt)
ckpt_config = CheckpointConfig(save_checkpoint_steps=callback_size, keep_checkpoint_max=1,
                                  integrated_save=False)
ckpoint_cb = ModelCheckpoint(prefix="src_checkpoint",
                                directory = "../src_checkpoints/rank_{}".format(rank_id),
                                config=ckpt_config)
callback = [TimeMonitor(callback_size), LossMonitor(callback_size), ckpoint_cb]
model.train(2, dataset, callbacks=callback, dataset_sink_mode=True)
```

- `dataset`: MindData objects, which need to be constructed in advance to feed into `model.train`.

The 8-card training script execution command executed in the example is as follows:

```bash
bash run_train_8p.sh ../output/wmt14.en_fr.txt
```

After execution, the source Checkpoint file directory and the source sharding strategy file will be generated:

```text
src_checkpoints/
src_strategy.ckpt
```

### Converting the Distributed Checkpoint

#### Executing Compilation on the Target Network

Perform the distributed checkpoint conversion, which depends on the original distributed strategy file and the target distributed strategy file. When the network training under the original strategy is executed, the distributed strategy file is already stored, so the distributed strategy file under the target strategy needs to be obtained separately.
The distributed strategy file of the target strategy network can be obtained by performing compilation on the network of the target strategy. Compilation is performed on the network alone by the `model.build` interface.

```python
import mindspore as ms
from mindspore.train import Model
import mindspore.communication as D
D.init()
device_num = D.get_group_size()
rank_id = D.get_rank()
net = Net()
ms.set_auto_parallel_context(parallel_mode=ms.ParallelMode.SEMI_AUTO_PARALLEL)
if rank_id == 0:
    ms.set_auto_parallel_context(strategy_ckpt_save_file="../dst_strategy.ckpt")
opt = Momentum(learning_rate=0.01, momentum=0.9, params=net.get_parameters())
model = Model(net, optimizer=opt)
model.build(train_dataset=dataset, epoch=1)
```

- `dataset`: MindData objects, which need to be constructed in advance to feed into `model.train`.

When the target network is for inference, `model.build` is replaced with `model.infer_preict_layout` to perform compilation.

```python
import mindspore as ms
from mindspore.train import Model
import mindspore.communication as D
D.init()
device_num = D.get_group_size()
rank_id = D.get_rank()
net = Net()
ms.set_auto_parallel_context(parallel_mode=ms.ParallelMode.SEMI_AUTO_PARALLEL)
if rank_id == 0:
    ms.set_auto_parallel_context(strategy_ckpt_save_file="../dst_strategy.ckpt")
opt = Momentum(learning_rate=0.01, momentum=0.9, params=net.get_parameters())
model = Model(net, optimizer=opt)
model.infer_predict_layout(Tensor(np.ones(shape=data_shape)))
```

- `data_shape`: Shape of the inference data.

The 4-card target network compilation command executed in the example is as follows:

```bash
bash run_compile_4p.sh ../output/wmt14.en_fr.txt
```

After execution, the target sharding strategy files will be generated:

```text
dst_strategy.ckpt
```

#### Performing Distributed Checkpoint Conversions

In the example, the original strategy is trained with 8-card, model parallelism with 4-card, data parallelism with 2-card. The optimizer parallelism is turned on, and the strategy file is named `src_strategy.ckpt`.
The target strategy is trained with 4-card, model parallelism with 4-card, data parallelism with 1-card. The optimizer parallelism is turned off, and the strategy file is named `dst_stategy.ckpt`.

Distributed Checkpoint provides two interfaces to convert Checkpoint. The first interface, `transform_checkpoints`, requires the user to place all Checkpoints in one directory, and the subdirectories must be named in the format "rank_0, rank_1, rank_2, ... ".
The user calls this interface to convert the entire directory directly. This approach is easier to use, but the memory overhead required for conversion is slightly higher. The second interface, `transform_checkpoint_by_rank`, is used to get the Checkpoint of a specific rank, which has more flexibility and lower memory overhead.
It needs to be used with the `rank_list_for_transform` interface to get which original Checkpoint is needed for the target Checkpoint of this rank.

1. Use interface `transform_checkpoints`.

   ```python
   import mindspore as ms
   ms.transform_checkpoints(src_checkpoints_dir, dst_checkpoints_dir,
                            "transformed", src_strategy_file, dst_strategy_file)
   ```

   > The subdirectories in src_checkpoints_dir are required to be stored in the format "rank_x/checkpoint_x.ckpt".

   In the example, the script execution command for the conversion of the entire Checkpoint directory is:

    ```bash
    python transform_checkpoint_dir.py --src_strategy_file=./src_strategy.ckpt --dst_strategy_file=./dst_strategy.ckpt --src_checkpoints_dir=./src_checkpoints --dst_checkpoints_dir=./dst_checkpoints
    ```

2. Call `transform_checkpoint_by_rank` interface to merge the parameters of `transform_rank`.

   ```python
   import os
   import mindspore as ms
   rank_list = ms.rank_list_for_transform(transform_rank, src_strategy_file, dst_strategy_file)
   checkpoint_file_map = {}
   for rank_id in rank_list:
       checkpoint_file_map[rank_id] = os.path.join(src_checkpoints_dir, "rank_{}".format(rank_id), "src_checkpoint{}.ckpt".format(rank_id))
   save_checkpoint_path = os.path.join(dst_checkpoints_dir, "rank_{}".format(transform_rank),
                                       "dst_checkpoint{}.ckpt".format(transform_rank))
   ms.transform_checkpoint_by_rank(transform_rank, checkpoint_file_map, save_checkpoint_path,
                                   src_strategy_file, dst_strategy_file)
   ```

   In the example, the script execution command to convert Checkpoint by rank one by one is:

   ```bash
   bash transform_by_rank.sh ./src_strategy.ckpt ./dst_strategy.ckpt ./src_checkpoints ./dst_checkpoints
   ```

After execution, the following directory of converted target Checkpoint files will be generated:

```text
dst_checkpoints/
```

## Loading the Checkpoint Files Obtained from Conversion

### Overall Process

Compile the network for the target strategy and call the `load_checkpoint` interface to load the model parameter data from the converted Checkpoint file.

### Compiling and Executing the Target Network

Compile the network by using the `model.build` (for training) or `model.infer_predict_layout` (for inference) interfaces. At this time, the weight Shape is sliced in the compilation process. Call the `load_checkpoint` interface to load the model parameter data of each card from the Checkpoint file.

The target network is the training scenario:

```python
import mindspore as ms
from mindspore.train import Model
import mindspore.communication as D
D.init()
device_num = D.get_group_size()
rank_id = D.get_rank()
net = Net()
ms.set_auto_parallel_context(parallel_mode=ms.ParallelMode.SEMI_AUTO_PARALLEL)
if rank_id == 0:
    ms.set_auto_parallel_context(strategy_ckpt_save_file="../dst_strategy.ckpt")
opt = Momentum(learning_rate=0.01, momentum=0.9, params=net.get_parameters())
model = Model(net, optimizer=opt)
param_dict = ms.load_checkpoint(ckpt_file)
model.build(train_dataset=dataset, epoch=2)
ms.load_param_into_net(net, param_dict)
model.train(2, dataset, callbacks=callback, dataset_sink_mode=True)
```

- `ckpt_file`: The name of the Checkpoint model parameter file to be loaded.

The target network is the inference scenario:

```python
import mindspore as ms
from mindspore.train import Model
import mindspore.communication as D
D.init()
device_num = D.get_group_size()
rank_id = D.get_rank()
net = Net()
ms.set_auto_parallel_context(parallel_mode=ms.ParallelMode.SEMI_AUTO_PARALLEL)
if rank_id == 0:
    ms.set_auto_parallel_context(strategy_ckpt_save_file="../dst_strategy.ckpt")
opt = Momentum(learning_rate=0.01, momentum=0.9, params=net.get_parameters())
model = Model(net, optimizer=opt)
param_dict = ms.load_checkpoint(ckpt_file)
model.infer_predict_layout(predict_data)
ms.load_param_into_net(net, param_dict)
model.predict(2, predict_data)
```

- `predict_data`: Tensor data used to infer.

In the example, the script execution command to load the converted Checkpoint for two-stage fine-tuning training is:

```bash
bash run_train_4p.sh ../output/wmt14.en_fr.txt
```

After the execution is completed, the loss can be seen to decrease from 6.45.

```text
epoch: 1 step: 73, loss is 6.45995
epoch: 1 step: 73, loss is 6.13733
```

## Pipeline Parallel Dimension Conversion

[Pipeline Parallelism](https://www.mindspore.cn/tutorials/experts/en/r2.0.0-alpha/parallel/pipeline_parallel.html) is to slice the linear network to get multiple sub-networks, which are pipelined between multiple cards. Therefore the sharding strategy file stored down for each subgraph is inconsistent, and all sharding strategies are aggregated to get the complete sharding information of the network.
Therefore, for the pipelined parallel dimensions, compared to the conversion of other dimensions, it is necessary to perform an aggregated shardig strategy file operation in advance to obtain the aggregated sharding strategy file, and use this file as the strategy file for the distributed Checkpoint conversion dependency. In addition, there is no difference from the previous section [Sharding Strategy Conversion](https://www.mindspore.cn/tutorials/experts/en/r2.0.0-alpha/parallel/resilience_train_and_predict.html#converting-the-distributed-checkpoint).

First, execute an 8-card pipeline parallel training, where the pipeline parallel dimension is 2, the operator-level model parallel dimension is 4, and the data parallel dimension is 1.

```python
from mindspore import train
import mindspore as ms
import mindspore.communication as D
D.init()
device_num = D.get_group_size()
rank_id = D.get_rank()
net = Net()
net = PipelineCell(net, 4) # micro_batch=4
ms.set_auto_parallel_context(parallel_mode=ms.ParallelMode.SEMI_AUTO_PARALLEL, pipeline_stages=2)
ms.set_auto_parallel_context(strategy_ckpt_save_file="../src_pipeline_strategys/src_strategy{}.ckpt")
opt = Momentum(learning_rate=0.01, momentum=0.9, params=net.get_parameters())
model = ms.Model(net, optimizer=opt)
ckpt_config = train.CheckpointConfig(save_checkpoint_steps=callback_size, keep_checkpoint_max=1,
                                     integrated_save=False)
ckpoint_cb = train.ModelCheckpoint(prefix="src_checkpoint",
                                   directory = "../src_checkpoints/rank_{}".format(rank_id),
                                   config=ckpt_config)
callback = [train.TimeMonitor(callback_size), train.LossMonitor(callback_size), ckpoint_cb]
model.train(2, dataset, callbacks=callback, dataset_sink_mode=True)
```

- `dataset`: MindData objects, which need to be constructed in advance to feed into `model.train`.

The 8-card training script execution command executed in the example is as follows:

```bash
bash run_train_8p_pipeline.sh ../output/wmt14.en_fr.txt
```

After execution, the source Checkpoint file directory and the source sharding strategy file will be generated:

```text
src_checkpoints_pipeline/
src_pipeline_strategys/
```

Refer to "performing compilation on target network module" of [Sharding Strategy Conversion](https://www.mindspore.cn/tutorials/experts/en/r2.0.0-alpha/parallel/resilience_train_and_predict.html#converting-the-distributed-checkpoint) section to also compile the target network to get the sharding strategy file for the target network.

The 4-card target network compilation command executed in the example is as follows:

```bash
bash run_compile_4p.sh ../output/wmt14.en_fr.txt
```

After execution, the target sharding strategy files will be generated:

```text
dst_strategy.ckpt
```

The next step unfolds the distributed Checkpoint dimension conversion that contains the pipeline parallel dimension. First, the `merge_pipeline_strategys` interface is used to merge the sharding strategy files obtained from pipeline training, and then the distributed checkpoint conversion is performed by using the interface `transform_checkpoints` or `transform_checkpoint_by_rank`.

The example gives the interface using `transform_checkpoints`. For the interface using `transform_checkpoint_by_rank` please refer to introduction in [sharding strategy conversion](https://www.mindspore.cn/tutorials/experts/en/r2.0.0-alpha/parallel/resilience_train_and_predict.html#converting-the-distributed-checkpoint).

```python
import mindspore as ms
ms.merge_pipeline_strategys(src_pipeline_strategys_dir, src_strategy_file)
ms.transform_checkpoints(src_checkpoints_dir, dst_checkpoints_dir,
                            "transformed", src_strategy_file, dst_strategy_file)
```

> The subdirectories in src_checkpoints_dir are required to be stored in the format "rank_x/checkpoint_x.ckpt".

In the example, the script execution command for the entire Checkpoint directory conversion is:

```bash
python transform_checkpoint_dir_pipeline.py --src_strategy_dir=./src_pipeline_strategys --dst_strategy_file=dst_strategy.ckpt --src_checkpoints_dir=./src_checkpoints --dst_checkpoints_dir=./dst_checkpoints
```

After the conversion is complete, refer to the [Execute Target Network Chapter](https://www.mindspore.cn/tutorials/experts/en/r2.0.0-alpha/parallel/resilience_train_and_predict.html#loading-the-checkpoint-files-obtained-from-conversion). Load the distributed checkpoint obtained from the conversion and execute the distributed network without the pipeline dimension.

In the example, the script execution command to load the converted Checkpoint for two-stage fine-tuning training is:

```bash
bash run_train_4p.sh ../output/wmt14.en_fr.txt
```

After the execution is completed, the loss can be seen to decrease from 6.45.

```text
epoch: 1 step: 73, loss is 6.45995
epoch: 1 step: 73, loss is 6.13733
```
