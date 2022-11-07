# Distributed Fault Recovery

<a href="https://gitee.com/mindspore/docs/blob/master/tutorials/experts/source_en/parallel/fault_recover.md" target="_blank"><img src="https://mindspore-website.obs.cn-north-4.myhuaweicloud.com/website-images/master/resource/_static/logo_source_en.png"></a>

## Overview

It is very common to encounter failures when performing distributed training, similar to single-card training, which can be continued by loading the saved weight information during training. Distinct from pure data parallel training, when model parallelism is applied, the weights are sliced and the weight information saved between cards may not be consistent.
To solve this problem, one option is to aggregate the weights through the [AllGather](https://www.mindspore.cn/tutorials/experts/en/master/parallel/communicate_ops.html#allgather) before saving the weight checkpoint file, where each card stores a complete information about the weights. This one function has been introduced in [Distributed training model parameter saving and loading](https://www.mindspore.cn/tutorials/experts/en/master/parallel/train_ascend.html#saving-and-loading-distributed-training-model-parameters).
However, for large models, the overhead of using aggregated preservation is too large for all kinds of resources, so this document presents a recovery scheme where each card only saves its own weight information. For large models, both data parallelism and model parallelism are often applied, and the devices divided by the dimensions of data parallelism, which hold exactly the same weight information, provide a redundant backup for large models. This document will also point out how to go about obtaining this redundant information.
For the relationship between the parallel strategy and the slicing division of the weights, the following mapping can be performed. For more information on the concepts of data parallelism and model parallelism, please refer to [Distributed Training](https://www.mindspore.cn/tutorials/experts/en/master/parallel/train_ascend.html). For more information about optimizer parallelism, please refer to [Optimizer Parallelism](https://www.mindspore.cn/tutorials/experts/en/master/parallel/optimizer_parallel.html).

- Data parallelism + keep optimizer parallelism off: The ranks in the parallel communication domain hold the same weight slice.
- Model parallism: The ranks in the parallel communication domain hold different weight slices.
- Data parallelism + keep optimizer parallelism on + the number of shards in optimizer parallelism is smaller than the number of all data parallel dimensions: Within the parallel communication domain, the rank within the communication domain sliced by the optimizer holds different weight slices, and the communication domain sliced by each optimizer holds the same weight slice between them.

Also, it should be noted that this document introduces the distributed faults recovery scheme, which needs to be used in sink mode. This document will introduce the scheme as an example of distributed parallel training Transformer model. For detailed information of transformer, please refer to this tutorial.

> Download the complete sample code here: [distributed_training_transformer](https://gitee.com/mindspore/docs/tree/master/docs/sample_code/distributed_training_transformer)

The directory structure is as follows:

```text
└─sample_code
    ├─distribute_training_transformer
        ├── dataset.py
        ├── model.py
        ├── rank_table_8pcs.json
        ├── run_parallel_save_ckpt.sh
        ├── run_parallel_recover_ckpt.sh
        ├── parallel_save_ckpt_train.py
        └── parallel_recover_train.py
```

## Slicing Preservation Weight

To save the weight information of the slices, simply configure integrated_save to False in CheckpointConfig. Also, configure the environment variable GROUP_INFO_FILE to store redundant information about the weights.

```bash
export GROUP_INFO_FILE=./group_info.pb
```

The code section for weight storage is as follows. Note that training is configured to sink mode by specifying dataset_sink_mode to True.

```python
import mindspore as ms
from mindspore.train import CheckpointConfig, ModelCheckpoint
from mindspore.nn import PipelineCell

def train():
    # model create
    # checkpoint save
    ckpt_config = CheckpointConfig(save_ckpt_steps=callback_size, keep_ckpt_max=4,
                                      integrated_save=False)
    ckpoint_cb = ModelCheckpoint(prefix="test", config=ckpt_config)
    callback = [ckpoint_cb]
    model.train(4, dataset, callbacks=callback, dataset_sink_mode=True)
```

## Loading Weights to Continue Training

After saving the weight slices in the previous step, the following files can be seen in the directory obtained from the training, taking the 0-card directory as an example.

```text
└─ckpt_dir0
    ├── group_info.pb
    ├── test-1_77.ckpt
    └── train.log0
```

In train.log0, you can see the current loss value after training, similar to the following.

```text
epoch: 1 step: 77, loss is 7.187697
epoch: 1 step: 77, loss is 6.612632
epoch: 1 step: 77, loss is 6.393444
epoch: 1 step: 77, loss is 6.271424
```

Reading group_info.pb can get the redundant information of the weights. The file will be parsed out to get a list with the value of rank_id, which means that the weight slices corresponding to the rank_id in these lists are all the same and can be replaced with each other.
As in the following example, after the 0-card group_info.pb of is parsed, it is found that the weight slicing of 0-card and 4-card are exactly the same. When the 0-card checkpoint is lost, the 4-card checkpoint can be directly copied as the 0-card checkpoint and the 0-card checkpoint can be recovered.

```python
import mindspore as ms
rank_list = ms.restore_group_info_list("./ckpt_dir0/group_info.pb")
print(rank_list) // [0, 4]
```

Distributed fault recovery requires prior access to the slicing scores, thus, it is necessary to first call [model.build](https://www.mindspore.cn/docs/zh-CN/master/api_python/train/mindspore.train.Model.html# mindspore.train.model.build) to compile and then perform the training.

```python
import os
import mindspore as ms
def recover_train():
    # model create
    # checkpoint load
    if args_opt.ckpt_file:
        param_dict = ms.load_checkpoint(args_opt.ckpt_file)
        model.build(train_dataset=dataset, epoch=4)
        ms.load_param_into_net(net, param_dict)
    model.train(2, dataset, callbacks=callback, dataset_sink_mode=True)
```

## Running the Code

First, please refer to the [Preparation Session](https://www.mindspore.cn/tutorials/experts/en/master/parallel/transformer.html#Preparation) in the Distributed Parallel Training Transformer Model tutorial to prepare the dataset.
After entering the code directory, execute the training script that saves the slice weights.

```bash
bash run_parallel_save_ckpt.sh DATASET_PATH
```

Then, the fault recovery training script is executed.

```bash
bash run_parallel_recover_ckpt.sh DATASET_PATH
```

After the recovery training, the loss is as follows. You can see that the loss starts to drop directly from 6.465892, indicating that the loading is successful.

```text
epoch: 1 step: 77, loss is 6.465892
epoch: 1 step: 77, loss is 6.239279
```
