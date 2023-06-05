# Distributed Fault Recovery

<a href="https://gitee.com/mindspore/docs/blob/master/tutorials/experts/source_en/parallel/fault_recover.md" target="_blank"><img src="https://mindspore-website.obs.cn-north-4.myhuaweicloud.com/website-images/master/resource/_static/logo_source_en.png"></a>

## Overview

It is very common to encounter failures when performing distributed training, similar to single-card training, which can be continued by loading the saved weight information during training. Distinct from pure data parallel training, when model parallelism is applied, the weights are sliced and the weight information saved between cards may not be consistent.
To solve this problem, one option is to aggregate the weights through the [AllGather](https://www.mindspore.cn/docs/en/master/api_python/samples/ops/communicate_ops.html#allgather) before saving the weight checkpoint file, where each card stores a complete information about the weights. This one function has been introduced in [Distributed training model parameter saving and loading](https://www.mindspore.cn/tutorials/experts/en/master/parallel/train_ascend.html#saving-and-loading-distributed-training-model-parameters).
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

Distributed fault recovery requires prior access to the slicing scores, thus, it is necessary to first call [model.build](https://www.mindspore.cn/docs/zh-CN/master/api_python/train/mindspore.train.Model.html#mindspore.train.Model.build) to compile and then perform the training.

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

## Preparation

### Downloading the Dataset

- [Download WMT14 En-Fr dataset](http://statmt.org/wmt14/test-full.tgz). If you download unsuccessfully to click the link, please try to download it after copying the link address.

Use `newstest2014-fren-ref.en.sgm` as the training set for this task, combine and clean this dataset. Extract the dataset to the `docs/sample_code/distributed_training_transformer` directory.

### Pre-processing

Executing the following code to pre-process the data will generate the `output` directory in the current directory, which will produce the files `wmt14.en_fr.txt` and `wmt14.fr_en.txt`, each line of which is a sentence pair in French and English. We will use `wmt14.fr_en.txt` as the training data.

```python
python preprocess.py
```

### Configuring the Distributed Environment Variables

When performing distributed training in a bare-metal environment (compared to an on-cloud environment, i.e. with Ascend 910 AI processors locally), you need to configure the networking information file for the current multi-card environment. If you use Huawei cloud environment, you can skip this subsection because the cloud service itself is well configured.

Taking Ascend 910 AI processor as an example, a sample json configuration file for one 8-card environment is as follows. This sample names the configuration file as `rank_table_8pcs.json`. 2-card environment configuration can refer to the `rank_table_2pcs.json` file in the sample code.

```json
{
    "version": "1.0",
    "server_count": "1",
    "server_list": [
        {
            "server_id": "10.155.111.140",
            "device": [
                {"device_id": "0","device_ip": "192.1.27.6","rank_id": "0"},
                {"device_id": "1","device_ip": "192.2.27.6","rank_id": "1"},
                {"device_id": "2","device_ip": "192.3.27.6","rank_id": "2"},
                {"device_id": "3","device_ip": "192.4.27.6","rank_id": "3"},
                {"device_id": "4","device_ip": "192.1.27.7","rank_id": "4"},
                {"device_id": "5","device_ip": "192.2.27.7","rank_id": "5"},
                {"device_id": "6","device_ip": "192.3.27.7","rank_id": "6"},
                {"device_id": "7","device_ip": "192.4.27.7","rank_id": "7"}],
             "host_nic_ip": "reserve"
        }
    ],
    "status": "completed"
}
```

Among the parameter items that need to be modified according to the actual training environment are:

- `server_count` represents the number of machines involved in the training.
- `server_id` represents IP address of the current machine.
- `device_id` represents the physical serial number of the card, i.e. the actual serial number in the machine where the card is located.
- `device_ip` represents the IP address of the integration NIC. You can execute the command `cat /etc/hccn.conf` on the current machine, and the key value of `address_x` is the NIC IP address.
- `rank_id` represents the logic serial number of card, fixed numbering from 0.

### Calling the Collective Communication Repository

The MindSpore distributed parallel training communication uses the Huawei Collective Communication Library `Huawei Collective Communication Library` (hereinafter referred to as HCCL), which can be found in the package that accompanies the Ascend AI processor. `mindspore.communication.management` encapsulates the collective communication interface provided by HCCL to facilitate user configuration of distributed information.
> HCCL implements multi-machine multi-card communication based on Ascend AI processor. There are some usage restrictions. We list the common ones by using distributed services, and you can check the corresponding usage documentation of HCCL for details.
>
> - Support 1, 2, 4, 8-card device clusters in single machine scenario and 8*n-card device clusters in multi-machine scenario.
> - 0-3 cards and 4-7 cards of each machine are consisted of two clusters respectively. 2-card and 4-card must be connected and do not support cross-group creation of clusters during training.
> - When building a multi-machine cluster, you need to ensure that each machine uses the same switch.
> - The server hardware architecture and operating system needs to be SMP (Symmetrical Multi-Processing) processing mode.

The following is sample code for calling the collection communication repository:

```python
import os
from mindspore.communication import init
import mindspore as ms

if __name__ == "__main__":
    ms.set_context(mode=ms.GRAPH_MODE, device_target="Ascend", device_id=int(os.environ["DEVICE_ID"]))
    init()
    ...
```

where,

- `mode=GRAPH_MODE`: Using distributed training requires specifying the run mode as graph mode (PyNative mode does not support parallelism).
- `device_id`: The physical serial number of the card, i.e. the actual serial number in the machine where the card is located.
- `init`: Enables HCCL communication and completes distributed training initialization operations.

## Running the Code

After preparing the dataset and entering the code directory, execute the training script that saves the slice weights.

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
