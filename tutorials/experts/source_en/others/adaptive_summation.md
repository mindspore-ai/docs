# Adaptive Gradient Summation Algorithm

<a href="https://gitee.com/mindspore/docs/blob/master/tutorials/experts/source_en/others/adaptive_summation.md" target="_blank"><img src="https://mindspore-website.obs.cn-north-4.myhuaweicloud.com/website-images/master/resource/_static/logo_source.png"></a>

## Overview

This tutorial shows how to use the adaptive gradient summation algorithm [Scaling Distributed Training with Adaptive Summation](https://arxiv.org/pdf/2006.02924.pdf) in distributed training to improve the critical batch size of network training and speed up network convergence.

In traditional distributed training, after each compute node calculates the loss and gradient, the gradients of all nodes are averaged, and then the gradient is updated.

Unlike gradient renewal in traditional distributed training, adaptive gradient summation takes the direction of the gradient into account. In the early stage of network training, the gradient update direction obtained by different batches is basically parallel, but as the training progresses, the gradient update direction tends to be orthogonal. Moreover, the difference in orthogonality of gradient updates at different layers of the network is also relatively large.

Taking two training nodes as an example, the update principle of the gradient is as follows:
$$
\begin{aligned}
w^{’} &= w_0 - \alpha \cdot [(1 - \frac{g^T_2 \cdot g_1}{2 \cdot ||g_1||^2}) \cdot g_1 + (1 - \frac{g^T_2 \cdot g_1}{2 \cdot ||g_2||^2}) \cdot g_2] \\
&= w_0 - \alpha \cdot Adasum(g_1,g_2)
\end{aligned}
$$
where $g_1$ is the gradient of training node 1, and $g_2$ is the gradient of training node 2. When the training node expands to $n$($n = 2^x, x = 1,2,3 cdots$), the problem is decomposed recursively as follows:
$$
Adasum(g_{|0,n|}) = Adasum(Adasum(g_{|0, n/2|}), Adasum(g_{|n/2, n|}))
$$
As can be seen from the above formulas, the paper is an update to the gradient. Considering that the optimizer's operation on the gradient does not necessarily satisfy the linear conversion, the optimization is to do adasum operation on the network weight difference (delta weights) after the optimizeizer.

This tutorial will show you how to implement adaptive gradient summation in Boost mode, taking the ttraining process of ResNet-50 on the ImageNet 2012 dataset on Ascend910 as an example. `mindspore.boost` integrates various algorithms for network training acceleration, and provides a configuration interface to start the acceleration algorithm.

It should be noted that after experimental verification, in small distributed training (such as 2 nodes in this experiment), the effect of adasum experiment is not obvious, and the effect will be more obvious as the number of nodes increases. This tutorial is only to illustrate how to use adasum, so use the 2-node as an example.

## Preparation

>Download the complete sample code from:
>
><https://gitee.com/mindspore/docs/tree/master/docs/sample_code/adasum>.
>
>The models library links referenced in the code:
>
><https://gitee.com/mindspore/models>

The directory structure is as follows:

```text
└─sample_code
    ├─adasum
    │      rank_table_16pcs.json
    │      resnet.py
    │      training.py
    │      run_node1.sh
    │      run_node2.sh
```

where rank_table_16pcs.jsons are networking information files for a Doka environment, resnet.py and train.py are files that define the network structure, and run_node1.py and run_node2.py are running scripts.

### Configuring the Distributed Environment Variables

In this tutorial, the configuration information of the json file is as follows, taking 2 nodes and 16-card environment as an example:

```json
{
  "version": "1.0",
  "server_count": "2",
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
    },
    {
      "server_id": "10.155.111.141",
      "device": [
        {"device_id": "0","device_ip": "192.1.27.8","rank_id": "8"},
        {"device_id": "1","device_ip": "192.2.27.8","rank_id": "9"},
        {"device_id": "2","device_ip": "192.3.27.8","rank_id": "10"},
        {"device_id": "3","device_ip": "192.4.27.8","rank_id": "11"},
        {"device_id": "4","device_ip": "192.1.27.9","rank_id": "12"},
        {"device_id": "5","device_ip": "192.2.27.9","rank_id": "13"},
        {"device_id": "6","device_ip": "192.3.27.9","rank_id": "14"},
        {"device_id": "7","device_ip": "192.4.27.9","rank_id": "15"}],
      "host_nic_ip": "reserve"
    }
  ],
  "status": "completed"
}
```

rank_table can be generated by using the [hccl_tools.py](https://gitee.com/mindspore/models/blob/master/utils/hccl_tools/hccl_tools.py) below the models, and [merge_hccl. py](https://gitee.com/mindspore/models/blob/master/utils/hccl_tools/merge_hccl.py) stitches multiple rank_table files. The script usage method can be seen [README.md](https://gitee.com/mindspore/models/blob/master/utils/hccl_tools/README.md#).

### Preparing the Dataset

Used dataset: [ImageNet 2012](http://www.image-net.org/)

- The size of the dataset: 1000 classes and 224*224 color images in totoal

    - Training set: 1,281,167 images in total

    - Test set: 50000 images in total

- Data format: JPEG
- Download the dataset, and the directory structure is as follows:

```text
└─dataset
    ├─train                 # Training the dataset
    └─validation_preprocess # Evaluating the dataset
```

## Configuring the Operating Mode

Specify the operating mode, running card number, parallel mode through the context interface provided by MindSpore, and initialize the HCCL communication through init.

```python
set_context(mode=GRAPH_MODE, device_target="Ascend")
device_id = int(os.getenv('DEVICE_ID'))
set_context(device_id=device_id)
set_auto_parallel_context(device_num=16, parallel_mode=ParallelMode.DATA_PARALLEL, gradients_mean=True)
set_algo_parameters(elementwise_op_strategy_follow=True)
init()
```

## Loading Dataset in the Data Parallel Mode

During distributed training, data is imported in parallel with the data. Image loading interface ImageFolderDataset is used to load the ImageNet 2012 dataset by using MindSpore. The dataset is processed through the data augmentation interface provided by MindSpore, and this part of the code is imported by [dataset.py](https://gitee.com/mindspore/models/blob/master/official/cv/resnet/src/dataset.py) in the `resnet` directory in the models.

```python
# define train dataset
train_data_path = os.path.join(args.data_path, "train")
ds_train = create_dataset(dataset_path=train_data_path, do_train=True, batch_size=256, train_image_size=224,
                          eval_image_size=224, target="Ascend", distribute=True)
step_size = ds_train.get_dataset_size()
```

## Defining the Network

The build code for the ResNet-50 network is imported by [resnet.py](https://gitee.com/mindspore/docs/blob/master/docs/sample_code/adasum/resnet.py).

```python
# define net
net = resnet(num_classes=1001)
init_weight(net=net)
```

## Defining the Training Model

When defining the network, we need to set the boost_level to "O2" to turn on the adasum algorithm.

```python
# define loss
loss = CrossEntropySmooth(sparse=True, reduction="mean", smooth_factor=0.1, num_classes=1001)
loss_scale = FixedLossScaleManager(1024, drop_overflow_update=False)

# define optimizer
group_params = init_group_params(net)
lr = get_lr(lr_init=0, lr_end=0.0, lr_max=0.8, warmup_epochs=5, total_epochs=90, steps_per_epoch=step_size,
            lr_decay_mode="linear")
lr = Tensor(lr)
opt = Momentum(group_params, lr, 0.9, loss_scale=1024)

# define model
model = Model(net, loss_fn=loss, optimizer=opt, loss_scale_manager=loss_scale, amp_level="O2", boost_level="O2",
              keep_batchnorm_fp32=False)
# define eval_network
dist_eval_network = ClassifyCorrectCell(net)
```

It should be noted that the "O2" mode includes other acceleration algorithms, and if we only want to turn on adasum, we can do it by configuring boost_config_dict.

```python
# define boost config dictionary
boost_dict = {
    "boost": {
        "mode": "manual",
        "less_bn": False,
        "grad_freeze": False,
        "adasum": True,
        "grad_accumulation": False,
        "dim_reduce": False
    }
}

# define model
model = Model(net, loss_fn=loss, optimizer=opt, loss_scale_manager=loss_scale, amp_level="O2", boost_level="O2",
              keep_batchnorm_fp32=False, boost_config_dict=boost_dict, eval_network=dist_eval_network)
```

## Training the Model

Before the training starts, define the callback function callback, and add the training time information output and the loss information output.

```python
# define callback
cb = [TimeMonitor(data_size=step_size), LossMonitor()]

print("============== Starting Training ==============")
model.train(90, ds_train, callbacks=cb, sink_size=step_size, dataset_sink_mode=True)
```

## Running Script

2-host 16-card training model runs the script run_node1.sh on machine 1, and runs the script run_node2.sh on machine 2.

```bash
bash run_node{i}.sh ./imagenet
```

The core configuration for running the script is as follows, which needs to be modified when running a machine amplification. RANK_TABLE_FILE is the card environment profile, RANK_SIZE is the total number of cards, DEVICE_NUM is the number of cards per machine, and SERVER_ID is the serial number of the current machine.

```bash
export RANK_TABLE_FILE=${EXEC_PATH}/rank_table_16pcs.json
export RANK_SIZE=16
export DEVICE_NUM=8

export SERVER_ID=0
rank_start=$((DEVICE_NUM * SERVER_ID))
```

The output is as follows, and you can see that the loss value gradually decreases with the training:

```text
============== Starting Training ==============
epoch: 1 step: 312 loss is  5.5303826
...
epoch: 10 step: 312 loss is  3.3762435
...
...
...

```

