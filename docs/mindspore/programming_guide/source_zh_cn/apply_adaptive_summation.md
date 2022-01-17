# 应用自适应梯度求和算法

`Ascend` `模型调优` `分布式训练`

<a href="https://gitee.com/mindspore/docs/blob/master/docs/mindspore/programming_guide/source_zh_cn/apply_adaptive_summation.md" target="_blank"><img src="https://gitee.com/mindspore/docs/raw/master/resource/_static/logo_source.png"></a>

## 概述

本教程介绍在分布式训练中，如何使用自适应梯度求和算法 [Scaling Distributed Training with Adaptive Summation](https://arxiv.org/pdf/2006.02924.pdf)，提升网络训练的临界批量（critical batch size），并加快网络收敛。

传统的分布式训练中，每个计算节点计算得到loss和梯度后，会将所有节点的梯度求均值，然后进行梯度更新。

与传统的分布式训练中的梯度更新不同，自适应梯度求和考虑到梯度的方向。在网络训练初期，不同batch获得的梯度更新方向基本是平行的，但是随着训练进行，梯度更新方向趋向于正交。而且网络的不同层梯度更新的正交性差异也是比较大的。

以两个训练节点为例，梯度的更新原理如下：

$$
\begin{aligned}
w^{’} &= w_0 - \alpha \cdot [(1 - \frac{g^T_2 \cdot g_1}{2 \cdot ||g_1||^2}) \cdot g_1 + (1 - \frac{g^T_2 \cdot g_1}{2 \cdot ||g_2||^2}) \cdot g_2] \\
&= w_0 - \alpha \cdot Adasum(g_1,g_2)
\end{aligned}
$$

其中，$g_1$ 是训练节点1的梯度，$g_2$ 是训练节点2的梯度。当训练节点拓展到 $n$（$n = 2^x, x = 1,2,3 \cdots$） 个时，采用递归的方式来对问题进行分解，递归公式如下：

$$
Adasum(g_{|0,n|}) = Adasum(Adasum(g_{|0, n/2|}), Adasum(g_{|n/2, n|}))
$$

从上述公式中可见，论文中是对梯度更新，考虑到优化器（optimizer）对梯度的操作不一定满足线性转换，因此优化为对经过optimizer后的网络权重差值（delta weights）做adasum操作。

本篇教程将在Ascend910上，以ResNet-50在ImageNet 2012数据集上的训练过程为例，介绍在Boost模式下如何实现自适应梯度求和。`mindspore.boost`中集合了网络训练加速的各类算法，并对外提供配置接口来开启加速算法。

需要注意的是，经实验验证，在小型分布式训练中（例如本实验中2个节点），adasum实验效果不明显，随着节点数的增加，效果也会越明显。本教程仅为了说明如何使用adasum，因此以2节点为例进行说明。

## 准备环节

> 你可以在这里下载完整的样例代码：
>
> <https://gitee.com/mindspore/docs/tree/master/docs/sample_code/adasum>
>
> 代码中引用到的models库链接：
>
> <https://gitee.com/mindspore/models>

目录结构如下：

```text
└─sample_code
    ├─adasum
    │      rank_table_16pcs.json
    │      resnet.py
    │      training.py
    │      run_node1.sh
    │      run_node2.sh
```

其中，rank_table_16pcs.jsons是多卡环境的组网信息文件，resnet.py和train.py是定义网络结构的文件，run_node1.py和run_node2.py是执行脚本。

### 配置分布式环境变量

本教程以2个节点，16卡环境为例，json文件的配置信息如下：

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

rank_table可以使用models下面的[hccl_tools.py](https://gitee.com/mindspore/models/blob/master/utils/hccl_tools/hccl_tools.py)生成，[merge_hccl.py](https://gitee.com/mindspore/models/blob/master/utils/hccl_tools/merge_hccl.py)可将多个rank_table文件进行拼接。脚本使用方法可见[README.md](https://gitee.com/mindspore/models/blob/master/utils/hccl_tools/README.md)。

### 数据集准备

使用的数据集：[ImageNet 2012](http://www.image-net.org/)

- 数据集大小：共1000个类、224*224彩色图像
    - 训练集：共1,281,167张图像
    - 测试集：共50,000张图像
- 数据格式：JPEG
- 下载数据集，目录结构如下：

```text
└─dataset
    ├─train                 # 训练数据集
    └─validation_preprocess # 评估数据集
```

## 运行模式配置

通过MindSpore提供的context接口指定运行模式、运行卡号、并行模式等，通过init初始化HCCL通信。

```python
context.set_context(mode=context.GRAPH_MODE, device_target="Ascend")
device_id = int(os.getenv('DEVICE_ID'))
context.set_context(device_id=device_id)
context.set_auto_parallel_context(device_num=16, parallel_mode=ParallelMode.DATA_PARALLEL, gradients_mean=True)
set_algo_parameters(elementwise_op_strategy_follow=True)
init()
```

## 数据并行模式加载数据集

分布式训练时，数据以数据并行的方式导入。利用MindSpore提供图片加载接口ImageFolderDataset加载ImageNet 2012数据集，同时通过MindSpore提供的数据增强接口对数据集进行处理，此部分代码由models中`resnet`目录下的[dataset.py](https://gitee.com/mindspore/models/blob/master/official/cv/resnet/src/dataset.py)导入。

```python
# define train dataset
train_data_path = os.path.join(args.data_path, "train")
ds_train = create_dataset(dataset_path=train_data_path, do_train=True, batch_size=256, train_image_size=224,
                          eval_image_size=224, target="Ascend", distribute=True)
step_size = ds_train.get_dataset_size()
```

## 定义网络

ResNet-50网络的构建代码由[resnet.py](https://gitee.com/mindspore/docs/blob/master/docs/sample_code/adasum/resnet.py)导入。

```python
# define net
net = resnet(num_classes=1001)
init_weight(net=net)
```

## 定义训练模型

在定义网络的时候，我们需要将boost_level设置为"O2"来开启adasum算法。

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

值得注意的是，”O2"模式包含了其他的加速算法，如果我们只想开启adasum，我们可以通过配置boost_config_dict来实现。

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

## 训练模型

训练开始前，定义回调函数callback，添加训练时间信息输出，loss信息输出。

```python
# define callback
cb = [TimeMonitor(data_size=step_size), LossMonitor()]

print("============== Starting Training ==============")
model.train(90, ds_train, callbacks=cb, sink_size=step_size, dataset_sink_mode=True)
```

## 运行脚本

2机16卡训练模型，在机器1上运行脚本run_node1.sh，在机器2上运行脚本run_node2.sh。

```basH
bash run_node{i}.sh ./imagenet
```

运行脚本的核心配置如下，当运行机器扩增时，需要进行修改。其中RANK_TABLE_FILE是卡环境配置文件，RANK_SIZE是总的卡数，DEVICE_NUM是每台机器的卡数，SERVER_ID是当前机器的序号。

```bash
export RANK_TABLE_FILE=${EXEC_PATH}/rank_table_16pcs.json
export RANK_SIZE=16
export DEVICE_NUM=8

export SERVER_ID=0
rank_start=$((DEVICE_NUM * SERVER_ID))
```

输出如下，可以看到loss值随着训练逐步降低：

```text
============== Starting Training ==============
epoch: 1 step: 312 loss is  5.5303826
...
epoch: 10 step: 312 loss is  3.3762435
...
...
...
