# 降维训练算法

 `Ascend` `模型调优` `分布式并行`

<a href="https://gitee.com/mindspore/docs/blob/r1.6/docs/mindspore/programming_guide/source_zh_cn/apply_dimention_reduce_training.md" target="_blank"><img src="https://gitee.com/mindspore/docs/raw/r1.6/resource/_static/logo_source.png"></a>

## 概述

本教程介绍降维训练的训练方式，目的是为了解决网络在训练后期收敛缓慢的问题。

一般网络训练，前期收敛比较快，后期收敛进入稳定阶段，收敛较慢。为了提升后期的收敛速度，降维训练将网络训练分为两个阶段。第一阶段，以传统的方式训练网络，保留N（N>32）个权重文件，建议放弃比较初期的权重文件，比如第一阶段训练50个epoch，从第21个epoch开始保存权重文件，每个epoch保存2个权重文件，则第一阶段结束有60个权重文件。第二阶段，加载第一阶段得到的权重文件，做PCA（Principal Component Analysis）降维，将权重从高维（M）降到低维（32），在低维上搜索权重的梯度下降方向和长度，然后反投影到高维，更新权重。

鉴于单卡训练比较耗时，本篇教程将在Ascend 910 AI处理器硬件平台上，利用数据并行模式，以ResNet-50在ImageNet 2012上的训练过程为例，介绍第一阶段的常规训练，以及在Boost模式下如何实现第二阶段的降维训练。

## 准备环节

> 你可以在这下载完整的样例代码：
>
> <https://gitee.com/mindspore/docs/tree/r1.6/docs/sample_code/dimension_reduce_training>
>
> 代码中引用到的models库链接：
>
> <https://gitee.com/mindspore/models>

### 配置分布式环境变量

在本地Ascend处理器上进行分布式训练时，需要配置当前多卡环境的组网信息文件，1个8卡环境的json文件配置如下，本样例将该配置文件命名为rank_table_8pcs.json。rank_table可以使用models下面的[hccl_tools.py](https://gitee.com/mindspore/models/blob/r1.6/utils/hccl_tools/hccl_tools.py)生成。

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

### 数据集准备

使用的数据集：[ImageNet2012](http://www.image-net.org/)

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

## 第一阶段：常规训练

> 第一阶段主要的训练样例代码：
>
> <https://gitee.com/mindspore/docs/blob/r1.6/docs/sample_code/dimension_reduce_training/train_stage_1.py>

### 运行模式和后端设备设置

通过MindSpore提供的context接口指定运行模式、运行卡号、并行模式等，通过init初始化HCCL通信。

```python
context.set_context(mode=context.GRAPH_MODE, device_target=args.device_target)
device_id = int(os.getenv('DEVICE_ID'))
context.set_context(device_id=device_id)
context.set_auto_parallel_context(device_num=8, parallel_mode=ParallelMode.DATA_PARALLEL, gradients_mean=True)
set_algo_parameters(elementwise_op_strategy_follow=True)
all_reduce_fusion_config = [85, 160]
context.set_auto_parallel_context(all_reduce_fusion_config=all_reduce_fusion_config)
init()
```

### 加载数据集

利用MindSpore提供图片加载接口ImageFolderDataset加载ImageNet 2012数据集，同时通过MindSpore提供的数据增强接口对数据集进行处理，此部分代码由models中`resnet`目录下的[dataset.py](https://gitee.com/mindspore/models/blob/r1.6/official/cv/resnet/src/dataset.py)导入。

```python
# define train dataset
train_data_path = os.path.join(args.data_path, "train")
ds_train = create_dataset(dataset_path=train_data_path, do_train=True, batch_size=256, train_image_size=224,
                          eval_image_size=224, target=args.device_target, distribute=True)
step_size = ds_train.get_dataset_size()
```

### 定义网络

ResNet-50网络的构建代码由[resnet.py](https://gitee.com/mindspore/docs/blob/r1.6/docs/sample_code/dimension_reduce_training/resnet.py)导入。

```python
# define net
net = resnet(class_num=1001)
init_weight(net=net)
```

### 定义训练模型

定义模型所需的损失函数loss、optimizer等。

loss使用CrossEntropySmooth，由ModelZoo中`resnet`目录下的[CrossEntropySmooth.py](https://gitee.com/mindspore/models/blob/r1.6/official/cv/resnet/src/CrossEntropySmooth.py)导入。

学习率lr的构建代码由models中`resnet`目录下的[lr_generator.py](https://gitee.com/mindspore/models/blob/r1.6/official/cv/resnet/src/lr_generator.py)导入。

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

# define metrics
metrics = {"acc"}

# define model
model = Model(net, loss_fn=loss, optimizer=opt, loss_scale_manager=loss_scale, metrics=metrics, amp_level="O2",
              boost_level="O0", keep_batchnorm_fp32=False)
```

### 训练模型

训练开始前，定义回调函数callback，添加训练时间信息输出、loss信息输出，权重保存等，其中模型权重只在0卡上保存。

callback_1保存网络的所有权重和优化器的状态，callback_2仅保存网络中参加更新的权重，用于第二阶段的PCA。

```python
# define callback_1
cb = [TimeMonitor(data_size=step_size), LossMonitor()]
if get_rank_id() == 0:
    config_ck = CheckpointConfig(save_checkpoint_steps=step_size * 10, keep_checkpoint_max=10)
    ck_cb = ModelCheckpoint(prefix="resnet", directory="./checkpoint_stage_1", config=config_ck)
    cb += [ck_cb]

# define callback_2: save weights for stage 2
if get_rank_id() == 0:
    config_ck = CheckpointConfig(save_checkpoint_steps=step_size, keep_checkpoint_max=40,
                                 saved_network=net)
    ck_cb = ModelCheckpoint(prefix="resnet", directory="./checkpoint_stage_1/checkpoint_pca", config=config_ck)
    cb += [ck_cb]

print("============== Starting Training ==============")
model.train(70, ds_train, callbacks=cb, sink_size=step_size, dataset_sink_mode=True)
```

### 测试模型

首先定义测试模型，然后加载测试数据集，测试模型的精度。通过rank_id的判断，只在0卡上测试模型。

```python
if get_rank_id() == 0:
    print("============== Starting Testing ==============")
    eval_data_path = os.path.join(args.data_path, "val")
    ds_eval = create_dataset(dataset_path=eval_data_path, do_train=False, batch_size=256, target="Ascend")
    if ds_eval.get_dataset_size() == 0:
        raise ValueError("Please check dataset size > 0 and batch_size <= dataset size")

    acc = model.eval(ds_eval)
    print("============== {} ==============".format(acc))
```

### 实验结果

在经历了70轮epoch之后，在测试集上的精度约为66.05%。

1. 调用运行脚本[run_stage_1.sh](https://gitee.com/mindspore/docs/blob/r1.6/docs/sample_code/dimension_reduce_training/run_stage_1.sh)，查看运行结果。运行脚本需要给定数据集路径，模型默认保存在device0_stage_1/checkpoint_stage_1下。

   ```bash
   bash run_stage_1.sh ./imagenet
   ```

   输出如下，可以看到loss值随着训练逐步降低：

   ```text
   ============== Starting Training ==============
   epoch: 1 step: 625 loss is  5.2477064
   ...
   epoch: 10 step: 625 loss is  3.0178385
   ...
   epoch: 30 step: 625 loss is  2.4231198
   ...
   ...
   epoch: 70 step: 625 loss is  2.3120291
   ```

2. 查看推理精度。

   ```text
   ============== Starting Testing ==============
   ============== {'Accuracy': 0.6604992988782051} ==============
   ```

## 第二阶段：boost模式

基于第一阶段得到的权重文件，在Boost模式下，我们只要简单调用mindspore.boost的降维训练接口，即可实现降维训练的功能。

> 第二阶段主要的训练样例代码：
>
> <https://gitee.com/mindspore/docs/blob/r1.6/docs/sample_code/dimension_reduce_training/train_boost_stage_2.py>

第二阶段和第一阶段的代码基本相同，下面仅说明不一致的地方。

### 定义网络并加载初始化权重

构建网络后，加载第一阶段训练结束时的权重文件，在此基础上进行第二阶段的训练。

```python
# define net
net = resnet(class_num=1001)
if os.path.isfile(args.pretrained_weight_path):
    weight_dict = load_checkpoint(args.pretrained_weight_path)
load_param_into_net(net, weight_dict)
```

### 定义训练模型

和第一阶段不同的是，第二阶段使用SGD优化器来更新网络权重。除此之外，还需要配置降维训练所需的参数，该配置以字典的形式构建。首先需要将boost的类型设置为人工（manual）设置，同时打开降维训练（dim_reduce），然后配置每个服务器节点使用的卡数，最后配置降维训练的一些超参数。除此之外，在定义模型的时候，还需要打开boost_level（设置为"O1"或者”O2“）。

```python
# define loss
loss = CrossEntropySmooth(sparse=True, reduction="mean", smooth_factor=0.1, num_classes=1001)
loss_scale = FixedLossScaleManager(1024, drop_overflow_update=False)

# define optimizer
group_params = init_group_params(net)
opt = SGD(group_params, learning_rate=1, loss_scale=loss_scale)

# define metrics
metrics = {"acc"}

# define boost config dictionary
boost_dict = {
    "boost": {
        "mode": "manual",
        "dim_reduce": True
    },
    "common": {
        "device_num": 8
    },
    "dim_reduce": {
        "rho": 0.55,
        "gamma": 0.9,
        "alpha": 0.001,
        "sigma": 0.4,
        "n_component": 32,                                                 # PCA component
        "pca_mat_path": args.pca_mat_path,                                 # the path to load pca mat
        "weight_load_dir": "./checkpoint_stage_1/checkpoint_pca",          # the directory to load weight file saved as ckpt.
        "timeout": 1200
    }
}

# define model
model = Model(net, loss_fn=loss, optimizer=opt, loss_scale_manager=loss_scale, metrics=metrics,
              amp_level="O2", boost_level="O1", keep_batchnorm_fp32=False, boost_config_dict=boost_dict)
```

### 训练模型

第二阶段，我们设置epoch为2。

```python
# define callback
cb = [TimeMonitor(data_size=step_size), LossMonitor()]
if get_rank_id() == 0:
    config_ck = CheckpointConfig(save_checkpoint_steps=step_size, keep_checkpoint_max=2)
    ck_cb = ModelCheckpoint(prefix="resnet", directory="./checkpoint_stage_2", config=config_ck)
    cb += [ck_cb]

print("============== Starting Training ==============")
model.train(2, ds_train, callbacks=cb, sink_size=step_size, dataset_sink_mode=True)
```

### 实验结果

在经历了2轮epoch之后，在测试集上的精度约为74.31%。

1. 调用运行脚本[run_stage_2.sh](https://gitee.com/mindspore/docs/blob/r1.6/docs/sample_code/dimension_reduce_training/run_stage_2.sh)，查看运行结果。运行脚本需要给定数据集路径，第一阶段训练结束时的权重文件，第二阶段保存的权重文件默认保存在device0_stage_2/checkpoint_stage_2下。

   ```bash
   bash run_stage_2.sh ./imagenet ./device0_stage_1/checkpoint_stage_1/resnet-70_625.ckpt
   ```

   如果已经对第一阶段的权重做了PCA，并保存了特征转换矩阵，则可以给定特征转换矩阵文件路径，省去求特征转换矩阵的过程。

   ```bash
   bash run_stage_2.sh ./imagenet ./device0_stage_1/checkpoint_stage_1/resnet-70_625.ckpt /path/pca_mat.npy
   ```

   输出如下，可以看到loss值随着训练逐步降低：

   ```text
   epoch: 1 step: 625 loss is  2.3422508
   epoch: 2 step: 625 loss is  2.1641185
   ```

2. 查看推理精度，代码中会将checkpoint保存到当前目录，随后会加载该checkpoint执行推理。

   ```text
   ============== Starting Testing ==============
   ============== {'Accuracy': 0.7430964543269231} ==============
   ```

一般ResNet-50训练80个epoch也只达到70.18%，使用降维训练，我们可以在72个epoch达到74.31%。

