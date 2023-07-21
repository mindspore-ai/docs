# ResNet-50二阶优化实践

`Ascend` `GPU` `模型开发` `模型调优` `高级`
[![查看源文件](../_static/logo_source.png)](https://gitee.com/mindspore/docs/blob/r0.7/tutorials/source_zh_cn/advanced_use/second_order_optimizer_for_resnet50_application.md)&nbsp;&nbsp;

## 概述

常见的优化算法可分为一阶优化算法和二阶优化算法。经典的一阶优化算法如SGD等，计算量小、计算速度快，但是收敛的速度慢，所需的迭代次数多。而二阶优化算法使用目标函数的二阶导数来加速收敛，能更快地收敛到模型最优值，所需要的迭代次数少，但由于二阶优化算法过高的计算成本，导致其总体执行时间仍然慢于一阶，故目前在深度神经网络训练中二阶优化算法的应用并不普遍。二阶优化算法的主要计算成本在于二阶信息矩阵（Hessian矩阵、[FIM矩阵](https://arxiv.org/pdf/1808.07172.pdf)等）的求逆运算，时间复杂度约为$O(n^3)$。

MindSpore开发团队在现有的自然梯度算法的基础上，对FIM矩阵采用近似、切分等优化加速手段，极大的降低了逆矩阵的计算复杂度，开发出了可用的二阶优化器THOR。使用8块Ascend 910 AI处理器，THOR可以在72min内完成ResNet50-v1.5网络和ImageNet数据集的训练，相比于SGD+Momentum速度提升了近一倍。


本篇教程将主要介绍如何在Ascend 910 以及GPU上，使用MindSpore提供的二阶优化器THOR训练ResNet50-v1.5网络和ImageNet数据集。
> 你可以在这里下载完整的示例代码：
<https://gitee.com/mindspore/mindspore/tree/r0.7/model_zoo/official/cv/resnet_thor> 。

### 示例代码目录结构

```shell
├── resnet_thor
    ├── README.md
    ├── scripts                     
        ├── run_distribute_train.sh         # launch distributed training for Ascend 910
        └── run_eval.sh                     # launch inference for Ascend 910
        ├── run_distribute_train_gpu.sh     # launch distributed training for GPU
        └── run_eval_gpu.sh                 # launch inference for GPU
    ├── src                                  
        ├── crossentropy.py                 # CrossEntropy loss function
        ├── config.py                       # parameter configuration
        ├── dataset_helper.py               # dataset helper for minddata dataset
        ├── grad_reducer_thor.py            # grad reduce for thor
        ├── model_thor.py                   # model for train
        ├── resnet_thor.py                  # resnet50_thor backone
        ├── thor.py                         # thor optimizer
        ├── thor_layer.py                   # thor layer
        └── dataset.py                      # data preprocessing    
    ├── eval.py                             # infer script
    └── train.py                            # train script
    
```

整体执行流程如下：
1. 准备ImageNet数据集，处理需要的数据集；
2. 定义ResNet50网络；
3. 定义损失函数和THOR优化器；
4. 加载数据集并进行训练，训练完成后，查看结果及保存模型文件；
5. 加载保存的模型，进行推理。


## 准备环节

实践前，确保已经正确安装MindSpore。如果没有，可以通过[MindSpore安装页面](https://www.mindspore.cn/install)安装MindSpore。

### 准备数据集

下载完整的ImageNet2012数据集，将数据集解压分别存放到本地工作区的`ImageNet2012/ilsvrc`、`ImageNet2012/ilsvrc_eval`路径下。

目录结构如下：

```
└─ImageNet2012
    ├─ilsvrc
    │      n03676483/
    │      n04067472/
    │      n01622779/
    │      ......
    └─ilsvrc_eval
    │      n03018349/
    │      n02504013
    │      n07871810
    │      ......

```
### 配置分布式环境变量
#### Ascend 910
Ascend 910 AI处理器的分布式环境变量配置参考[分布式并行训练 (Ascend)](https://www.mindspore.cn/tutorial/zh-CN/r0.7/advanced_use/distributed_training_ascend.html#id4)。

#### GPU
GPU的分布式环境配置参考[分布式并行训练 (GPU)](https://www.mindspore.cn/tutorial/zh-CN/r0.7/advanced_use/distributed_training_gpu.html#id4)。


## 加载处理数据集

分布式训练时，通过并行的方式加载数据集，同时通过MindSpore提供的数据增强接口对数据集进行处理。加载处理数据集的脚本在源码的`src/dataset.py`脚本中。
```python
import os
import mindspore.common.dtype as mstype
import mindspore.dataset.engine as de
import mindspore.dataset.transforms.vision.c_transforms as C
import mindspore.dataset.transforms.c_transforms as C2
from mindspore.communication.management import init, get_rank, get_group_size

def create_dataset(dataset_path, do_train, repeat_num=1, batch_size=32, target="Ascend"):
    if target == "Ascend":
        device_num, rank_id = _get_rank_info()
    else:
        init("nccl")
        rank_id = get_rank()
        device_num = get_group_size()
    if device_num == 1:
        ds = de.ImageFolderDatasetV2(dataset_path, num_parallel_workers=8, shuffle=True)
    else:
        ds = de.ImageFolderDatasetV2(dataset_path, num_parallel_workers=8, shuffle=True,
                                     num_shards=device_num, shard_id=rank_id)

    image_size = 224
    mean = [0.485 * 255, 0.456 * 255, 0.406 * 255]
    std = [0.229 * 255, 0.224 * 255, 0.225 * 255]
    # define map operations
    if do_train:
        trans = [
            C.RandomCropDecodeResize(image_size, scale=(0.08, 1.0), ratio=(0.75, 1.333)),
            C.RandomHorizontalFlip(prob=0.5),
            C.Normalize(mean=mean, std=std),
            C.HWC2CHW()
        ]
    else:
        trans = [
            C.Decode(),
            C.Resize(256),
            C.CenterCrop(image_size),
            C.Normalize(mean=mean, std=std),
            C.HWC2CHW()
        ]
    type_cast_op = C2.TypeCast(mstype.int32)
    ds = ds.map(input_columns="image", num_parallel_workers=8, operations=trans)
    ds = ds.map(input_columns="label", num_parallel_workers=8, operations=type_cast_op)

    # apply batch operations
    ds = ds.batch(batch_size, drop_remainder=True)

    # apply dataset repeat operation
    ds = ds.repeat(repeat_num)

    return ds
```

> MindSpore支持进行多种数据处理和增强的操作，各种操作往往组合使用，具体可以参考[数据处理与数据增强](https://www.mindspore.cn/tutorial/zh-CN/r0.7/use/data_preparation/data_processing_and_augmentation.html)章节。


## 定义网络
本示例中使用的网络模型为ResNet50-v1.5，先定义[ResNet50网络](https://gitee.com/mindspore/mindspore/blob/r0.7/model_zoo/official/cv/resnet/src/resnet.py)，然后使用二阶优化器自定义的算子替换`Conv2d`和
和`Dense`算子。定义好的网络模型在在源码`src/resnet_thor.py`脚本中，自定义的算子`Conv2d_thor`和`Dense_thor`在`src/thor_layer.py`脚本中。

-  使用`Conv2d_thor`替换原网络模型中的`Conv2d`
-  使用`Dense_thor`替换原网络模型中的`Dense`

> 使用THOR自定义的算子`Conv2d_thor`和`Dense_thor`是为了保存模型训练中的二阶矩阵信息，新定义的网络与原网络模型的backbone一致。

网络构建完成以后，在`__main__`函数中调用定义好的ResNet50：
```python
...
from src.resnet_thor import resnet50
...
f __name__ == "__main__":
    ...
    # define the net
    net = resnet50(class_num=config.class_num, damping=damping, loss_scale=config.loss_scale,
                   frequency=config.frequency, batch_size=config.batch_size)
    ...
```


## 定义损失函数及THOR优化器


### 定义损失函数

MindSpore支持的损失函数有`SoftmaxCrossEntropyWithLogits`、`L1Loss`、`MSELoss`等。THOR优化器需要使用`SoftmaxCrossEntropyWithLogits`损失函数。

损失函数的实现步骤在`src/crossentropy.py`脚本中。这里使用了深度网络模型训练中的一个常用trick：label smoothing，通过对真实标签做平滑处理，提高模型对分类错误标签的容忍度，从而可以增加模型的泛化能力。
```python
class CrossEntropy(_Loss):
    """CrossEntropy"""
    def __init__(self, smooth_factor=0., num_classes=1000):
        super(CrossEntropy, self).__init__()
        self.onehot = P.OneHot()
        self.on_value = Tensor(1.0 - smooth_factor, mstype.float32)
        self.off_value = Tensor(1.0 * smooth_factor / (num_classes - 1), mstype.float32)
        self.ce = nn.SoftmaxCrossEntropyWithLogits()
        self.mean = P.ReduceMean(False)

    def construct(self, logit, label):
        one_hot_label = self.onehot(label, F.shape(logit)[1], self.on_value, self.off_value)
        loss = self.ce(logit, one_hot_label)
        loss = self.mean(loss, 0)
        return loss
```
在`__main__`函数中调用定义好的损失函数：

```python
...
from src.crossentropy import CrossEntropy
...
if __name__ == "__main__":
    ...
    # define the loss function
    if not config.use_label_smooth:
        config.label_smooth_factor = 0.0
    loss = CrossEntropy(smooth_factor=config.label_smooth_factor, num_classes=config.class_num)
    ...
```

### 定义优化器

THOR优化器的参数更新公式如下：

$$ \theta^{t+1} = \theta^t + \alpha F^{-1}\nabla E$$

参数更新公式中各参数的含义如下：
- $\theta$：网络中的可训参数；
- $t$：迭代次数；
- $\alpha$：学习率值，参数的更新步长；
- $F^{-1}$：FIM矩阵，在网络中计算获得；
- $\nabla E$：一阶梯度值。

从参数更新公式中可以看出，THOR优化器需要额外计算的是每一层的FIM矩阵，每一层的FIM矩阵就是之前在自定义的网络模型中计算获得的。FIM矩阵可以对每一层参数更新的步长和方向进行自适应的调整，加速收敛的同时可以降低调参的复杂度。

```python
...
if args_opt.device_target == "Ascend":
    from src.thor import THOR
else:
    from src.thor import THOR_GPU as THOR
...

if __name__ == "__main__":
    ...
    # learning rate setting
    lr = get_model_lr(0, config.lr_init, config.lr_decay, config.lr_end_epoch, step_size, decay_epochs=39)
    # define the optimizer
    net_opt = opt = THOR(filter(lambda x: x.requires_grad, net.get_parameters()), Tensor(lr), config.momentum,
               filter(lambda x: 'matrix_A' in x.name, net.get_parameters()),
               filter(lambda x: 'matrix_G' in x.name, net.get_parameters()),
               filter(lambda x: 'A_inv_max' in x.name, net.get_parameters()),
               filter(lambda x: 'G_inv_max' in x.name, net.get_parameters()),
               config.weight_decay, config.loss_scale)
    ...
```

## 训练网络

### 配置模型保存

MindSpore提供了callback机制，可以在训练过程中执行自定义逻辑，这里使用框架提供的`ModelCheckpoint`函数。
`ModelCheckpoint`可以保存网络模型和参数，以便进行后续的fine-tuning操作。
`TimeMonitor`、`LossMonitor`是MindSpore官方提供的callback函数，可以分别用于监控训练过程中单步迭代时间和`loss`值的变化。

```python
...
from mindspore.train.callback import ModelCheckpoint, CheckpointConfig, TimeMonitor, LossMonitor
...
if __name__ == "__main__":
    ...
    # define callbacks
    time_cb = TimeMonitor(data_size=step_size)
    loss_cb = LossMonitor()
    cb = [time_cb, loss_cb]
    if config.save_checkpoint:
        config_ck = CheckpointConfig(save_checkpoint_steps=config.save_checkpoint_epochs * step_size,
                                     keep_checkpoint_max=config.keep_checkpoint_max)
        ckpt_cb = ModelCheckpoint(prefix="resnet", directory=ckpt_save_dir, config=config_ck)
        cb += [ckpt_cb]
    ...
```

### 配置训练网络

通过MindSpore提供的`model.train`接口可以方便地进行网络的训练。THOR优化器通过降低二阶矩阵更新频率，来减少计算量，提升计算速度，故重新定义一个Model_Thor类，继承MindSpore提供的Model类。在Model_Thor类中增加二阶矩阵更新频率控制参数，用户可以通过调整该参数，优化整体的性能。


```python
...
from mindspore.train.loss_scale_manager import FixedLossScaleManager
from src.model_thor import Model_Thor as Model
...

if __name__ == "__main__":
    ...
    loss_scale = FixedLossScaleManager(config.loss_scale, drop_overflow_update=False)
    if target == "Ascend":
        model = Model(net, loss_fn=loss, optimizer=opt, amp_level='O2', loss_scale_manager=loss_scale,
                      keep_batchnorm_fp32=False, metrics={'acc'}, frequency=config.frequency)
    else:
        model = Model(net, loss_fn=loss, optimizer=opt, loss_scale_manager=loss_scale, metrics={'acc'},
                      amp_level="O2", keep_batchnorm_fp32=True, frequency=config.frequency)
    ...
```

### 运行脚本
训练脚本定义完成之后，调`scripts`目录下的shell脚本，启动分布式训练进程。
#### Ascend 910
目前MindSpore分布式在Ascend上执行采用单卡单进程运行方式，即每张卡上运行1个进程，进程数量与使用的卡的数量一致。其中，0卡在前台执行，其他卡放在后台执行。每个进程创建1个目录，目录名称为`train_parallel`+ `device_id`，用来保存日志信息，算子编译信息以及训练的checkpoint文件。下面以使用8张卡的分布式训练脚本为例，演示如何运行脚本：

使用以下命令运行脚本：
```
sh run_distribute_train.sh [RANK_TABLE_FILE] [DATASET_PATH] [DEVICE_NUM]
```
脚本需要传入变量`RANK_TABLE_FILE`、`DATASET_PATH`和`DEVICE_NUM`，其中：
- `RANK_TABLE_FILE`：组网信息文件的路径。
- `DATASET_PATH`：训练数据集路径。
- `DEVICE_NUM`: 实际的运行卡数。
其余环境变量请参考安装教程中的配置项。

训练过程中loss打印示例如下：

```bash
...
epoch: 1 step: 5004, loss is 4.4182425
epoch: 2 step: 5004, loss is 3.740064
epoch: 3 step: 5004, loss is 4.0546017
epoch: 4 step: 5004, loss is 3.7598825
epoch: 5 step: 5004, loss is 3.3744206
......
epoch: 40 step: 5004, loss is 1.6907625
epoch: 41 step: 5004, loss is 1.8217756
epoch: 42 step: 5004, loss is 1.6453942
...
```

训练完后，每张卡训练产生的checkpoint文件保存在各自训练目录下，`device_0`产生的checkpoint文件示例如下:

```bash
└─train_parallel0
    ├─resnet-1_5004.ckpt
    ├─resnet-2_5004.ckpt
    │      ......
    ├─resnet-42_5004.ckpt
```

其中，
`*.ckpt`：指保存的模型参数文件。checkpoint文件名称具体含义：*网络名称*-*epoch数*_*step数*.ckpt。

##### GPU
在GPU硬件平台上，MindSpore采用OpenMPI的`mpirun`进行分布式训练，进程创建1个目录，目录名称为`train_parallel`，用来保存日志信息和训练的checkpoint文件。下面以使用8张卡的分布式训练脚本为例，演示如何运行脚本：
```
sh run_distribute_train_gpu.sh [DATASET_PATH] [DEVICE_NUM]
```
脚本需要传入变量`DATASET_PATH`和`DEVICE_NUM`，其中：
- `DATASET_PATH`：训练数据集路径。
- `DEVICE_NUM`: 实际的运行卡数。

在GPU训练时，无需设置`DEVICE_ID`环境变量，因此在主训练脚本中不需要调用`int(os.getenv('DEVICE_ID'))`来获取卡的物理序号，同时`context`中也无需传入`device_id`。我们需要将device_target设置为GPU，并需要调用`init("nccl")`来使能NCCL。

训练过程中loss打印示例如下：
```bash
...
epoch: 1 step: 5004, loss is 4.3069
epoch: 2 step: 5004, loss is 3.5695
epoch: 3 step: 5004, loss is 3.5893
epoch: 4 step: 5004, loss is 3.1987
epoch: 5 step: 5004, loss is 3.3526
......
epoch: 40 step: 5004, loss is 1.9482
epoch: 41 step: 5004, loss is 1.8950
epoch: 42 step: 5004, loss is 1.9023
...
```

训练完后，保存的模型文件示例如下:

```bash
└─train_parallel
    ├─ckpt_0
        ├─resnet-1_5004.ckpt
        ├─resnet-2_5004.ckpt
    	│      ......
        ├─resnet-42_5004.ckpt
	......
    ├─ckpt_7
        ├─resnet-1_5004.ckpt
        ├─resnet-2_5004.ckpt
        │      ......
        ├─resnet-42_5004.ckpt

```

## 模型推理

使用训练过程中保存的checkpoint文件进行推理，验证模型的泛化能力。首先通过`load_checkpoint`接口加载模型文件，然后调用`Model`的`eval`接口对输入图片类别作出预测，再与输入图片的真实类别做比较，得出最终的预测精度值。

### 定义推理网络

1. 使用`load_checkpoint`接口加载模型文件。
2. 使用`model.eval`接口读入测试数据集，进行推理。
3. 计算得出预测精度值。

```python
...
from mindspore.train.serialization import load_checkpoint, load_param_into_net
...

if __name__ == "__main__":
    ...
    # define net
    net = resnet(class_num=config.class_num)
    net.add_flags_recursive(thor=False)

    # load checkpoint
    param_dict = load_checkpoint(args_opt.checkpoint_path)
    keys = list(param_dict.keys())
    for key in keys:
        if "damping" in key:
            param_dict.pop(key)
    load_param_into_net(net, param_dict)
    net.set_train(False)

    # define model
    model = Model(net, loss_fn=loss, metrics={'top_1_accuracy', 'top_5_accuracy'})
	
    # eval model
    res = model.eval(dataset)
    print("result:", res, "ckpt=", args_opt.checkpoint_path)
```

### 执行推理
推理网络定义完成之后，调用/scripts目录下的shell脚本，进行推理。
#### Ascend 910
在Ascend 910硬件平台上，推理的执行命令如下：
```
sh run_eval.sh [DATASET_PATH] [CHECKPOINT_PATH]
```
脚本需要传入变量`DATASET_PATH`和`CHECKPOINT_PATH`，其中：
- `DATASET_PATH`：推理数据集路径。
- `CHECKPOINT_PATH`: 保存的checkpoint路径。

目前推理使用的是单卡（默认device 0）进行推理，推理的结果如下：
```
result: {'top_5_accuracy': 0.9295574583866837, 'top_1_accuracy': 0.761443661971831} ckpt=train_parallel0/resnet-42_5004.ckpt
```
- `top_5_accuracy`：对于一个输入图片，如果预测概率排名前五的标签中包含真实标签，即认为分类正确；
- `top_1_accuracy`：对于一个输入图片，如果预测概率最大的标签与真实标签相同，即认为分类正确。
#### GPU

在GPU硬件平台上，推理的执行命令如下：
```
sh run_eval_gpu.sh [DATASET_PATH] [CHECKPOINT_PATH]
```
脚本需要传入变量`DATASET_PATH`和`CHECKPOINT_PATH`，其中：
- `DATASET_PATH`：推理数据集路径。
- `CHECKPOINT_PATH`: 保存的checkpoint路径。

推理的结果如下：
```
result: {'top_5_accuracy': 0.9281169974391805, 'top_1_accuracy': 0.7593830025608195} ckpt=train_parallel0/resnet-42_5004.ckpt
```
