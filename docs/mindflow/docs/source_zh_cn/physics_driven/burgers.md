# 基于PINNs的Burgers' equation求解

<a href="https://gitee.com/mindspore/docs/blob/r2.0.0-alpha/docs/mindflow/docs/source_zh_cn/physics_driven/burgers.md" target="_blank"><img src="https://mindspore-website.obs.cn-north-4.myhuaweicloud.com/website-images/master/resource/_static/logo_source.png"></a>

## 概述

计算流体力学是21世纪流体力学领域的重要技术之一，其通过使用数值方法在计算机中对流体力学的控制方程进行求解，从而实现流动的分析、预测和控制。传统的有限元法（finite element method，FEM）和有限差分法（finite difference method，FDM）常囿于复杂的仿真流程（物理建模，网格划分，数值离散，迭代求解等）和较高的计算成本，往往效率低下。因此，借助AI提升流体仿真效率是十分必要的。

在经典理论与结合计算机性能的数值求解方法的发展趋于平缓的时候，近年来机器学习方法通过神经网络结合大量数据，实现流场的快速仿真，获得了接近传统方法的求解精度，为流场求解提供了新思路。

伯格斯方程（Burgers' equation）是一个模拟冲击波的传播和反射的非线性偏微分方程，被广泛应用于流体力学，非线性声学，气体动力学等领域，它以约翰内斯·马丁斯汉堡（1895-1981）的名字命名。本案例采用MindFlow流体仿真套件，基于物理驱动的PINNs (Physics Informed Neural Networks)方法，求解一维有粘性情况下的Burgers'方程。

## 问题描述

Burgers'方程的形式如下：

$$
u_t + uu_x = \epsilon u_{xx}, \quad x \in[-1,1], t \in[0, T],
$$

其中$\epsilon=0.01/\pi$，等号左边为对流项，右边为耗散项，本案例使用迪利克雷边界条件和正弦函数的初始条件，形式如下：

$$
u(t, -1) = u(t, 1) = 0,
$$

$$
u(0, x) = -sin(\pi x),
$$

本案例利用PINNs方法学习位置和时间到相应物理量的映射$(x, t) \mapsto u$，实现Burgers'方程的求解。

## 技术路径

MindFlow求解该问题的具体流程如下：

1. 创建训练数据集。
2. 构建模型。
3. 优化器。
4. 约束。
5. 模型训练。
6. 模型推理及可视化。

## 创建数据集

本案例根据求解域、初始条件及边值条件进行随机采样，生成训练数据集与测试数据集，具体设置如下：

```python
from mindflow.data import Dataset
from mindflow.geometry import Interval, TimeDomain, GeometryWithTime
from mindflow.geometry import generate_sampling_config
from mindflow.utils import load_yaml_config


def create_random_dataset(config):
    """create training dataset by online sampling"""
    geom_config = config["geometry"]
    data_config = config["data"]

    time_interval = TimeDomain("time", geom_config["time_min"], geom_config["time_max"])
    spatial_region = Interval("domain", geom_config["coord_min"], geom_config["coord_max"])
    region = GeometryWithTime(spatial_region, time_interval)
    region.set_sampling_config(generate_sampling_config(data_config))

    geom_dict = {region: ["domain", "IC", "BC"]}
    dataset = Dataset(geom_dict)

    return dataset

# load configurations
config = load_yaml_config('burgers_cfg.yaml')

# create dataset
burgers_train_dataset = create_random_dataset(config)
train_dataset = burgers_train_dataset.create_dataset(batch_size=config["train_batch_size"],
                                                     shuffle=True,
                                                     prebatched_data=True,
                                                     drop_remainder=True)
```

## 构建模型

本例使用简单的全连接网络，深度为6层，激发函数为`tanh`函数。

```python
import numpy as np

import mindspore as ms
from mindspore import set_seed
from mindspore import context, nn
from mindspore.train import DynamicLossScaleManager
from mindspore.train import ModelCheckpoint, CheckpointConfig
from mindspore import load_checkpoint, load_param_into_net

from mindflow.loss import Constraints
from mindflow.solver import Solver
from mindflow.common import LossAndTimeMonitor
from mindflow.cell import FCSequential
from mindflow.pde import Burgers1D
from mindflow.utils import load_yaml_config


model = FCSequential(in_channels=config["model"]["in_channels"],
                     out_channels=config["model"]["out_channels"],
                     layers=config["model"]["layers"],
                     neurons=config["model"]["neurons"],
                     residual=config["model"]["residual"],
                     act=config["model"]["activation"])
if config["load_ckpt"]:
    param_dict = load_checkpoint(config["load_ckpt_path"])
    load_param_into_net(model, param_dict)
if context.get_context(attr_key='device_target') == "Ascend":
    model.to_float(ms.float16)
```

## 优化器

```python
# define optimizer
optimizer = nn.Adam(model.trainable_params(), config["optimizer"]["initial_lr"])
```

## 约束

定义`constraint`将PDE问题同数据集关联起来，`Burgers1D`包含控制方程，边界条件和初始条件。

```python
burgers_problems = [Burgers1D(model=model) for _ in range(burgers_train_dataset.num_dataset)]
train_constraints = Constraints(burgers_train_dataset, burgers_problems)
```

## 模型训练

调用`Solver`接口用于模型的训练和推理。

```python
# define solvers
solver = Solver(model,
                optimizer=optimizer,
                train_constraints=train_constraints,
                loss_scale_manager=DynamicLossScaleManager(),
                )

# define callbacks
callbacks = [LossAndTimeMonitor(len(burgers_train_dataset))]
if config["save_ckpt"]:
    config_ck = CheckpointConfig(save_checkpoint_steps=10, keep_checkpoint_max=2)
    ckpoint_cb = ModelCheckpoint(prefix='burgers_1d', directory=config["save_ckpt_path"], config=config_ck)
    callbacks += [ckpoint_cb]

# run the solver to train the model with callbacks
train_epochs = 1000
solver.train(train_epochs, train_dataset, callbacks=callbacks, dataset_sink_mode=True)

```

模型结果如下：

```python
epoch: 991 step: 8, loss is 0.00016303812
epoch time: 0.257 s, per step time: 32.074 ms
epoch: 992 step: 8, loss is 0.00011361649
epoch time: 0.272 s, per step time: 33.975 ms
epoch: 993 step: 8, loss is 0.00014278122
epoch time: 0.259 s, per step time: 32.426 ms
epoch: 994 step: 8, loss is 0.002453908
epoch time: 0.284 s, per step time: 35.503 ms
epoch: 995 step: 8, loss is 0.0036222944
epoch time: 0.244 s, per step time: 30.465 ms
epoch: 996 step: 8, loss is 0.010429059
epoch time: 0.264 s, per step time: 32.949 ms
epoch: 997 step: 8, loss is 0.0036520353
epoch time: 0.250 s, per step time: 31.305 ms
epoch: 998 step: 8, loss is 0.0018088069
epoch time: 0.258 s, per step time: 32.295 ms
epoch: 999 step: 8, loss is 0.0017990978
epoch time: 0.257 s, per step time: 32.150 ms
epoch: 1000 step: 8, loss is 0.0032512385
epoch time: 0.250 s, per step time: 31.225 ms
End-to-End total time: 258.6200895309448 s
```

## 模型推理及可视化

训练后可对流场内所有数据点进行推理，并可视化相关结果。

```python
from src import visual_result

visual_result(model, resolution=config["visual_resolution"])
```

![PINNS结果](images/result.jpg)