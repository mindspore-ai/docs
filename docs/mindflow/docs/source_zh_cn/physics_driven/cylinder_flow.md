
# 基于PINNs关于圆柱绕流的Navier-Stokes equation求解

<a href="https://gitee.com/mindspore/docs/blob/master/docs/mindflow/docs/source_zh_cn/physics_driven/cylinder_flow.md" target="_blank"><img src="https://mindspore-website.obs.cn-north-4.myhuaweicloud.com/website-images/master/resource/_static/logo_source.png"></a>

## 概述

圆柱绕流，是指二维圆柱低速定常绕流的流型只与`Re`数有关。在`Re`≤1时，流场中的惯性力与粘性力相比居次要地位，圆柱上下游的流线前后对称，阻力系数近似与`Re`成反比，此`Re`数范围的绕流称为斯托克斯区；随着Re的增大，圆柱上下游的流线逐渐失去对称性。这种特殊的现象反映了流体与物体表面相互作用的奇特本质，求解圆柱绕流则是流体力学中的经典问题。

由于控制方程纳维-斯托克斯方程（Navier-Stokes equation）难以得到泛化的理论解，使用数值方法对圆柱绕流场景下控制方程进行求解，从而预测流场的流动，成为计算流体力学中的样板问题。传统求解方法通常需要对流体进行精细离散化，以捕获需要建模的现象。因此，传统有限元法（finite element method，FEM）和有限差分法（finite difference method，FDM）往往成本比较大。

物理启发的神经网络方法（Physics-informed Neural Networks），以下简称`PINNs`，通过使用逼近控制方程的损失函数以及简单的网络构型，为快速求解复杂流体问题提供了新的方法。本案例利用神经网络数据驱动特性，结合`PINNs`求解圆柱绕流问题。

## 纳维-斯托克斯方程（Navier-Stokes equation）

纳维-斯托克斯方程（Navier-Stokes equation），简称`N-S`方程，是流体力学领域的经典偏微分方程，在粘性不可压缩情况下，无量纲`N-S`方程的形式如下：

$$
\frac{\partial u}{\partial x} + \frac{\partial v}{\partial y} = 0
$$

$$
\frac{\partial u} {\partial t} + u \frac{\partial u}{\partial x} + v \frac{\partial u}{\partial y} = - \frac{\partial p}{\partial x} + \frac{1} {Re} (\frac{\partial^2u}{\partial x^2} + \frac{\partial^2u}{\partial y^2})
$$

$$
\frac{\partial v} {\partial t} + u \frac{\partial v}{\partial x} + v \frac{\partial v}{\partial y} = - \frac{\partial p}{\partial y} + \frac{1} {Re} (\frac{\partial^2v}{\partial x^2} + \frac{\partial^2v}{\partial y^2})
$$

其中，`Re`表示雷诺数。

## 问题描述

本案例利用PINNs方法学习位置和时间到相应流场物理量的映射，实现`N-S`方程的求解：

$$
(x, y, t) \mapsto (u, v, p)
$$

## 技术路径

MindFlow求解该问题的具体流程如下：

1. 创建数据集。
2. 构建模型。
3. 优化器。
4. 约束。
5. 模型训练。
6. 模型推理及可视化

## 训练示例

### 创建数据集

本案例对已有的雷诺数为100的标准圆柱绕流进行初始条件和边界条件数据的采样。对于训练数据集，构建平面矩形的问题域以及时间维度，再对已知的初始条件，边界条件采样；基于已有的流场中的点构造验证集。

```python
def create_training_dataset(config):
    """create training dataset by online sampling"""
    coord_min = config["coord_min"]
    coord_max = config["coord_max"]
    rectangle = Rectangle("rect", coord_min, coord_max)

    time_interval = TimeDomain("time", 0.0, config["range_t"])
    domain_region = GeometryWithTime(rectangle, time_interval)
    domain_region.set_name("domain")
    domain_region.set_sampling_config(create_config_from_edict(domain_sampling_config))

    geom_dict = {domain_region: ["domain"]}

    data_path = config["train_data_path"]
    config_bc = ExistedDataConfig(name="bc",
                                  data_dir=[data_path + "/bc_points.npy", data_path + "/bc_label.npy"],
                                  columns_list=["points", "label"],
                                  constraint_type="BC",
                                  data_format="npy")
    config_ic = ExistedDataConfig(name="ic",
                                  data_dir=[data_path + "/ic_points.npy", data_path + "/ic_label.npy"],
                                  columns_list=["points", "label"],
                                  constraint_type="IC",
                                  data_format="npy")
    dataset = Dataset(geom_dict, existed_data_list=[config_bc, config_ic])
    return dataset

```

### 自适应损失的多任务学习

同一时间，基于PINNs的方法需要优化多个loss，给优化过程带来的巨大的挑战。我们采用***Kendall, Alex, Yarin Gal, and Roberto Cipolla. "Multi-task learning using uncertainty to weigh losses for scene geometry and semantics." CVPR, 2018.*** 论文中提出的不确定性权重算法动态调整权重。

```python
    mtl = MTLWeightedLossCell(num_losses=cylinder_dataset.num_dataset)
```

### 模型构建

本示例使用一个简单的全连接网络，深度为6层，激活函数是`tanh`函数。

```python
import numpy as np

import mindspore as ms
from mindspore.common import set_seed
from mindspore import context, Tensor, nn
from mindspore.train import DynamicLossScaleManager
from mindspore.train import ModelCheckpoint, CheckpointConfig
from mindspore.train import load_checkpoint, load_param_into_net

from mindflow.loss import Constraints
from mindflow.solver import Solver
from mindflow.common import L2, LossAndTimeMonitor
from mindflow.loss import MTLWeightedLossCell
from mindflow.pde import NavierStokes2D

from src import FlowNetwork

model = FlowNetwork(config["model"]["in_channels"],
                    config["model"]["out_channels"],
                    coord_min=config["geometry"]["coord_min"] + [config["geometry"]["time_min"]],
                    coord_max=config["geometry"]["coord_max"] + [config["geometry"]["time_max"]],
                    num_layers=config["model"]["layers"],
                    neurons=config["model"]["neurons"],
                    residual=config["model"]["residual"])

if config["load_ckpt"]:
    param_dict = load_checkpoint(config["load_ckpt_path"])
    load_param_into_net(model, param_dict)
    load_param_into_net(mtl, param_dict)

if context.get_context(attr_key='device_target') == "Ascend":
    model.to_float(ms.float16)
```

### 模型训练

调用`Solver`接口进行模型训练，调用`callback`接口进行评估。

```python
# define solver
solver = Solver(model,
                optimizer=optim,
                train_constraints=train_constraints,
                test_constraints=None,
                metrics={'l2': L2(), 'distance': nn.MAE()},
                loss_fn='smooth_l1_loss',
                loss_scale_manager=DynamicLossScaleManager(init_loss_scale=2 ** 10, scale_window=2000),
                mtl_weighted_cell=mtl,
                )

loss_time_callback = LossAndTimeMonitor(steps_per_epoch)
callbacks = [loss_time_callback]
if config.get("train_with_eval", False):
    inputs, label = create_evaluation_dataset(config["test_data_path"])
    predict_callback = PredictCallback(model, inputs, label, config=config, visual_fn=visualization)
    callbacks += [predict_callback]
if config["save_ckpt"]:
    config_ck = CheckpointConfig(save_checkpoint_steps=10,
                                 keep_checkpoint_max=2)
    ckpoint_cb = ModelCheckpoint(prefix='ckpt_flow_past_cylinder_Re100',
                                 directory=config["save_ckpt_path"], config=config_ck)
    callbacks += [ckpoint_cb]

solver.train(config["train_epoch"], train_dataset, callbacks=callbacks, dataset_sink_mode=True)
```

## 网络训练结果

运行结果如下：

```python
epoch: 4991 step: 8, loss is 0.0063523385
epoch time: 0.863 s, per step time: 107.902 ms
epoch: 4992 step: 8, loss is 0.006585151
epoch time: 0.864 s, per step time: 107.974 ms
epoch: 4993 step: 8, loss is 0.006354205
epoch time: 0.862 s, per step time: 107.711 ms
epoch: 4994 step: 8, loss is 0.006413138
epoch time: 0.865 s, per step time: 108.074 ms
epoch: 4995 step: 8, loss is 0.0062734303
epoch time: 0.860 s, per step time: 107.502 ms
epoch: 4996 step: 8, loss is 0.006455861
epoch time: 0.862 s, per step time: 107.750 ms
epoch: 4997 step: 8, loss is 0.006378171
epoch time: 0.864 s, per step time: 107.976 ms
epoch: 4998 step: 8, loss is 0.00636143
epoch time: 0.862 s, per step time: 107.709 ms
epoch: 4999 step: 8, loss is 0.006477215
epoch time: 0.864 s, per step time: 108.024 ms
epoch: 5000 step: 8, loss is 0.0064105876
epoch time: 0.862 s, per step time: 107.746 ms
==================================================================================================
predict total time: 0.024950027465820312 s
l2_error, U:  0.011893196515698487 , V:  0.052116949016282374 , P:  0.2798291882189069 , Total:  0.04287303192192062
==================================================================================================
End-to-End total time: 5388.308397293091 s
```

### 分析

训练过程中的error如图所示，随着epoch增长，error逐渐下降。

5000 epochs 对应的loss：

![epoch5000](images/TimeError_epoch5000.png)

计算过程中callback记录了每个时刻U，V，P的预测情况，与真实值偏差比较小。

![image_flow](images/image-flow.png)
