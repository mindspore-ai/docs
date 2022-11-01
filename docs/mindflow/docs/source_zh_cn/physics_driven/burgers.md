# 基于PINNs的Burgers' equation求解

## 概述

计算流体力学是21世纪流体力学领域的重要技术之一，其通过使用数值方法在计算机中对流体力学的控制方程进行求解，从而实现流动的分析、预测和控制。传统的有限元法（finite element method，FEM）和有限差分法（finite difference method，FDM）常囿于复杂的仿真流程（物理建模，网格划分，数值离散，迭代求解等）和较高的计算成本，往往效率低下。因此，借助AI提升流体仿真效率是十分必要的。

在经典理论与结合计算机性能的数值求解方法的发展趋于平缓的时候，近年来机器学习方法通过神经网络结合大量数据，实现流场的快速仿真，获得了接近传统方法的求解精度，为流场求解提供了新思路。

伯格斯方程（Burgers' equation）是一个模拟冲击波的传播和反射的非线性偏微分方程，被广泛应用于流体力学，非线性声学，气体动力学等领域，它以约翰内斯·马丁斯汉堡（1895-1981）的名字命名。本案例采用MindFlow流体仿真套件，基于物理驱动的PINNs (Physics Informed Neural Networks)方法，求解一维有粘性情况下的Burgers'方程。

## 问题描述

Burgers'方程的形式如下：

$$
u_t + uu_x = \epsilon u_{xx}, \quad x \in[-1,1], t \in[0, T],
$$

其中 $\epsilon=0.01/\pi$ ，等号左边为对流项，右边为耗散项，本案例使用迪利克雷边界条件和正弦函数的初始条件，形式如下：

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
2. 构建神经网络。
3. 问题建模。
4. 模型训练。
5. 模型推理及可视化。

## 创建数据集

本案例根据求解域、初始条件及边值条件进行随机采样，生成训练数据集与测试数据集，具体设置如下：

```python
from mindflow.data import Dataset
from mindflow.geometry import FixedPoint, Interval, TimeDomain, GeometryWithTime
from mindflow.geometry import create_config_from_edict
from .sampling_config import src_sampling_config, bc_sampling_config


def create_random_dataset(config):
    """create training dataset by online sampling"""
    coord_min = config["coord_min"]
    coord_max = config["coord_max"]

    time_interval = TimeDomain("time", 0.0, config["range_t"])
    spatial_region = Interval("flow_region", coord_min, coord_max)
    region = GeometryWithTime(spatial_region, time_interval)
    region.set_sampling_config(create_config_from_edict(src_sampling_config))

    point1 = FixedPoint("point1", coord_min)
    boundary_1 = GeometryWithTime(point1, time_interval)
    boundary_1.set_name("bc1")
    boundary_1.set_sampling_config(create_config_from_edict(bc_sampling_config))

    point2 = FixedPoint("point2", coord_max)
    boundary_2 = GeometryWithTime(point2, time_interval)
    boundary_2.set_name("bc2")
    boundary_2.set_sampling_config(create_config_from_edict(bc_sampling_config))

    geom_dict = {region: ["domain", "IC"],
                 boundary_1: ["BC"],
                 boundary_2: ["BC"]}

    dataset = Dataset(geom_dict)

    return dataset
```

## 构建神经网络

本例使用简单的全连接网络，深度为6层，激发函数为`tanh`函数。

```python
from mindflow.cell import FCSequential

model = FCSequential(in_channel=2, out_channel=1, layers=6, neurons=20, residual=False, act="tanh")
```

## 问题建模

`Problem`包含求解问题的控制方程、边界条件、初始条件等。

```python
from math import pi as PI
from mindspore import ops
from mindspore import Tensor
from mindspore import dtype as mstype
from mindflow.solver import Problem
from mindflow.operators import Grad, SecondOrderGrad


class Burgers1D(Problem):
    """The 1D Burger's equations with constant boundary condition."""

    def __init__(self, model, config, domain_name=None, bc_name=None, bc_normal=None, ic_name=None):
        super(Burgers1D, self).__init__()
        self.domain_name = domain_name
        self.bc_name = bc_name
        self.ic_name = ic_name
        self.model = model
        self.grad = Grad(self.model)
        self.u_xx_cell = SecondOrderGrad(self.model, input_idx1=0, input_idx2=0, output_idx=0)
        self.reshape = ops.Reshape()
        self.split = ops.Split(1, 2)
        self.mu = Tensor(0.01 / PI, mstype.float32)
        self.pi = Tensor(PI, mstype.float32)

    def governing_equation(self, *output, **kwargs):
        """Burgers equation"""
        u = output[0]
        data = kwargs[self.domain_name]

        du_dxt = self.grad(data, None, 0, u)
        du_dx, du_dt = self.split(du_dxt)
        du_dxx = self.u_xx_cell(data)

        pde_r = du_dt + u * du_dx - self.mu * du_dxx

        return pde_r

    def boundary_condition(self, *output, **kwargs):
        """constant boundary condition"""
        u = output[0]
        return u

    def initial_condition(self, *output, **kwargs):
        """initial condition: u = - sin(x)"""
        u = output[0]
        data = kwargs[self.ic_name]
        x = self.reshape(data[:, 0], (-1, 1))
        return u + ops.sin(self.pi * x)
```

求解问题，定义`constraint`，作为模型训练的损失函数。

```python
train_prob = {}
for dataset in burgers_train_dataset.all_datasets:
    train_prob[dataset.name] = Burgers1D(model=model, config=config,
                                         domain_name="{}_points".format(dataset.name),
                                         ic_name="{}_points".format(dataset.name),
                                         bc_name="{}_points".format(dataset.name))
print("check problem: ", train_prob)
train_constraints = Constraints(burgers_train_dataset, train_prob)
```

## 模型训练

调用`Solver`接口用于模型的训练和推理。

```python
params = model.trainable_params()
optim = nn.Adam(params, 5e-3)

if config["load_ckpt"]:
    param_dict = load_checkpoint(config["load_ckpt_path"])
    load_param_into_net(model, param_dict)

solver = Solver(model,
                optimizer=optim,
                train_constraints=train_constraints,
                test_constraints=None,
                loss_scale_manager=DynamicLossScaleManager(),
                )

loss_time_callback = LossAndTimeMonitor(steps_per_epoch)
callbacks = [loss_time_callback]

if config["save_ckpt"]:
    config_ck = CheckpointConfig(save_checkpoint_steps=10, keep_checkpoint_max=2)
    ckpoint_cb = ModelCheckpoint(prefix='burgers_1d', directory=config["save_ckpt_path"], config=config_ck)
    callbacks += [ckpoint_cb]

solver.train(config["train_epoch"], train_dataset, callbacks=callbacks, dataset_sink_mode=True)
```

模型结果如下：

```python
epoch time: 1.695 s, per step time: 211.935 ms
epoch: 4991 step: 8, loss is 0.006608422
epoch time: 1.660 s, per step time: 207.480 ms
epoch: 4992 step: 8, loss is 0.006609884
epoch time: 1.691 s, per step time: 211.332 ms
epoch: 4993 step: 8, loss is 0.0065038507
epoch time: 1.675 s, per step time: 209.326 ms
epoch: 4994 step: 8, loss is 0.0066139675
epoch time: 1.684 s, per step time: 210.445 ms
epoch: 4995 step: 8, loss is 0.00651852
epoch time: 1.657 s, per step time: 207.111 ms
epoch: 4996 step: 8, loss is 0.006519169
epoch time: 1.686 s, per step time: 210.733 ms
epoch: 4997 step: 8, loss is 0.006666567
epoch time: 1.666 s, per step time: 208.297 ms
epoch: 4998 step: 8, loss is 0.006616782
epoch time: 1.698 s, per step time: 212.293 ms
epoch: 4999 step: 8, loss is 0.0066004843
epoch time: 1.666 s, per step time: 208.225 ms
epoch: 5000 step: 8, loss is 0.006627152
epoch time: 1.690 s, per step time: 211.255 ms
==================================================================================================
predict total time: 0.03775811195373535 s
==================================================================================================
End-to-End total time: 3358.674560308456 s
```

## 模型推理及可视化

训练后可对流场内所有数据点进行推理，并可视化相关结果。

```python
from src import visual_result

visual_result(model, resolution=config["visual_resolution"])
```

![PINNS结果](images/result.jpg)
