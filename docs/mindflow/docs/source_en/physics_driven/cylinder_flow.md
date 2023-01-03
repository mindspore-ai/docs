# PINNS-based solution for flow past a cylinder

<a href="https://gitee.com/mindspore/docs/blob/r2.0.0-alpha/docs/mindflow/docs/source_en/physics_driven/cylinder_flow.md" target="_blank"><img src="https://mindspore-website.obs.cn-north-4.myhuaweicloud.com/website-images/master/resource/_static/logo_source_en.png"></a>

## Overview

Flow past cylinder problem is a two-dimensional low velocity steady flow around a cylinder which is only related to the `Re` number. When `Re` is less than or equal to 1, the inertial force in the flow field is secondary to the viscous force, the streamlines in the upstream and downstream directions of the cylinder are symmetrical, and the drag coefficient is approximately inversely proportional to `Re` . The flow around this `Re` number range is called the Stokes zone; With the increase of `Re` , the streamlines in the upstream and downstream of the cylinder gradually lose symmetry. This special phenomenon reflects the peculiar nature of the interaction between the fluid and the surface of the body. Solving flow past a cylinder is a classical problem in hydromechanics.

Since it is difficult to obtain the generalized theoretical solution of the Navier-Stokes equation,the numerical method is used to solve the governing equation in the flow past cylinder scenario to predict the flow field, which is also a classical problem in computational fluid mechanics. Traditional solutions often require fine discretization of the fluid to capture the phenomena that need to be modeled. Therefore, traditional finite element method (FEM) and finite difference method (FDM) are often costly.

Physics-informed Neural Networks (PINNs) provides a new method for quickly solving complex fluid problems by using loss functions that approximate governing equations coupled with simple network configurations. In this case, the data-driven characteristic of neural network is used along with `PINNs` to solve the flow past cylinder problem.

## Navier-Stokes equation

The Navier-Stokes equation, referred to as `N-S` equation, is a classical partial differential equation in the field of fluid mechanics. In the case of viscous incompressibility, the dimensionless `N-S` equation has the following form:

$$
\frac{\partial u}{\partial x} + \frac{\partial v}{\partial y} = 0
$$

$$
\frac{\partial u} {\partial t} + u \frac{\partial u}{\partial x} + v \frac{\partial u}{\partial y} = - \frac{\partial p}{\partial x} + \frac{1} {Re} (\frac{\partial^2u}{\partial x^2} + \frac{\partial^2u}{\partial y^2})
$$

$$
\frac{\partial v} {\partial t} + u \frac{\partial v}{\partial x} + v \frac{\partial v}{\partial y} = - \frac{\partial p}{\partial y} + \frac{1} {Re} (\frac{\partial^2v}{\partial x^2} + \frac{\partial^2v}{\partial y^2})
$$

where `Re` stands for Reynolds number.

## Problem Description

In this case, the PINNs method is used to learn the mapping from the location and time to flow field quantities to solve the `N-S` equation.

$$
(x, y, t) \mapsto (u, v, p)
$$

## Technology Path

MindFlow solves the problem as follows:

1. Training Dataset Construction.
2. Multi-task Learning for Adaptive Losses.
3. Model Construction.
4. Optimizer.
5. Constraints.
6. Model Traning.

## Training Example

### Training Dataset Construction

In this case, the initial condition and boundary condition data of the existing flow around a cylinder with Reynolds number 100 are sampled. For the training dataset, the problem domain and time dimension of planar rectangle are constructed. Then the known initial conditions and boundary conditions are sampled. The validation set is constructed based on the existing points in the flow field.

```python
from mindflow.utils import load_yaml_config

from src import create_training_dataset, create_evaluation_dataset

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

config = load_yaml_config('cylinder_flow.yaml')

cylinder_dataset = create_training_dataset(config)
train_dataset = cylinder_dataset.create_dataset(batch_size=config["train_batch_size"],
                                                shuffle=True,
                                                prebatched_data=True,
                                                drop_remainder=True)
```

## Multi-task Learning for Adaptive Losses

The PINNs method needs to optimize multiple losses at the same time, and brings challenges to the optimization process. Here, we adopt the uncertainty weighting algorithm proposed in ***Kendall, Alex, Yarin Gal, and Roberto Cipolla. "Multi-task learning using uncertainty to weigh losses for scene geometry and semantics." CVPR, 2018.*** to dynamically adjust the weights.

```python
mtl = MTLWeightedLossCell(num_losses=cylinder_dataset.num_dataset)
```

## Model Construction

This example uses a simple fully-connected network with a depth of 6 layers and the activation function is the `tanh` function.

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

### Optimizer

```python
from src import MultiStepLR

params = model.trainable_params() + mtl.trainable_params()
lr_scheduler = MultiStepLR(config["optimizer"]["initial_lr"],
                           config["optimizer"]["milestones"],
                           config["optimizer"]["gamma"],
                           len(cylinder_dataset),
                           config["train_epochs"])
lr = lr_scheduler.get_lr()
optimizer = nn.Adam(params, learning_rate=Tensor(lr))
```

### Constraints

The `Constraints` class relates the PDE problems with the training datasets. `NavierStokes2D` contains the governing equations, boundary conditions and initial conditions for solving the problem. The governing equation directly uses the incompressible `N-S` equation. The initial conditions and boundary conditions are obtained from the known data. Users can set different boundary conditions according to different data sets.

```python
problem_list = [NavierStokes2D(model=model, re=config["Re"]) for i in range(cylinder_dataset.num_dataset)]
train_constraints = Constraints(cylinder_dataset, problem_list)
```

### Model Training

Invoke the `Solver` interface for model training and `callback` interface for evaluation.

```python
from src import create_evaluation_dataset, PredictCallback, visualization

# define solver
solver = Solver(model,
                optimizer=optimizer,
                train_constraints=train_constraints,
                metrics={'l2': L2(), 'distance': nn.MAE()},
                loss_fn='smooth_l1_loss',
                loss_scale_manager=DynamicLossScaleManager(init_loss_scale=2 ** 10, scale_window=2000),
                mtl_weighted_cell=mtl,
                )

loss_time_callback = LossAndTimeMonitor(len(cylinder_dataset))
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

solver.train(config["train_epochs"], train_dataset, callbacks=callbacks, dataset_sink_mode=True)
```

## Network training result

The command output is as follows:

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

### Analysis

The following figure shows the errors versus time training epoch 5000. As the epoch number increases, the errors decreases accordingly.
Loss corresponding to 5000 epochs:

![epoch5000](images/TimeError_epoch5000.png)

During the calculation, the callback records the predicted values of U, V, and P at each step. The difference between the predicted values and the actual values is small.

![image_flow](images/image-flow.png)
