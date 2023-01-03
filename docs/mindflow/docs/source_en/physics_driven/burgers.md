# Solve Burgers' Equation based on PINNs

<a href="https://gitee.com/mindspore/docs/blob/r2.0.0-alpha/docs/mindflow/docs/source_en/physics_driven/burgers.md" target="_blank"><img src="https://mindspore-website.obs.cn-north-4.myhuaweicloud.com/website-images/master/resource/_static/logo_source_en.png"></a>

## Overview

Computational fluid dynamics is one of the most important techniques in the field of fluid mechanics in the 21st century. The flow analysis, prediction and control can be realized by solving the governing equations of fluid mechanics by numerical method. Traditional finite element method (FEM) and finite difference method (FDM) are inefficient because of the complex simulation process (physical modeling, meshing, numerical discretization, iterative solution, etc.) and high computing costs. Therefore, it is necessary to improve the efficiency of fluid simulation with AI.

In recent years, while the development of classical theories and numerical methods with computer performance tends to be smooth, machine learning methods combine a large amount of data with neural networks realize the flow field's fast simulation. These methods can obtain the accuracy close to the traditional methods, which provides a new idea for flow field solution.

Burgers' equation is a nonlinear partial differential equation that simulates the propagation and reflection of shock waves. It is widely used in the fields of fluid mechanics, nonlinear acoustics, gas dynamics et al. It is named after Johannes Martins Hamburg (1895-1981). In this case, MindFlow fluid simulation suite is used to solve the Burgers' equation in one-dimensional viscous state based on the physical-driven PINNs (Physics Informed Neural Networks) method.

## Problem Description

The form of Burgers' equation is as follows:

$$
u_t + uu_x = \epsilon u_{xx}, \quad x \in[-1,1], t \in[0, T],
$$

where $\epsilon=0.01/\pi$, the left of the equal sign is the convection term, and the right is the dissipation term. In this case, the Dirichlet boundary condition and the initial condition of the sine function are used. The format is as follows:

$$
u(t, -1) = u(t, 1) = 0,
$$

$$
u(0, x) = -sin(\pi x),
$$

In this case, the PINNs method is used to learn the mapping $(x, t) \mapsto u$ from position and time to corresponding physical quantities. So that the solution of Burgers' equation is realized.

## Technology Path

MindFlow solves the problem as follows:

1. Training Dataset Construction.
2. Model Construction.
3. Optimizer.
4. Constrains.
5. Model Training.
6. Model Evaluation and Visualization.

## Training Dataset Construction

In this case, random sampling is performed according to the solution domain, initial condition and boundary value condition to generate training data sets and test data sets. The specific settings are as follows:

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

## Model Construction

This example uses a simple fully-connected network with a depth of 6 layers and the activation function is the `tanh` function.

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

## Optimizer

```python
# define optimizer
optimizer = nn.Adam(model.trainable_params(), config["optimizer"]["initial_lr"])
```

## Constraints

The `Constraints` class relates the PDE problems with the training datasets. `Burgers1D` contains the governing equations, boundary conditions, initial conditions et al. to solve the problem.

```python
burgers_problems = [Burgers1D(model=model) for _ in range(burgers_train_dataset.num_dataset)]
train_constraints = Constraints(burgers_train_dataset, burgers_problems)
```

## Model Training

Invoke the `Solver` interface for model training and inference.

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

The model results are as follows:

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

## Model Evaluation and Visualization

After training, all data points in the flow field can be inferred. And related results can be visualized.

```python
from src import visual_result

visual_result(model, resolution=config["visual_resolution"])
```

![PINNs_results](images/result.jpg)