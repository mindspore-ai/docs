# Solve Burgers' equation based on Fourier Neural Operator

<a href="https://gitee.com/mindspore/docs/blob/master/docs/mindflow/docs/source_en/data_driven/fno1d.md" target="_blank"><img src="https://mindspore-website.obs.cn-north-4.myhuaweicloud.com/website-images/master/resource/_static/logo_source_en.png"></a>

## Overview

Computational fluid dynamics is one of the most important techniques in the field of fluid mechanics in the 21st century. The flow analysis, prediction and control can be realized by solving the governing equations of fluid mechanics by numerical method. Traditional finite element method (FEM) and finite difference method (FDM) are inefficient because of the complex simulation process (physical modeling, meshing, numerical discretization, iterative solution, etc.) and high computing costs. Therefore, it is necessary to improve the efficiency of fluid simulation with AI.

Machine learning methods provide a new paradigm for scientific computing by providing a fast solver similar to traditional methods. Classical neural networks learn mappings between finite dimensional spaces and can only learn solutions related to a specific discretization. Different from traditional neural networks, Fourier Neural Operator (FNO) is a new deep learning architecture that can learn mappings between infinite-dimensional function spaces. It directly learns mappings from arbitrary function parameters to solutions to solve a class of partial differential equations.  Therefore, it has a stronger generalization capability. More information can be found in [paper] (https://arxiv.org/abs/2010.08895).

This tutorial describes how to solve the 1-d Burgers' equation using Fourier neural operator.

## Burgers' equation

The 1-d Burgers’ equation is a non-linear PDE with various applications including modeling the one
dimensional flow of a viscous fluid. It takes the form

$$
\partial_t u(x, t)+\partial_x (u^2(x, t)/2)=\nu \partial_{xx} u(x, t), \quad x \in(0,1), t \in(0, 1]
$$

$$
u(x, 0)=u_0(x), \quad x \in(0,1)
$$

where $u$ is the velocity field, $u_0$ is the initial condition and $\nu$ is the viscosity coefficient.

## Description

We aim to learn the operator mapping the initial condition to the solution at time one:

$$
u_0 \mapsto u(\cdot, 1)
$$

The process of using MindFlow to solve the problem is as follows:

1. Configure network and training parameters.
2. Create datasets.
3. Build a neural network.
4. Define the loss function.
5. Define the model testing module.
6. Model Training.

## Fourier Neural Operator

The Fourier Neural Operator consists of the Lifting Layer, Fourier Layers, and the Decoder Layer.

![Fourier Neural Operator model structure](images/FNO.png)

Fourier layers: Start from input V. On top: apply the Fourier transform $\mathcal{F}$; a linear transform R on the lower Fourier modes and filters out the higher modes; then apply the inverse Fourier transform $\mathcal{F}^{-1}$. On the bottom: apply a local linear transform W.  Finally, the Fourier Layer output vector is obtained through the activation function.

![Fourier Layer structure](images/FNO-2.png)

## Example

### Configuration file

The configuration file contains four types of parameters: model parameters, data parameters, optimizer parameters, and output parameters. The most important parameters of the FNO model are modes, width, and depth, which control the number of reserved frequencies, the number of lifting channels, and the number of fourier layers in the model, respectively. The default values of the parameters are as follows:

```python
model:
  name: FNO1D                       # Model name
  input_dims: 1                     # Input data channel
  output_dims: 1                    # Output data channel
  input_resolution: 1024            # Column resolution
  modes: 12                         # Number of frequency to keep
  width: 32                         # The number of channels after dimension lifting of the input
  depth: 8                          # The number of fourier layer
data:
  name: "burgers1d"                 # Data name
  path: "/path/to/data"             # Path to data
  train_size: 1000                  # Size of train set
  test_size: 200                    # Size of test set
  batch_size: 8                     # Batch size
  sub: 8                            # Sampling interval when creating a dataset
optimizer:
  initial_lr: 0.001                 # Initial learning rate
  weight_decay: 0.0                 # Weight decay of the optimizer
  gamma: 0.5                        # The gamma value of the optimizer
  train_epochs: 100                 # The number of train epoch
  valid_epochs: 10                  # The number of validation epoch
callback:
  summary_dir: "/path/to/summary"   # Path to Summary
  save_checkpoint_steps: 50         # Frequency of checkpoint results
  keep_checkpoint_max: 10           # Maximum number of checkpoint files can be saved
```

### Import dependencies

Import the modules and interfaces on which this tutorial depends:

```python
import mindspore.nn as nn
from mindspore import set_seed
from mindspore import Tensor
from mindspore import context
from mindspore.train import LossMonitor, TimeMonitor
from mindspore.train import DynamicLossScaleManager

from mindflow.solver import Solver
from mindflow.cell.neural_operators import FNO1D
from src.lr_scheduler import warmup_cosine_annealing_lr
from src.dataset import create_dataset
from src.utils import load_config
from src.loss import RelativeRMSELoss
```

### Create datasets

In this case, training datasets and test data sets are generated according to Zongyi Li's data set in Fourier Neural Operator for Parametric Partial Differential Equations(https://arxiv.org/pdf/2010.08895.pdf) . The settings are as follows:

the initial condition $w_0(x)$ is generated according to periodic boundary conditions:

$$
w_0 \sim \mu, \mu=\mathcal{N}\left(0,7^{3 / 2}(-\Delta+49 I)^{-2.5}\right)
$$

The forcing function is defined as:

$$
f(x)=0.1\left(\sin \left(2 \pi\left(x_1+x_2\right)\right)+\right.\cos(2 \pi(x_1+x_2)))
$$

We use a time-step of 1e-4 for the Crank–Nicolson scheme in the data-generated process where we record the solution every t = 1 time units.  All data are generated on a 256 × 256 grid and are downsampled to 64 × 64.  In this case, the viscosity coefficient $\nu=1e-5$, the number of samples in the training set is 1000, and the number of samples in the test set is 200.

```python
# create dataset for train
config = load_config('path/to/config')
data_params = config["data"]
model_params = config["model"]
train_dataset = create_dataset(data_params,
                               input_resolution=model_params["input_resolution"],
                               shuffle=True)
```

`load_config` is referenced from `utils.py`, and `create_dataset` is referenced from `dataset.py`. Parameters in data_params and model_params are configured in the configuration file.

### Build a neural network

The network consists of a lifting layer, multiple Fourier layers, and a decoder layer:

- The lifting layer corresponds to `FNO1D.fc0` in the sample code. It maps the output data $x$ to high-level dimensions.
- Multi-layer Fourier Layers corresponds to `FNO1D.fno_seq` in the sample code. In this case, discrete Fourier transform is used to implement time-domain and frequency-domain conversion.
- The decoder layer corresponds to `FNO1D.fc1` and `FNO1D.fc2` in the code to obtain the final predicted value.

The model is initialized based on the foregoing network structure. The configuration in model_params can be modified in the configuration file.

```python
model = FNO1D(input_dims=model_params["input_dims"],
              output_dims=model_params["output_dims"],
              resolution=model_params["resolution"],
              modes=model_params["modes"],
              channels=model_params["channels"],
              depth=model_params["depth"],
              mlp_ratio=model_params["mlp_ratio"])
```

### Define the loss function

This case uses the rms error as the loss function:

```python
import mindspore
import mindspore.nn as nn
from mindspore import ops

class RelativeRMSELoss(nn.LossBase):
    def __init__(self, reduction="sum"):
        super(RelativeRMSELoss, self).__init__(reduction=reduction)

    def construct(self, prediction, label):
        prediction = ops.Cast()(prediction, mindspore.float32)
        batch_size = prediction.shape[0]
        diff_norms = ops.square(prediction.reshape(batch_size, -1) - label.reshape(batch_size, -1)).sum(axis=1)
        label_norms = ops.square(label.reshape(batch_size, -1)).sum(axis=1)
        rel_error = ops.div(ops.sqrt(diff_norms), ops.sqrt(label_norms))
        return self.get_loss(rel_error)
```

### Define the model testing module

The customised PredictCallback function is used to implement inference during training. You can directly load the test data set and output the inference precision of the test set every n epochs are trained. The value of n is specified by eval_interval in the configuration file.

```python
import time

import numpy as np
from mindspore.train.callback import Callback
from mindspore.train.summary import SummaryRecord
from mindspore import Tensor
from mindspore import dtype as mstype

class PredictCallback(Callback):
    def __init__(self, model, inputs, label, config, summary_dir):
        super(PredictCallback, self).__init__()
        self.model = model
        self.inputs = inputs
        self.label = label
        self.length = label.shape[0]
        self.summary_dir = summary_dir
        self.predict_interval = config.get("eval_interval", 10)
        self.batch_size = config.get("test_batch_size", 1)

        self.rms_error = 1.0
        print("check test dataset shape: {}, {}".format(self.inputs.shape, self.label.shape))

    def __enter__(self):
        self.summary_record = SummaryRecord(self.summary_dir)
        return self

    def __exit__(self, *exc_args):
        self.summary_record.close()

    def epoch_end(self, run_context):
        """
        Evaluate the model at the end of epoch.

        Args:
            run_context (RunContext): Context of the train running.
        """
        cb_params = run_context.original_args()
        if cb_params.cur_epoch_num % self.predict_interval == 0:
            print("================================Start Evaluation================================")
            time_beg = time.time()
            rms_error = 0.0
            max_error = 0.0
            for i in range(self.length):
                label = self.label[i:i + 1]
                test_batch = Tensor(self.inputs[i:i + 1], dtype=mstype.float32)
                prediction = self.model(test_batch)
                prediction = prediction.asnumpy()
                rms_error_step = self._calculate_error(label, prediction)
                rms_error += rms_error_step

                if rms_error_step >= max_error:
                    max_error = rms_error_step

            self.rms_error = rms_error / self.length
            print("mean rms_error:", self.rms_error)
            self.summary_record.add_value('scalar', 'rms_error', Tensor(self.rms_error))
            print("=================================End Evaluation=================================")
            print("predict total time: {} s".format(time.time() - time_beg))
            self.summary_record.record(cb_params.cur_step_num)

    def _calculate_error(self, label, prediction):
        """calculate l2-error to evaluate accuracy"""
        rel_error = np.sqrt(np.sum(np.square(label.reshape(self.batch_size, -1) -
                                             prediction.reshape(self.batch_size, -1)))) / \
                    np.sqrt(np.sum(np.square(prediction.reshape(self.batch_size, -1))))
        return rel_error

    def get_rms_error(self):
        return self.rms_error
```

Initialize the PredictCallback：

```python
pred_cb = PredictCallback(model=model,
                              inputs=test_input,
                              label=test_label,
                              config=config,
                              summary_dir=summary_dir)
```

### Model Training and Evaluation

The Solver is an interface for model training and inference. The solver is defined by the optimizer, network model, loss function, loss scaling policy, and so on. In this case, MindSpore + Ascend mixed precision model is used to train the network to solve the 1D Burgers' equation. In the code, parameters of optimizer_params and model_params are modified in the configuration file.

```python
# optimizer
steps_per_epoch = train_dataset.get_dataset_size()
lr = warmup_cosine_annealing_lr(lr=optimizer_params["initial_lr"],
                                steps_per_epoch=steps_per_epoch,
                                warmup_epochs=1,
                                max_epoch=optimizer_params["train_epochs"])
optimizer = nn.Adam(model.trainable_params(), learning_rate=Tensor(lr))

# prepare loss function
loss_scale = DynamicLossScaleManager()
loss_fn = RelativeRMSELoss()

# define solver
solver = Solver(model,
                optimizer=optimizer,
                loss_scale_manager=loss_scale,
                loss_fn=loss_fn,
                )
solver.train(epoch=optimizer_params["train_epochs"],
             train_dataset=train_dataset,
             callbacks=[LossMonitor(), TimeMonitor(), pred_cb],
             dataset_sink_mode=True)
```

## Training Result

The model training result is as follows:

```python
......
epoch: 91 step: 125, loss is 0.023883705958724022
Train epoch time: 1661.097 ms, per step time: 13.289 ms
epoch: 92 step: 125, loss is 0.03884338587522507
Train epoch time: 1697.662 ms, per step time: 13.581 ms
epoch: 93 step: 125, loss is 0.022736992686986923
Train epoch time: 1582.604 ms, per step time: 12.661 ms
epoch: 94 step: 125, loss is 0.020916245877742767
Train epoch time: 1452.545 ms, per step time: 11.620 ms
epoch: 95 step: 125, loss is 0.02459101565182209
Train epoch time: 1440.897 ms, per step time: 11.527 ms
epoch: 96 step: 125, loss is 0.021852120757102966
Train epoch time: 1443.364 ms, per step time: 11.547 ms
epoch: 97 step: 125, loss is 0.02737339772284031
Train epoch time: 1413.037 ms, per step time: 11.304 ms
epoch: 98 step: 125, loss is 0.024358950555324554
Train epoch time: 1427.533 ms, per step time: 11.420 ms
epoch: 99 step: 125, loss is 0.026329636573791504
Train epoch time: 1427.751 ms, per step time: 11.422 ms
epoch: 100 step: 125, loss is 0.029131107032299042
Train epoch time: 1444.621 ms, per step time: 11.557 ms
================================Start Evaluation================================
mean rms_error: 0.003944212107453496
=================================End Evaluation=================================
......
```
