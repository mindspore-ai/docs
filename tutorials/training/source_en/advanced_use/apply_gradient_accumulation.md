# Applying Gradient Accumulation Algorithm

`Linux` `GPU` `Model Optimization` `Intermediate` `Expert`

[![View Source On Gitee](../_static/logo_source.png)](https://gitee.com/mindspore/docs/blob/r1.0/tutorials/training/source_en/advanced_use/apply_gradient_accumulation.md)

## Overview

This tutorial describes the gradient accumulation training method to solve the problem that some large-scale networks cannot train large batch_size due to insufficient memory.
  
In a traditional training method, after a loss and a gradient are calculated, a parameter is directly updated by using the obtained gradient.
  
Different from the traditional training method, the concept of mini-batch is introduced to the gradient accumulation. The loss and gradient are computed for each mini-batch data, but the model parameters are not updated immediately. Instead, the obtained gradients are accumulated first, and then after the number (N) of mini-batches is specified, the accumulated gradient is used to update the network parameters. Before the next training, the accumulated gradients are cleared and re-accumulated.
  
The ultimate objective is to achieve the same effect as training with N x mini-batch data.

> This tutorial is applicable to GPUs. You can download the main training sample code from <https://gitee.com/mindspore/docs/tree/r1.0/tutorials/tutorial_code/gradient_accumulation>.

## Creating a Gradient Accumulation Model

The MNIST dataset is used as an example to describe how to customize a simple model to implement gradient accumulation.

### Importing Library Files

The following are the required public modules and MindSpore modules and library files.

```python
import argparse
import os
from collections.abc import Iterable

import mindspore.nn as nn
from mindspore import ParameterTuple
from mindspore import context
from mindspore.nn import Cell
import mindspore.ops as ops
from mindspore.train.dataset_helper import DatasetHelper
from mindspore.train.serialization import save_checkpoint
from model_zoo.official.cv.lenet.src.dataset import create_dataset
from model_zoo.official.cv.lenet.src.lenet import LeNet5
```

### Loading the Dataset

Use the `MnistDataset` API provided by the dataset of MindSpore to load the MNIST dataset. The code is imported from [dataset.py](<https://gitee.com/mindspore/mindspore/blob/r1.0/model_zoo/official/cv/lenet/src/dataset.py>) in the lenet directory of model_zoo.

### Defining the Network

The following uses the LeNet network as an example. You can also use other networks, such as ResNet-50 and BERT. The code is imported from [lenet.py](<https://gitee.com/mindspore/mindspore/blob/r1.0/model_zoo/official/cv/lenet/src/lenet.py>) in the lenet directory of model_zoo.

### Defining the Training Model

The training process is divided into three parts: forward and backward training, parameter update, and accumulated gradient clearance.

- `TrainForwardBackward` calculates the loss and gradient, and uses grad_sum to implement gradient accumulation.
- `TrainOptim` updates parameters.
- `TrainClear` clears the gradient accumulation variable grad_sum.

```python
_sum_op = ops.MultitypeFuncGraph("grad_sum_op")
_clear_op = ops.MultitypeFuncGraph("clear_op")


@_sum_op.register("Tensor", "Tensor")
def _cumulative_gard(grad_sum, grad):
    """Apply gard sum to cumulative gradient."""
    add = ops.AssignAdd()
    return add(grad_sum, grad)


@_clear_op.register("Tensor", "Tensor")
def _clear_grad_sum(grad_sum, zero):
    """Apply zero to clear grad_sum."""
    success = True
    success = ops.depend(success, ops.assign(grad_sum, zero))
    return success


class TrainForwardBackward(Cell):
    def __init__(self, network, optimizer, grad_sum, sens=1.0):
        super(TrainForwardBackward, self).__init__(auto_prefix=False)
        self.network = network
        self.network.set_grad()
        self.network.add_flags(defer_inline=True)
        self.weights = ParameterTuple(network.trainable_params())
        self.optimizer = optimizer
        self.grad_sum = grad_sum
        self.grad = ops.GradOperation(get_by_list=True, sens_param=True)
        self.sens = sens
        self.hyper_map = ops.HyperMap()

    def construct(self, *inputs):
        weights = self.weights
        loss = self.network(*inputs)
        sens = ops.Fill()(ops.DType()(loss), ops.Shape()(loss), self.sens)
        grads = self.grad(self.network, weights)(*inputs, sens)
        return ops.depend(loss, self.hyper_map(ops.partial(_sum_op), self.grad_sum, grads))


class TrainOptim(Cell):
    def __init__(self, optimizer, grad_sum):
        super(TrainOptim, self).__init__(auto_prefix=False)
        self.optimizer = optimizer
        self.grad_sum = grad_sum

    def construct(self):
        return self.optimizer(self.grad_sum)


class TrainClear(Cell):
    def __init__(self, grad_sum, zeros):
        super(TrainClear, self).__init__(auto_prefix=False)
        self.grad_sum = grad_sum
        self.zeros = zeros
        self.hyper_map = ops.HyperMap()

    def construct(self):
        success = self.hyper_map(ops.partial(_clear_op), self.grad_sum, self.zeros)
        return success
```

### Defining the Training Process

Each mini-batch calculates the loss and gradient through forward and backward training, and uses mini_steps to control the accumulated times before each parameter update. After the number of accumulation times is reached, the parameter is updated and the accumulated gradient variable is cleared.

```python
class GradientAccumulation:
    def __init__(self, network, loss_fn, optimizer):
        self._network = network
        self._loss_fn = loss_fn
        self._optimizer = optimizer

        params = self._optimizer.parameters
        self._grad_sum = params.clone(prefix="grad_sum", init='zeros')
        self._zeros = params.clone(prefix="zeros", init='zeros')
        self._train_forward_backward = self._build_train_forward_backward_network()
        self._train_optim = self._build_train_optim()
        self._train_clear = self._build_train_clear()

    @staticmethod
    def _transform_callbacks(callbacks):
        """Transform callback to a list."""
        if callbacks is None:
            return []

        if isinstance(callbacks, Iterable):
            return list(callbacks)

        return [callbacks]

    def _build_train_forward_backward_network(self):
        """Build forward and backward network"""
        network = self._network
        network = nn.WithLossCell(network, self._loss_fn)
        loss_scale = 1.0
        network = TrainForwardBackward(network, self._optimizer, self._grad_sum, loss_scale).set_train()
        return network

    def _build_train_optim(self):
        """Build optimizer network"""
        network = TrainOptim(self._optimizer, self._grad_sum).set_train()
        return network

    def _build_train_clear(self):
        """Build clear network"""
        network = TrainClear(self._grad_sum, self._zeros).set_train()
        return network

    def train_process(self, epoch, train_dataset, mini_steps=None):
        """
        Training process. The data would be passed to network directly.
        """
        dataset_helper = DatasetHelper(train_dataset, dataset_sink_mode=False, epoch_num=epoch)

        for i in range(epoch):
            step = 0
            for k, next_element in enumerate(dataset_helper):
                loss = self._train_forward_backward(*next_element)
                if (k + 1) % mini_steps == 0:
                    step += 1
                    print("epoch:", i + 1, "step:", step, "loss is ", loss)
                    self._train_optim()
                    self._train_clear()

            train_dataset.reset()

        save_checkpoint(self._train_forward_backward, "gradient_accumulation.ckpt", )
```

### Training and Saving the Model

Call the network, optimizer, and loss function, and then customize the `train_process` API of `GradientAccumulation` to train the model.

```python
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='MindSpore Gard Cumulative Example')
    parser.add_argument('--device_target', type=str, default="GPU", choices=['GPU'],
                        help='device where the code will be implemented (default: GPU)')
    parser.add_argument('--data_path', type=str, default="./Data",
                        help='path where the dataset is saved')
    args = parser.parse_args()

    context.set_context(mode=context.GRAPH_MODE, device_target=args.device_target)
    ds_train = create_dataset(os.path.join(args.data_path, "train"), 32)

    network = LeNet5(10)
    net_loss = nn.SoftmaxCrossEntropyWithLogits(is_grad=False, sparse=True, reduction="mean")
    net_opt = nn.Momentum(network.trainable_params(), 0.01, 0.9)
    model = GradientAccumulation(network, net_loss, net_opt)

    print("============== Starting Training ==============")
    model.train_process(10, ds_train, mini_steps=4)
```

## Experiment Result

After 10 epochs, the accuracy on the test set is about 96.31%.

**Training Execution:**

1. Run the training code and view the running result.

    ```shell
    python train.py --data_path=./MNIST_Data
    ```

    The output is as follows. The loss value decreases during training.

    ```shell
    epoch: 1 step: 27 loss is  0.3660637
    epoch: 1 step: 28 loss is  0.25238192
    ...
    epoch: 3 step: 2 loss is  0.12296932
    epoch: 3 step: 3 loss is  0.15799297
    ...
    epoch: 10 step: 448 loss is  0.06443884
    epoch: 10 step: 449 loss is  0.0067842817
    ```

2. Check the saved checkpoint files.

    The model file `gradient_accumulation.ckpt` is saved during training.

**Model Validation:**

Use the saved checkpoint file to load the validation dataset through [eval.py](<https://gitee.com/mindspore/mindspore/blob/r1.0/model_zoo/official/cv/lenet/train.py>) in the lenet directory of model_zoo.

```shell
python eval.py --data_path=./MNIST_Data --ckpt_path=./gradient_accumulation.ckpt --device_target=GPU
```

The output is as follows. The accuracy of the validation dataset is about 96.31%, which is the same as the result when the value of batch_size is 32.

```shell
============== Starting Testing ==============
============== {'Accuracy': 0.9631730769230769} ==============
```
