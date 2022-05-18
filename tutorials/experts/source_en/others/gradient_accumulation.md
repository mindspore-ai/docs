# Gradient Accumulation Algorithm

<a href="https://gitee.com/mindspore/docs/blob/master/tutorials/experts/source_en/others/gradient_accumulation.md" target="_blank"><img src="https://mindspore-website.obs.cn-north-4.myhuaweicloud.com/website-images/master/resource/_static/logo_source_en.png"></a>

## Overview

This tutorial introduces the training algorithm of gradient accumulation, the purpose of which is to solve the OOM (Out Of Memory) problem that the Batch size is too large to train the neural network or the network model is too large to load due to insufficient memory.

## Gradient Accumulation Principle

Gradient accumulation is a way of training a neural network in which data samples are split into several small Batches by Batch and then calculated sequentially.

Before we discuss the gradient accumulation further, check the calculation process of the neural network.

Deep learning models are made up of many interconnected neural network units, and in all neural network layers, sample data propagates continuously forward. After passing through all the layers, the network model outputs the predicted values of the samples, and then calculates the loss values (errors) for each sample through the loss function. The neural network calculates the gradient of the loss value relative to the model parameters by backpropagation. Finally, the gradient information is used to update the parameters in the network model.

The optimizer is a mathematical formula used to update the weight parameters of the network model. Take a simple stochastic gradient descent (SGD) algorithm as an example.

Assuming the Loss Function function formula is:

$$Loss(\theta)=\frac{1}{2}\left(h(x^{k})-y^{k}\right)^{2}$$

When building a model, the optimizer is used to calculate the algorithm that minimizes losses. Here the SGD algorithm uses the Loss function to update the weight parameter formula as follows:

$$\theta{i}=\theta_{i-1}-lr * grad_{i}$$

where $\theta$ is the trainable parameter (weight or error) in the network model. lr is the learning rate, and $grad_{i}$ is the loss relative to network model parameter.

Gradient accumulation only calculates the neural network model, does not update the parameters of the network model in time, and accumulates the gradient information when calculation, and finally uses the accumulated gradient to update the parameters.

$$accumulated=\sum_{i=0}^{N} grad_{i}$$

When the model variables are not updated, the original data Batch is actually divided into several Mini-Batches, and the samples used in each step are actually smaller data sets.

The variables are not updated within N steps, so that all Mini-Batches use the same model variables to calculate the gradient, to ensure that the same gradient and weight information is calculated, which is equivalent to using the original Batch size without splitting.

$$\theta{i}=\theta_{i-1}-lr * \sum_{i=0}^{N} grad_{i}$$

Eventually accumulating the gradient in the previous step yields the sum of the gradients of the same size as using the global Batche size.

![](https://mindspore-website.obs.cn-north-4.myhuaweicloud.com/website-images/master/tutorials/experts/source_zh_cn/others/images/GradientAccumulation1.png)

In the actual project, there are two points to pay attention to on the tuning parameters and algorithms:

1. **learning rate**: Under certain conditions, the larger the Batch size, the better the training effect. The gradient accumulation simulates the effect of the increase of the Batch size. If the accumulation steps is 4, the Batch size is increased by 4 times. According to experience, the learning rate needs to be appropriately amplified when using gradient accumulation.
2. **Batch Norm**: Batch size simulation amplification effect is performed when the accumulation steps are 4. Compared with the real Batch size, the distribution of the data is not exactly the same, and the mean and variance calculated by BN of 4 times Batch size is not the same as the actual data mean and variance, so some implementations will use Group Norm instead of Batch Norm.

## Gradient Accumulation Implement

The following tutorial content will introduce the implementation of gradient accumulation training in standalone mode and Boost mode.

> Note: `auto_parallel` and `semi_auto_parallel` don't support the training way of gradient accumulation.

### Standalone Mode

In standalone mode, the training process consists of three parts: forward and backward training, parameter update, and accumulated gradient clearance.

MNIST is used as an example dataset. To customize a simple model to implement gradient accumulation, perform the following steps:

> Download the main training sample code: [train.py](https://gitee.com/mindspore/docs/blob/master/docs/sample_code/gradient_accumulation/train.py).
>

Since you need to use the LeNet network in the models repository, please execute the following command to pull the code of the models repository:

```text
git clone https://gitee.com/mindspore/models.git
```

If the models repository is not in the system path, it needs to be in ` train.py ` add the following two pieces of code at the beginning of the code.

```python
import sys
sys.path.append(path to models repository)
```

#### Importing Library Files

The following are the required public modules and MindSpore modules and library files.

```python
import argparse
import os
from collections.abc import Iterable

import mindspore.nn as nn
from mindspore import ParameterTuple, set_context, GRAPH_MODE
from mindspore import DatasetHelper, save_checkpoint
from mindspore.nn import Cell
import mindspore.ops as ops
from models.official.cv.lenet.src.dataset import create_dataset
from models.official.cv.lenet.src.lenet import LeNet5
```

#### Loading the Dataset

Use the `MnistDataset` API provided by `dataset` of MindSpore to load the MNIST dataset. The code is imported from [dataset.py](https://gitee.com/mindspore/models/blob/master/official/cv/lenet/src/dataset.py) in the `lenet` directory of models.

#### Defining the Network

LeNet is used as an example network. You can also use other networks, such as ResNet-50 and BERT. The code is imported from [lenet.py](https://gitee.com/mindspore/models/blob/master/official/cv/lenet/src/lenet.py) in the `lenet` directory of models.

#### Defining the Training Process

The training process consists of three parts: forward and backward training, parameter update, and accumulated gradient clearance.

- `TrainForwardBackward` calculates the loss and gradient, and uses grad_sum to implement gradient accumulation.
- `TrainOptim` updates parameters.
- `TrainClear` clears the gradient accumulation variable grad_sum.

```python
_sum_op = ops.MultitypeFuncGraph("grad_sum_op")
_clear_op = ops.MultitypeFuncGraph("clear_op")


@_sum_op.register("Tensor", "Tensor")
def _cumulative_grad(grad_sum, grad):
    """Apply grad sum to cumulative gradient."""
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

#### Defining the Training Model

Each mini-batch computes the loss and gradient through forward and backward training, and uses mini_steps to control the accumulated times before each parameter update. After the number of accumulation times is reached, the parameter is updated and the accumulated gradient variable is cleared.

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

#### Training and Saving the Model

Call the network, optimizer, and loss function, and then customize the `train_process` API of `GradientAccumulation` to train the model.

```python
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='MindSpore Grad Cumulative Example')
    parser.add_argument('--device_target', type=str, default="GPU", choices=['GPU'],
                        help='device where the code will be implemented (default: GPU)')
    parser.add_argument('--data_path', type=str, default="./Data",
                        help='path where the dataset is saved')
    args = parser.parse_args()

    set_context(mode=GRAPH_MODE, device_target=args.device_target)
    ds_train = create_dataset(os.path.join(args.data_path, "train"), 32)

    net = LeNet5(10)
    net_loss = nn.SoftmaxCrossEntropyWithLogits(sparse=True, reduction="mean")
    net_opt = nn.Momentum(net.trainable_params(), 0.01, 0.9)
    model = GradientAccumulation(net, net_loss, net_opt)

    print("============== Starting Training ==============")
    model.train_process(10, ds_train, mini_steps=4)
```

#### Experiment Result

After 10 epochs, the accuracy on the test set is about 96.31%.

**Start training:**

1. Run the training code and view the running result.

    ```bash
    python train.py --data_path=./MNIST_Data
    ```

    The output is as follows. You can see that the loss value decreases with the training.

    ```text
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

    The checkpoint file `gradient_accumulation.ckpt`, that is, the model file, is saved during training.

**Validate the model:**

Through the [eval.py](https://gitee.com/mindspore/models/blob/master/official/cv/lenet/eval.py) in the `lenet` directory in ModelZoo, use the saved CheckPoint file, load the verification dataset, and verify it.

```bash
python eval.py --data_path=./MNIST_Data --ckpt_path=./gradient_accumulation.ckpt --device_target=GPU
```

The output is as follows. The accuracy of the validation dataset is about 96.31%, which is the same as the result when the value of batch_size is 32.

```text
============== Starting Testing ==============
============== {'Accuracy': 0.9631730769230769} ==============
```

### Boost Model

In Boost mode, as long as you simply call Boost's gradient accumulation interface, you can realize the gradient accumulation function. MNIST is also used as a demonstration dataset to show how to call the Boost interface to implement the gradient accumulation function.

> You can download the main tranining example code here: [train_and_eval_boost.py](https://gitee.com/mindspore/docs/blob/master/docs/sample_code/gradient_accumulation/train_and_eval_boost.py).

#### Importing Library Files

The following are the required public modules and MindSpore modules and library files.

```python
import argparse
import os

import mindspore.nn as nn
from mindspore import Model, set_context, GRAPH_MODE
from mindspore.nn import WithLossCell, TrainOneStepCell, Accuracy
from mindspore.boost import GradientAccumulation
import mindspore.ops as ops
from mindspore import LossMonitor, TimeMonitor
from mindspore import load_checkpoint, load_param_into_net

from models.official.cv.lenet.src.dataset import create_dataset
from models.official.cv.lenet.src.lenet import LeNet5
```

#### Loading the Dataset

Use the `MnistDataset` API provided by `dataset` of MindSpore to load the MNIST dataset. The code is imported from [dataset.py](https://gitee.com/mindspore/models/blob/master/official/cv/lenet/src/dataset.py) in the `lenet` directory of models.

#### Defining the Network

LeNet is used as an example network. You can also use other networks, such as ResNet-50 and BERT. The code is imported from [lenet.py](https://gitee.com/mindspore/models/blob/master/official/cv/lenet/src/lenet.py) in the `lenet` directory of models.

#### Defining the Training Model

We can call `GradientAccumulation` under MindSpore Boost to enable gradient accumulation, controlling the number of accumulations before each parameter update by max_accumulation_step.

Parameter updates and zeroing of the accumulated gradient variables after the number of accumulations is reached, only the interface needs to be called based on the `TrainOneStepCell` definition `TrainGradAccumulationStepsCell`.

```python
class TrainGradAccumulationStepsCell(TrainOneStepCell):
    """construct train accu step cell"""
    def __init__(self, network, optimizer, sens=1.0, max_accumulation_step=1):
        super(TrainGradAccumulationStepsCell, self).__init__(network, optimizer, sens)
        self.max_accumulation_step = max_accumulation_step
        self.grad_accumulation = GradientAccumulation(self.max_accumulation_step, self.optimizer)

    def construct(self, *inputs):
        loss = self.network(*inputs)
        sens = ops.fill(loss.dtype, loss.shape, self.sens)
        grads = self.grad(self.network, self.weights)(*inputs, sens)
        grads = self.grad_reducer(grads)
        loss = self.grad_accumulation(loss, grads)
        return loss
```

#### Training the Model and Make Inference

After the network is defined, it can be trained. After the training, the ckpt file saved during the training process is loaded for inference, and the accuracy of the model can be obtained.

```python
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='MindSpore Grad Cumulative Example')
    parser.add_argument('--device_target', type=str, default="Ascend", choices=['Ascend', 'GPU'],
                        help='device where the code will be implemented (default: Ascend)')
    parser.add_argument('--data_path', type=str, default="./Data",
                        help='path where the dataset is saved')
    args = parser.parse_args()

    set_context(mode=GRAPH_MODE, device_target=args.device_target)
    ds_train = create_dataset(os.path.join(args.data_path, "train"), 32)

    net = LeNet5(10)
    net_loss = nn.SoftmaxCrossEntropyWithLogits(sparse=True, reduction="mean")
    net_opt = nn.Momentum(net.trainable_params(), 0.01, 0.9)
    time_cb = TimeMonitor(data_size=ds_train.get_dataset_size())

    train_net = nn.WithLossCell(net, net_loss)
    train_net = TrainGradAccumulationStepsCell(train_net, net_opt, 1.0, 5)
    model = Model(train_net)

    print("============== Starting Training ==============")
    model.train(10, ds_train, callbacks=[time_cb, LossMonitor()])

    print("============== Starting Testing ==============")
    model = Model(net, net_loss, net_opt, metrics={"Accuracy": Accuracy()})
    ds_eval = create_dataset(os.path.join(args.data_path, "test"), 32, 1)
    if ds_eval.get_dataset_size() == 0:
        raise ValueError("Please check dataset size > 0 and batch_size <= dataset size")

    acc = model.eval(ds_eval)
    print("============== {} ==============".format(acc))
```

#### Experiment Result

After 10 epochs, the accuracy on the test set is about 98.30%.

1. Run the training and inference code and view the results.

   ```bash
   python train_and_eval_boost.py --data_path=./MNIST_Data
   ```

   The output is as follows, and you can see that the loss value gradually decreases with the training:

   ```text
   epoch: 1 step: 1875 loss is  0.1889342879
   ...
   epoch: 5 step: 1875 loss is  0.11749879342
   ...
   epoch: 10 step: 1875 loss is  0.00029468764328
   ```

2. Looking at the inference precision, the code saves the checkpoint to the current directory, and then loads the checkpoint inference.

   ```text
   ============== Starting Testing ==============
   ============== {'Accuracy': 0.983072916666} ==============
   ```

## References

- [1] Hermans, Joeri R., Gerasimos Spanakis, and Rico Möckel. "Accumulated gradient normalization." Asian Conference on Machine Learning. PMLR, 2017.
- [2] Lin, Yujun, et al. "Deep gradient compression: Reducing the communication bandwidth for distributed training." arXiv preprint arXiv:1712.01887 (2017).
- [3] [how-to-break-gpu-memory-boundaries-even-with-large-batch-sizes](https://towardsdatascience.com/how-to-break-gpu-memory-boundaries-even-with-large-batch-sizes-7a9c27a400ce)
- [4] [what-is-gradient-accumulation-in-deep-learning](https://towardsdatascience.com/what-is-gradient-accumulation-in-deep-learning-ec034122cfa)