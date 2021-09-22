# Applying a Gradient Accumulation Algorithm

`Linux` `GPU` `Model Optimization` `Intermediate` `Expert`

<!-- TOC -->

- [Applying a Gradient Accumulation Algorithm](#applying-a-gradient-accumulation-algorithm)
    - [Overview](#overview)
    - [Standalone Mode](#standalone-mode)
        - [Importing Library Files](#importing-library-files)
        - [Loading the Dataset](#loading-the-dataset)
        - [Defining the Network](#defining-the-network)
        - [Defining the Training Process](#defining-the-training-process)
        - [Defining the Training Model](#defining-the-training-model)
        - [Training and Saving the Model](#training-and-saving-the-model)
    - [Experiment Result](#experiment-result)
    - [Parallel Mode](#parallel-mode)
        - [Defining the Parallel Training Process](#defining-the-parallel-training-process)
        - [Defining the Parallel Training Model](#defining-the-parallel-training-model)
        - [Training the Model](#training-the-model)

<!-- /TOC -->

<a href="https://gitee.com/mindspore/docs/blob/master/docs/mindspore/programming_guide/source_en/apply_gradient_accumulation.md" target="_blank"><img src="https://gitee.com/mindspore/docs/raw/master/resource/_static/logo_source_en.png"></a>

## Overview

This tutorial describes the gradient accumulation training methods to solve the problem that some large-scale networks cannot train large batch_size due to insufficient memory.

In a traditional training method, after a loss and a gradient are computed each time, a parameter is directly updated by using the obtained gradient.

Compared to the traditional training method, mini-batch is introduced to the gradient accumulation. The loss and gradient are computed for each mini-batch data, but the model parameters are not updated immediately. Instead, the obtained gradients are accumulated first, and then after a specified number (N) of mini-batches, the accumulated gradient is used to update the network parameters. Before the next training, the accumulated gradients are cleared and re-accumulated. The ultimate objective is to achieve the same effect as training with N x Mini-batch data.

This tutorial describes how to implement gradient accumulation training in standalone mode and parallel mode, respectively.

## Standalone Mode

In standalone mode, the training process consists of three parts: forward and backward training, parameter update, and accumulated gradient clearance. MNIST is used as an example dataset. To customize a simple model to implement gradient accumulation, perform the following steps:

> Download the main training sample code: <https://gitee.com/mindspore/docs/tree/master/docs/sample_code/gradient_accumulation>

### Importing Library Files

The following are the required public modules and MindSpore modules and library files.

```python
import argparse
import os
from collections.abc import Iterable

import mindspore.nn as nn
from mindspore import ParameterTuple
from mindspore import context, DatasetHelper, save_checkpoint
from mindspore.nn import Cell
import mindspore.ops as ops
from model_zoo.official.cv.lenet.src.dataset import create_dataset
from model_zoo.official.cv.lenet.src.lenet import LeNet5
```

### Loading the Dataset

Use the `MnistDataset` API provided by `dataset` of MindSpore to load the MNIST dataset. The code is imported from [dataset.py](<https://gitee.com/mindspore/models/blob/master/official/cv/lenet/src/dataset.py>) in the `lenet` directory of `model_zoo`.

### Defining the Network

LeNet is used as an example network. You can also use other networks, such as ResNet-50 and BERT. The code is imported from [lenet.py](<https://gitee.com/mindspore/models/blob/master/official/cv/lenet/src/lenet.py>) in the `lenet` directory of `model_zoo`.

### Defining the Training Process

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

### Defining the Training Model

Each mini-batch computes the loss and gradient through forward and backward training, and uses mini_steps to control the accumulated times before each parameter update. After the number of accumulation times is reached, the parameter is updated
and the accumulated gradient variable is cleared.

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
    parser = argparse.ArgumentParser(description='MindSpore Grad Cumulative Example')
    parser.add_argument('--device_target', type=str, default="GPU", choices=['GPU'],
                        help='device where the code will be implemented (default: GPU)')
    parser.add_argument('--data_path', type=str, default="./Data",
                        help='path where the dataset is saved')
    args = parser.parse_args()

    context.set_context(mode=context.GRAPH_MODE, device_target=args.device_target)
    ds_train = create_dataset(os.path.join(args.data_path, "train"), 32)

    net = LeNet5(10)
    net_loss = nn.SoftmaxCrossEntropyWithLogits(sparse=True, reduction="mean")
    net_opt = nn.Momentum(net.trainable_params(), 0.01, 0.9)
    model = GradientAccumulation(net, net_loss, net_opt)

    print("============== Starting Training ==============")
    model.train_process(10, ds_train, mini_steps=4)
```

## Experiment Result

After 10 epochs, the accuracy on the test set is about 96.31%.

**Start training.**

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

**Validate the model.**

Use the saved checkpoint file to load the validation dataset through [eval.py](<https://gitee.com/mindspore/models/blob/master/official/cv/lenet/train.py>) in the lenet directory of model_zoo.

```bash
python eval.py --data_path=./MNIST_Data --ckpt_path=./gradient_accumulation.ckpt --device_target=GPU
```

The output is as follows. The accuracy of the validation dataset is about 96.31%, which is the same as the result when the value of batch_size is 32.

```text
============== Starting Testing ==============
============== {'Accuracy': 0.9631730769230769} ==============
```

## Parallel Mode

If gradient accumulation is used in `SEMI_AUTO_PARALLEL` and `AUTO_PARALLEL` modes, the accumulation steps and update steps are delivered as two graphs and executed alternately. In an accumulation step graph, only the forward and backward operations and gradient accumulation are performed. In an update step graph, the forward and backward operations and parameter updates are performed. The example in [Parallel Distributed Training](https://www.mindspore.cn/docs/programming_guide/en/master/distributed_training_ascend.html) is used to describe the procedure.

> Download the main training sample code: <https://gitee.com/mindspore/docs/tree/master/docs/sample_code/distributed_training>

### Defining the Parallel Training Process

Generally, after the forward network is defined, [TrainOneStepCell](https://www.mindspore.cn/docs/api/en/master/api_python/nn/mindspore.nn.TrainOneStepCell.html) is used to associate the forward and backward networks with the optimizer. However, two different situations, accumulation and update, exist during gradient accumulation. We need to make some modifications based on the original class definition. The sample code is as follows:

```python
import numpy as np
from mindspore import dtype as mstype
from mindspore import ops, context, Tensor, Parameter
from mindspore.nn import TrainOneStepCell
from mindspore.common.initializer import initializer

zeroslike = ops.ZerosLike()
reset_accu_grads = ops.MultitypeFuncGraph("reset_accu_grads")

@reset_accu_grads.register("Tensor")
def _reset_accu_grads(accu_grad):
    succ = True
    return ops.depend(succ, ops.assign(accu_grad, zeroslike(accu_grad)))

cast = ops.Cast()
update_accu_grads = ops.MultitypeFuncGraph("update_accu_grads")


@update_accu_grads.register("Tensor", "Tensor")
def _update_accu_grads(accu_grad, grad):
    succ = True
    return ops.depend(succ, ops.assign_add(accu_grad, cast(grad, mstype.float32)))

class TrainAccuStepsCell(TrainOneStepCell):
    def __init__(self, network, optimizer, sens=1.0):
        super(TrainAccuStepsCell, self).__init__(network, optimizer, sens)
        self.accumulation = False
        self.accumulation_steps = context.get_auto_parallel_context("grad_accumulation_step")
        self.accu_grads = self.weights.clone(prefix="accu_grads", init='zeros')
        self.hyper_map = ops.HyperMap()

    def construct(self, *inputs):
        """Defines the computation performed."""
        weights = self.weights
        loss = self.network(*inputs)
        sens = ops.Fill()(ops.DType()(loss), ops.Shape()(loss), self.sens)
        grads = self.grad(self.network, weights)(*inputs, sens)
        if self.accumulation and self.accumulation_steps > 1:
            accu_succ = self.hyper_map(update_accu_grads, self.accu_grads, grads)
            loss = ops.depend(loss, accu_succ)
        if self.accumulation:
            succ = False
        else:
            grads = self.grad_reducer(grads)
            accu_grads = ops.depend(self.accu_grads, grads)
            accu_succ = self.hyper_map(reset_accu_grads, accu_grads)
            loss = ops.depend(loss, accu_succ)
            succ = self.optimizer(grads)
        return ops.depend(loss, succ)
```

On the basis of `TrainOneStepCell`, definitions of the accumulation flag `accumulation` and the accumulation gradient parameter `accu_grads` are added to distinguish the training process and save the accumulation gradient value, respectively. In an accumulation step graph, if `accumulation` is set to True, only the forward and backward operations are performed and gradients are accumulated to the parameter `accu_grads`. In an update step graph, if `accumulation` is set to False, the forward and backward operations and parameter updates are performed.

> The gradient accumulation in parallel mode needs to be implemented based on the internal graph optimization of the framework. Therefore, `accumulation` and `accu_grads` defined on the network are specific characters and cannot be modified.

In the dynamic loss scale scenario, in addition to the gradient, the overflow flag status also needs to be accumulated. The code can be modified based on [TrainOneStepWithLossScaleCell](https://www.mindspore.cn/docs/api/en/master/api_python/nn/mindspore.nn.TrainOneStepWithLossScaleCell.html#mindspore.nn.TrainOneStepWithLossScaleCell). The implementation code is as follows:

```python
import numpy as np
import mindspore.common.dtype as mstype
from mindspore import ops, context, Tensor, Parameter
from mindspore.nn import TrainOneStepWithLossScaleCell
from mindspore.nn.wrap.loss_scale import _grad_scale
from mindspore.common.initializer import initializer

zeroslike = ops.ZerosLike()
reset_accu_grads = ops.MultitypeFuncGraph("reset_accu_grads")

@reset_accu_grads.register("Tensor")
def _reset_accu_grads(accu_grad):
    succ = True
    return ops.depend(succ, ops.assign(accu_grad, zeroslike(accu_grad)))

cast = ops.Cast()
update_accu_grads = ops.MultitypeFuncGraph("update_accu_grads")


@update_accu_grads.register("Tensor", "Tensor")
def _update_accu_grads(accu_grad, grad):
    succ = True
    return ops.depend(succ, ops.assign_add(accu_grad, cast(grad, mstype.float32)))


class TrainAccuStepsWithLossScaleCell(TrainOneStepWithLossScaleCell):
    def __init__(self, network, optimizer, scale_sense):
        super(TrainAccuStepsWithLossScaleCell, self).__init__(network, optimizer, scale_sense)
        self.accumulation = False
        self.accumulation_steps = context.get_auto_parallel_context("grad_accumulation_step")
        self.one = Tensor(np.array([1]).astype(np.int32))
        self.zero = Tensor(np.array([0]).astype(np.int32))
        self.accu_grads = self.weights.clone(prefix="accu_grads", init='zeros')
        self.accu_overflow = Parameter(initializer(0, [1], mstype.int32))
        self.accu_loss = Parameter(initializer(0, [1], mstype.float32))
        self.cast = ops.Cast()
        self.logical_or = ops.LogicalOr()
        self.not_equal = ops.NotEqual()
        self.select = ops.Select()
        self.reshape = ops.Reshape()

    def construct(self, *inputs):
        """Defines the computation performed."""
        weights = self.weights
        loss = self.network(*inputs)
        scaling_sens = self.scale_sense
        status, scaling_sens = self.start_overflow_check(loss, scaling_sens)
        scaling_sens_filled = ops.ones_like(loss) * ops.cast(scaling_sens, ops.dtype(loss))
        grads = self.grad(self.network, weights)(*inputs, scaling_sens_filled)
        # accumulate gradients
        if self.accumulation and self.accumulation_steps > 1:
            accu_succ = self.hyper_map(update_accu_grads, self.accu_grads, grads)
            loss = ops.depend(loss, accu_succ)
        overflow = self.get_overflow_status(status, grads)
        overflow = self.logical_or(self.not_equal(self.accu_overflow, self.zero), overflow)
        accu_overflow = self.select(overflow, self.one, self.zero)

        if self.accumulation:
            succ = False
            self.accu_overflow = accu_overflow
        else:
            self.accu_overflow = self.zero
            # apply grad reducer on grads
            grads = self.grad_reducer(grads)
            grads = self.hyper_map(ops.partial(_grad_scale, scaling_sens), grads)
            accu_overflow = self.allreduce(accu_overflow)
            overflow = self.less_equal(self.base, accu_overflow)
            accu_grads = ops.depend(self.accu_grads, grads)
            accu_succ = self.hyper_map(reset_accu_grads, accu_grads)
            overflow = ops.depend(overflow, accu_succ)
            overflow = self.reshape(overflow, (()))
            overflow = self.process_loss_scale(overflow)
            if overflow:
                succ = False
            else:
                succ = self.optimizer(grads)

        ret = (loss, overflow, scaling_sens)
        return ops.depend(ret, succ)
```

`accu_overflow` is a parameter used to store the accumulation overflow flag status.

### Defining the Parallel Training Model

The network encapsulated by `cell_wrapper` contains the forward and backward operations and optimizer implementation. You need to connect the dataset to the network and execute the two graphs alternately. The preceding functions are implemented based on the [Model](https://www.mindspore.cn/docs/api/en/master/api_python/mindspore/mindspore.Model.html#mindspore.Model) API in the framework.

```python
import math
from mindspore.train.callback import RunContext
from mindspore import context
from mindspore.context import ParallelMode
from mindspore import Model, connect_network_with_dataset
from mindspore import pytype_to_dtype
from mindspore._c_expression import init_exec_dataset
from mindspore import DatasetHelper


def _convert_type(types):
    """
    Convert from numpy type to tensor type.

    Args:
        types (list): Numpy type list of element in dataset.

    Returns:
        list, list of element in dataset.
    """
    ms_types = []
    for np_type in types:
        ms_type = pytype_to_dtype(np_type)
        ms_types.append(ms_type)
    return ms_types


def _get_types_and_shapes(dataset):
    """Get dataset types and shapes."""
    dataset_types = _convert_type(dataset.output_types())
    dataset_shapes = dataset.output_shapes()
    return dataset_types, dataset_shapes


def _exec_datagraph(exec_dataset, dataset_size, phase='dataset'):
    """Initialize and execute the dataset graph."""
    batch_size = exec_dataset.get_batch_size()
    input_indexs = exec_dataset.input_indexs

    # transform data format
    dataset_types, dataset_shapes = _get_types_and_shapes(exec_dataset)
    init_exec_dataset(exec_dataset.__transfer_dataset__.queue_name,
                      dataset_size,
                      batch_size,
                      dataset_types,
                      dataset_shapes,
                      input_indexs,
                      phase=phase,
                      need_run=False)


class Model_ACCU(Model):
    def __init__(self, network, loss_fn=None, optimizer=None, metrics=None, eval_network=None,
                 eval_indexes=None, amp_level="O0", **kwargs):
        super(Model_ACCU, self).__init__(network, loss_fn, optimizer, metrics, eval_network,
                                         eval_indexes, amp_level, **kwargs)
        self._frequency = context.get_auto_parallel_context("grad_accumulation_step")
        self._train_network = self._build_train_network()

    def _exec_preprocess(self, network, is_train, phase, dataset, dataset_sink_mode, sink_size=-1,
                         epoch_num=1, iter_first_order=1):
        """Initializes dataset."""
        if dataset_sink_mode and not is_train:
            dataset.__loop_size__ = 1
        dataset_helper = DatasetHelper(dataset, dataset_sink_mode, sink_size, epoch_num, iter_first_order)

        if dataset_sink_mode and context.get_context("device_target") != "GPU":
            network = connect_network_with_dataset(network, dataset_helper)
        network.set_train(is_train)
        network.phase = phase

        if self._parallel_mode in (ParallelMode.SEMI_AUTO_PARALLEL, ParallelMode.AUTO_PARALLEL):
            network.set_auto_parallel()

        return dataset_helper, network

    def _train_dataset_sink_process(self, epoch, train_dataset, list_callback=None, cb_params=None, sink_size=-1):
        """
        Training process. The data would be passed to network through dataset channel.

        Args:
            epoch (int): Total number of iterations on the data.
            train_dataset (Dataset): A training dataset iterator. If there is no
                                     loss_fn, a tuple with multiple data (data1, data2, data3, ...) should be
                                     returned and passed to the network. Otherwise, a tuple (data, label) should
                                     be returned. The data and label would be passed to the network and loss
                                     function respectively.
            list_callback (Callback): Executor of callback list. Default: None.
            cb_params (_InternalCallbackParam): Callback parameters. Default: None.
            sink_size (int): Control the amount of data in each sink. Default: -1.
        """
        if sink_size == -1:
            epoch_num = epoch
        else:
            epoch_num = math.ceil(epoch * sink_size / train_dataset.get_dataset_size())

        iter_first_order = 1
        iter_second_order = self._frequency - 1
        train_dataset.__loop_size__ = iter_second_order
        dataset_helper, train_network = self._exec_preprocess(self._train_network,
                                                              is_train=True,
                                                              phase='train',
                                                              dataset=train_dataset,
                                                              dataset_sink_mode=True,
                                                              sink_size=sink_size,
                                                              epoch_num=epoch_num,
                                                              iter_first_order=iter_first_order)

        self._train_network = train_network
        cb_params.train_network = self._train_network
        cb_params.cur_step_num = 0

        run_context = RunContext(cb_params)
        list_callback.begin(run_context)

        # used to stop training for early stop, such as stopAtTIme or stopATStep
        should_stop = False
        switch_branch_one = True
        index_first_order = 0
        train_network_init_flag = True
        has_do_dataset_init = False

        for i in range(epoch):
            cb_params.cur_epoch_num = i + 1
            list_callback.epoch_begin(run_context)
            # for data sink dataset_helper only iter once, other wise iter epoch_size times.
            for inputs in dataset_helper:
                list_callback.step_begin(run_context)
                if switch_branch_one:
                    cb_params.cur_step_num += iter_second_order
                    if train_network_init_flag:
                        self._train_network.add_flags_recursive(accumulation=True)
                    self._train_network.phase = 'train0'
                else:
                    cb_params.cur_step_num += iter_first_order
                    if train_network_init_flag:
                        self._train_network.add_flags_recursive(accumulation=False)
                        train_network_init_flag = False
                    self._train_network.phase = 'train1'
                    if not has_do_dataset_init:
                        _exec_datagraph(train_dataset, iter_first_order, phase='train1_dataset')
                        has_do_dataset_init = True
                switch_branch_one = not switch_branch_one
                outputs = self._train_network(*inputs)
                cb_params.net_outputs = outputs
                list_callback.step_end(run_context)

            list_callback.epoch_end(run_context)
            should_stop = should_stop or run_context.get_stop_requested()
            if should_stop:
                break
        dataset_helper.stop_send()

        list_callback.end(run_context)
```

In the sample code, the subclass `Model_ACCU` rewrites the `_exec_preprocess` dataset encapsulation and the `_train_dataset_sink_process` training offload method of the base class, and delivers the data subgraph and training graph of the accumulation step graph `(accumulation=True)` and the update step graph `(accumulation=False)`, respectively. The training process is alternately executed. The number of offloaded steps of the accumulation step graph is the value of `grad_accumulation_step` minus 1 and that of the update step graph is 1.

### Training the Model

After the preceding definition is complete, you can use the training API to complete model training. Configure the `grad_accumulation_step` parameter in `context.set_auto_parallel_context` to enable gradient accumulation. Then, use the modified `cell_wrapper` to encapsulate the network structure and transfer it to the `Model_ACCU` to initialize the model.

```python
context.set_auto_parallel_context(parallel_mode=ParallelMode.AUTO_PARALLEL, gradients_mean=True, grad_accumulation_step=6)
loss_cb = LossMonitor()
data_path = os.getenv('DATA_PATH')
batch_size = 32
dataset = create_dataset(data_path, batch_size=batch_size)
num_classes = 10
net = resnet50(batch_size, num_classes)
loss = SoftmaxCrossEntropyExpand(sparse=True)
opt = Momentum(filter(lambda x: x.requires_grad, net.get_parameters()), 0.01, 0.9)
net_with_loss = nn.WithLossCell(net, loss)
net_with_loss = VirtualDatasetCell(net_with_loss)
wrap_net = TrainAccuStepsCell(net_with_loss, opt)
model = Model_ACCU(wrap_net)
model.train(epoch_size, dataset, callbacks=[loss_cb], dataset_sink_mode=True)
```

The following information can be found in the logs:

```text
epoch: 1 step: 234, loss is 1.7588712
epoch: 2 step: 234, loss is 1.7275971
epoch: 3 step: 234, loss is 1.5423206
epoch: 4 step: 234, loss is 1.2762429
epoch: 5 step: 234, loss is 1.0915408
```
