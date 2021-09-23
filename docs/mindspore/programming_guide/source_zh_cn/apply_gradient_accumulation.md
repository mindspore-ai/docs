# 应用梯度累积算法

`Linux` `GPU` `模型调优` `中级` `高级`

<!-- TOC -->

- [应用梯度累积算法](#应用梯度累积算法)
    - [概述](#概述)
    - [单机模式](#单机模式)
        - [导入需要的库文件](#导入需要的库文件)
        - [加载数据集](#加载数据集)
        - [定义网络](#定义网络)
        - [定义训练流程](#定义训练流程)
        - [定义训练模型](#定义训练模型)
        - [训练并保存模型](#训练并保存模型)
        - [实验结果](#实验结果)
    - [并行模式](#并行模式)
        - [定义并行训练流程](#定义并行训练流程)
        - [定义并行训练模型](#定义并行训练模型)
        - [训练模型](#训练模型)

<!-- /TOC -->

<a href="https://gitee.com/mindspore/docs/blob/r1.5/docs/mindspore/programming_guide/source_zh_cn/apply_gradient_accumulation.md" target="_blank"><img src="https://gitee.com/mindspore/docs/raw/r1.5/resource/_static/logo_source.png"></a>
&nbsp;&nbsp;
<a href="https://obs.dualstack.cn-north-4.myhuaweicloud.com/mindspore-website/notebook/r1.5/notebook/mindspore_apply_gradient_accumulation.ipynb"><img src="https://gitee.com/mindspore/docs/raw/r1.5/resource/_static/logo_notebook.png"></a>
&nbsp;&nbsp;
<a href="https://authoring-modelarts-cnnorth4.huaweicloud.com/console/lab?share-url-b64=aHR0cHM6Ly9vYnMuZHVhbHN0YWNrLmNuLW5vcnRoLTQubXlodWF3ZWljbG91ZC5jb20vbWluZHNwb3JlLXdlYnNpdGUvbm90ZWJvb2svbW9kZWxhcnRzL21pbmRzcG9yZV9hcHBseV9ncmFkaWVudF9hY2N1bXVsYXRpb24uaXB5bmI=&imageid=65f636a0-56cf-49df-b941-7d2a07ba8c8c" target="_blank"><img src="https://gitee.com/mindspore/docs/raw/r1.5/resource/_static/logo_modelarts.png"></a>

## 概述

本教程介绍梯度累积的训练方式，目的是为了解决由于内存不足导致某些大型网络无法训练大Batch_size的问题。

传统的训练方式是每次计算得到loss和梯度后，直接用所得梯度对参数进行更新。

与传统的训练方式不同，梯度累积引入Mini-batch的概念，首先对每个Mini-batch的数据计算loss和梯度，但不立即更新模型参数，而是先对所得梯度进行累加，然后在指定数量（N）个Mini-batch之后，用累积后的梯度更新网络参数。下次训练前清空过往累积梯度后重新累加，如此往复。最终目的是为了达到跟直接用N*Mini-batch数据训练几乎同样的效果。

本篇教程将分别介绍在单机模式和并行模式下如何实现梯度累积训练。

## 单机模式

在单机模式下，主要通过将训练流程拆分为正向反向训练、参数更新和累积梯度清理三个部分实现梯度累积。这里以MNIST作为示范数据集，自定义简单模型实现梯度累积需要如下几个步骤。

> 你可以在这里下载主要的训练样例代码：<https://gitee.com/mindspore/docs/tree/r1.5/docs/sample_code/gradient_accumulation>

### 导入需要的库文件

下列是我们所需要的公共模块及MindSpore的模块及库文件。

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

### 加载数据集

利用MindSpore的`dataset`提供的`MnistDataset`接口加载MNIST数据集，此部分代码由`model_zoo`中`lenet`目录下的[dataset.py](https://gitee.com/mindspore/models/blob/master/official/cv/lenet/src/dataset.py)导入。

### 定义网络

这里以LeNet网络为例进行介绍，当然也可以使用其它的网络，如ResNet-50、BERT等, 此部分代码由`model_zoo`中`lenet`目录下的[lenet.py](https://gitee.com/mindspore/models/blob/master/official/cv/lenet/src/lenet.py)导入。

### 定义训练流程

将训练流程拆分为正向反向训练、参数更新和累积梯度清理三个部分：

- `TrainForwardBackward`计算loss和梯度，利用grad_sum实现梯度累加。
- `TrainOptim`实现参数更新。
- `TrainClear`实现对梯度累加变量grad_sum清零。

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

### 定义训练模型

每个Mini-batch通过正反向训练计算loss和梯度，通过mini_steps控制每次更新参数前的累加次数。达到累加次数后进行参数更新和
累加梯度变量清零。

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

### 训练并保存模型

调用网络、优化器及损失函数，然后自定义`GradientAccumulation`的`train_process`接口，进行模型训练。

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

### 实验结果

在经历了10轮epoch之后，在测试集上的精度约为96.31%。

**执行训练:**

1. 运行训练代码，查看运行结果。

    ```bash
    python train.py --data_path=./MNIST_Data
    ```

    输出如下，可以看到loss值随着训练逐步降低：

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

2. 查看保存的CheckPoint文件。

    训练过程中保存了CheckPoint文件`gradient_accumulation.ckpt`，即模型文件。

**验证模型:**

通过`model_zoo`中`lenet`目录下的[eval.py](https://gitee.com/mindspore/models/blob/master/official/cv/lenet/train.py)，使用保存的CheckPoint文件，加载验证数据集，进行验证。

```bash
python eval.py --data_path=./MNIST_Data --ckpt_path=./gradient_accumulation.ckpt --device_target=GPU
```

输出如下，可以看到使用验证的数据集，正确率在96.31%左右，与batch_size为32的验证结果一致。

```text
============== Starting Testing ==============
============== {'Accuracy': 0.9631730769230769} ==============
```

## 并行模式

在`SEMI_AUTO_PARALLEL`和`AUTO_PARALLEL`模式下使用梯度累积，主要是将累积迭代和更新迭代作为两张图下发并且交替执行。在累积迭代图上，只执行正反向运算及梯度累加。在更新迭代图上，执行正反向运算和参数更新。本小节将以[分布式并行训练教程](https://www.mindspore.cn/docs/programming_guide/zh-CN/r1.5/distributed_training_ascend.html)中的样例为基础进行介绍，具体分为如下几个步骤。

> 你可以在这里下载主要的训练样例代码：<https://gitee.com/mindspore/docs/tree/r1.5/docs/sample_code/distributed_training>

### 定义并行训练流程

通常情况下，定义了正向网络后会使用[TrainOneStepCell](https://www.mindspore.cn/docs/api/zh-CN/r1.5/api_python/nn/mindspore.nn.TrainOneStepCell.html)将网络正反向及优化器关联到一起。但是梯度累积时存在累积和更新两种情况，所以我们要基于原有类定义做一些改造。样例代码如下：

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

在`TrainOneStepCell`的基础上，增加累积标记`accumulation`和累积梯度参数`accu_grads`的定义，分别用于区分训练流程和保存累积梯度值。在累积迭代图上，`accumulation`为True，只执行正反向运算并将梯度累加到参数`accu_grads`。在更新迭代图上，`accumulation`为False，执行正反向运算和参数更新。

> 由于并行模式下的梯度累积实现需要结合框架内部的图优化完成，所以网络中定义的`accumulation`和`accu_grads`为特定字符，不能修改。

在动态loss scale场景下，除了梯度需要累积外，溢出标志位也需要累积判断，可以基于[TrainOneStepWithLossScaleCell](https://www.mindspore.cn/docs/api/zh-CN/r1.5/api_python/nn/mindspore.nn.TrainOneStepWithLossScaleCell.html#mindspore.nn.TrainOneStepWithLossScaleCell)改造，实现代码如下：

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

其中`accu_overflow`是专门用来保存累积溢出标志位的参数。

### 定义并行训练模型

经过`cell_wrapper`封装的网络已经包含了正反向和优化器实现，我们还需要将数据集对接到网络并实现两张图交替执行。这里基于框架中的[Model](https://www.mindspore.cn/docs/api/zh-CN/r1.5/api_python/mindspore/mindspore.Model.html#mindspore.Model)接口实现上述功能。

```python
import math
from mindspore.train.callback import RunContext
from mindspore import context
from mindspore.context import ParallelMode
from mindspore import Model, connect_network_with_dataset
from mindspore import pytype_to_dtype
from mindspore._c_expression import init_exec_dataset
from mindspore.train.train_thor.dataset_helper import DatasetHelper


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

在样例代码中，子类`Model_ACCU`改写了基类的`_exec_preprocess`数据集封装和`_train_dataset_sink_process`训练循环下沉方法，分别下发累积迭代图`(accumulation=True）`及更新迭代图`(accumulation=False)`的数据子图和训练图，实现交替执行的训练过程。累积迭代图的下沉步数为`grad_accumulation_step`减1，更新迭代图下沉步数为1。

### 训练模型

完成上述定义后，即可利用训练接口完成模型训练。首先需要在`context.set_auto_parallel_context`配置`grad_accumulation_step`参数，使能梯度累积。其次利用改造的`cell_wrapper`封装网络结构，传入`Model_ACCU`中初始化模型。

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

在日志中可以检索到如下的日志打印：

```text
epoch: 1 step: 234, loss is 1.7588712
epoch: 2 step: 234, loss is 1.7275971
epoch: 3 step: 234, loss is 1.5423206
epoch: 4 step: 234, loss is 1.2762429
epoch: 5 step: 234, loss is 1.0915408
```
