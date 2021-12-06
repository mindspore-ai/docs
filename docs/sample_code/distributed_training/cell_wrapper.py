# Copyright 2021 Huawei Technologies Co., Ltd
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ============================================================================
"""
grad accumulation cell wrapper
"""
import numpy as np
from mindspore import dtype as mstype
from mindspore import ops, context, Tensor, Parameter
from mindspore.nn import Cell, TrainOneStepCell, TrainOneStepWithLossScaleCell
from mindspore.nn.wrap.loss_scale import _grad_scale
from mindspore.common.initializer import initializer
from mindspore.ops.operations.comm_ops import _VirtualDataset

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
    """construct train accu step cell"""
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


class TrainAccuStepsWithLossScaleCell(TrainOneStepWithLossScaleCell):
    """construct train accu step with loss scale cell"""
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


class VirtualDatasetCell(Cell):
    def __init__(self, backbone):
        super(VirtualDatasetCell, self).__init__(auto_prefix=False)
        self._backbone = backbone
        self._virtual_dataset = _VirtualDataset()

    def construct(self, *inputs):
        output = self._virtual_dataset(*inputs)
        return self._backbone(*output)
