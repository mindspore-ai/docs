# Heterogeneous Parallel Training

<a href="https://gitee.com/mindspore/docs/blob/r2.0.0-alpha/docs/mindspore/source_en/design/heterogeneous_training.md" target="_blank"><img src="https://mindspore-website.obs.cn-north-4.myhuaweicloud.com/website-images/r2.0.0-alpha/resource/_static/logo_source_en.png"></a>

## Overview

The heterogeneous parallel training method is to analyze the memory occupation and computational intensity of the operators on the graph, and slice the operators with huge memory consumption or suitable for CPU logic processing to the CPU subgraph, and slice the computationally intensive operators with less memory consumption to the hardware accelerator subgraph. The framework cooperates with different subgraphs for network training, so that subgraphs in different hardware and without dependencies can perform the execution process in parallel.

## Computational Process

A typical computational process for MindSpore heterogeneous parallel training is shown in the following figure:

1. Users set backend for network execution

```python
import mindspore as ms
ms.set_context(device_target="GPU")
```

2. Users set execution backend of specific operators

```python
from mindspore import ops

prim = ops.Add()

prim.set_device("CPU")
```

3. The framework is sliced according to the computational graph operator flag.
4. The framework schedules different back-end execution subgraphs.

Current scenarios that typically use heterogeneous parallel computing are: optimizer heterogeneity, Embedding heterogeneity, and PS heterogeneity.

## Optimizer Heterogeneity

During the training of a large model in PanGu or GPT3, the optimizer state takes up a large amount of memory, which in turn limits the size of the model that can be trained. Using optimizer heterogeneity, assigning optimizers to CPUs for execution can greatly scale the trainable models:

![heterogeneous-heter-opt](https://mindspore-website.obs.cn-north-4.myhuaweicloud.com/website-images/r2.0.0-alpha/docs/mindspore/source_zh_cn/design/images/heter-opt.png)

As shown in the figure, configuring the Adam operator to CPU execution while specifying an accelerator for FP16 computation reduces the parameter memory footprint to 1/3 of the original.

1. Configure the optimizer operators to CPU execution
2. Initialize weight parameters of FP16 and optimizer state variables of FP32
3. Convert the gradient of the input optimizer to FP16 (if the gradient is FP16, you can ignore this step)
4. The weights and gradients are converted to FP32 to participate in the optimizer operation
5. The updated FP32 weights are assigned to the FP16 weights

Sample code of the optimizer heterogeneity is as follows:

```python
import numpy as np
import mindspore as ms
import mindspore.ops as ops
from mindspore.common.initializer import initializer
from mindspore.nn import Optimizer
_adam_opt = ops.MultitypeFuncGraph("adam_opt")
host_assign = ops.Assign()
host_assign.set_device("CPU")
host_cast = ops.Cast()
host_cast.set_device("CPU")
device_cast = ops.Cast()

@_adam_opt.register("Function", "Tensor", "Tensor", "Tensor", "Tensor", "Number", "Tensor", "Tensor", "Tensor",
                    "Tensor", "Bool", "Bool")
def _update_run_kernel(opt, beta1, beta2, eps, lr, weight_decay, param, m, v, gradient, decay_flags, optim_filter):
    """
    Update parameters by AdamWeightDecay op.
    """
    success = True
    if optim_filter:
        param32 = host_cast(param, ms.float32)
        gradient = device_cast(gradient, ms.float32)
        if decay_flags:
            next_param = opt(param32, m, v, lr, beta1, beta2, eps, weight_decay, gradient)
        else:
            next_param = opt(param32, m, v, lr, beta1, beta2, eps, 0.0, gradient)
        ret = host_assign(param, host_cast(ops.depend(param32, next_param), ops.dtype(param)))
        return ops.depend(success, ret)
    return success

class AdamWeightDecayOp(Optimizer):
    def __init__(self, params, learning_rate=1e-3, beta1=0.9, beta2=0.999, eps=1e-6, weight_decay=0.0):
        super(AdamWeightDecayOp, self).__init__(learning_rate, params, weight_decay)
        self.beta1 = ms.Tensor(np.array([beta1]).astype(np.float32))
        self.beta2 = ms.Tensor(np.array([beta2]).astype(np.float32))
        self.eps = ms.Tensor(np.array([eps]).astype(np.float32))
        self.moments1 = self.clone_param32(prefix="adam_m", init='zeros')
        self.moments2 = self.clone_param32(prefix="adam_v", init='zeros')
        self.opt = ops.AdamWeightDecay()
        self.hyper_map = ops.HyperMap()
        self.opt.set_device("CPU")

    def construct(self, gradients):
        """AdamWeightDecayOp"""
        lr = self.get_lr()
        if self.is_group:
            if self.is_group_lr:
                optim_result = self.map_reverse(ops.partial(_adam_opt, self.opt, self.beta1, self.beta2, self.eps),
                                                lr, self.weight_decay, self.parameters, self.moments1, self.moments2,
                                                gradients, self.decay_flags, self.optim_filter)
            else:
                optim_result = self.map_reverse(ops.partial(_adam_opt, self.opt, self.beta1, self.beta2, self.eps, lr),
                                                self.weight_decay, self.parameters, self.moments1, self.moments2,
                                                gradients, self.decay_flags, self.optim_filter)
        else:
            optim_result = self.map_reverse(ops.partial(_adam_opt, self.opt, self.beta1, self.beta2, self.eps, lr,
                                                        self.weight_decay), self.parameters, self.moments1, self.moments2,
                                            gradients, self.decay_flags, self.optim_filter)
        return optim_result

    def clone_param32(self, prefix, init=None):
        new = []
        for old_param in self.parameters:
            param_init = init
            if init is None:
                param_init = old_param.init
            new_state = old_param.clone()
            new_state.set_dtype(ms.float32)
            new_state.set_data(initializer(param_init, shape=old_param.shape, dtype=ms.float32))
            new_state.name = prefix + '.' + new_state.name
            new.append(new_state)
        return ms.ParameterTuple(new)
```

Steps 4 and 5 can also be directly fused into the optimizer operator for further optimization. The complete optimizer heterogeneous training process can be found at: <https://gitee.com/mindspore/models/tree/r2.0.0-alpha/official/nlp/Pangu_alpha>.

## Embedding Heterogeneity

In some networks where large Embedding tables need to be checked, the Embedding tables are often hundreds of gigabytes in size, which is limited by the accelerator memory size and cannot be executed by loading the entire table directly onto the accelerator. By putting the operators connected to the weight table on the CPU for execution, we avoid the problem that the accelerator cannot train the network due to memory limitation.

![heterogeneous-heter-embed](https://mindspore-website.obs.cn-north-4.myhuaweicloud.com/website-images/r2.0.0-alpha/docs/mindspore/source_zh_cn/design/images/heter-embed.png)

1. Configure EmbeddingLookup operator to CPU execution

   ```python
   ops.EmbeddingLookup().set_device('CPU')
   ```

2. Configure related optimizers of EmbeddingLookup to CPU execution

   ```python
   use_locking = False
   use_nesterov = False
   ops.FusedSparseLazyAdam(use_locking, use_nesterov).set_device("CPU")
   ```

A sample code for setting up the EmbeddingLookup operator is as follows:

```python
import mindspore.nn as nn
import mindspore.ops as ops
import mindspore as ms
from mindspore.common.initializer import initializer

class EmbeddingLookup(nn.Cell):
    def __init__(self, vocab_size, embedding_size, param_init='normal',
                 target='CPU', sparse=True):
        """Initialize EmbeddingLookup."""
        super(EmbeddingLookup, self).__init__()
        validator.check_value_type('sparse', sparse, [bool], self.cls_name)
        self.vocab_size = validator.check_positive_int(vocab_size, 'vocab_size')
        self.target = target
        self.sparse = sparse
        if target not in ('CPU', 'DEVICE'):
            raise ValueError('Attr \'target\' of \'EmbeddingLookup\' Op passed '
                             + str(target) + ', should be one of values in \'CPU\', \'DEVICE\'.')
        if not sparse and target == 'CPU':
            raise ValueError('When target is CPU, embedding_lookup must be sparse.')
        if sparse:
            self.gatherv2 = ops.SparseGatherV2()
        else:
            self.gatherv2 = ops.Gather()
        self.embeddinglookup = ops.EmbeddingLookup().set_device('CPU')
        self.embedding_size = validator.check_positive_int(embedding_size, 'embedding_size')
        self.embedding_table = ms.Parameter(initializer(param_init, [self.vocab_size, self.embedding_size]),
                                            name='embedding_table')

    def construct(self, indices):
        if self.target == "CPU":
            out = self.embeddinglookup(self.embedding_table, indices, 0)
        else:
            out = self.gatherv2(self.embedding_table, indices, 0)
        return out
```

EmbeddingLookup, FTRL, LazyAdam and other operators in the current nn directory are encapsulated the heterogeneous interface, and the user only needs to set the target attribute to CPU or DEVICE to switch the execution backend.

For the overall calling process, refer to <https://gitee.com/mindspore/models/tree/r2.0.0-alpha/official/recommend/Wide_and_Deep>.

## PS Heterogeneity

When the EmbeddingTable reaches T level and the single machine memory cannot be put down, Parameter Server is used to pull and update the weights by heterogeneous Pull/Push operators.

![heterogeneous-heter-ps](https://mindspore-website.obs.cn-north-4.myhuaweicloud.com/website-images/r2.0.0-alpha/docs/mindspore/source_zh_cn/design/images/heter-ps.png)

Parameter Server encapsulates heterogeneous processes, and users only need to configure parameters to use PS. For the detailed configuration process, refer to [Parameter Server training process](https://www.mindspore.cn/tutorials/experts/en/r2.0.0-alpha/parallel/parameter_server_training.html).

In addition, the process of using PS is also available in the wide&deep network and can be found at: <https://gitee.com/mindspore/models/tree/r2.0.0-alpha/official/recommend/Wide_and_Deep>.

## Constraints

Currently requires the user to specify the back-end of the operator execution and does not support automatic configuration based on the network.
