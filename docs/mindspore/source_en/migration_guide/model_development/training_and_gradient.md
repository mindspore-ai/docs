# Network Training and Gadient Derivation

<a href="https://gitee.com/mindspore/docs/blob/r1.9/docs/mindspore/source_en/migration_guide/model_development/training_and_gradient.md" target="_blank"><img src="https://mindspore-website.obs.cn-north-4.myhuaweicloud.com/website-images/r1.9/resource/_static/logo_source_en.png"></a>

## Automatic Differentiation

After the forward network is constructed, MindSpore provides an interface to [automatic differentiation](https://mindspore.cn/tutorials/en/master/beginner/autograd.html) to calculate the gradient results of the model.
In the tutorial of [automatic derivation](https://mindspore.cn/tutorials/en/master/advanced/derivation.html), some descriptions of various gradient calculation scenarios are given.

## Network Training

The entire training network consists of the forward network (network and loss function), automatic gradient derivation and optimizer update. MindSpore provides three ways to implement this process.

1. Encapsulate `model` and perform network training by using `model.train` or `model.fit`, such as [model training](https://mindspore.cn/tutorials/en/master/beginner/train.html).

2. Apply the encapsulated [TrainOneStepCell](https://www.mindspore.cn/docs/en/master/api_python/nn/mindspore.nn.TrainOneStepCell.html) and [TrainOneStepWithLossScaleCell](https://www.mindspore.cn/docs/en/master/api_python/nn/mindspore.nn.TrainOneStepWithLossScaleCell.html) separately to common training process and training process with [loss_scale](https://mindspore.cn/tutorials/experts/en/master/others/mixed_precision.html), such as [Quick Start: Linear Fitting](https://mindspore.cn/tutorials/en/master/beginner/quick_start.html).

3. Customize training Cell.

### Customizing training Cell

The first two methods are illustrated with two examples from the official website. For the customized training Cell, we first review what is done in the TrainOneStepCell:

```python
import mindspore as ms
from mindspore import ops, nn
from mindspore.parallel._utils import (_get_device_num, _get_gradients_mean,
                                       _get_parallel_mode)
class TrainOneStepCell(nn.Cell):
    def __init__(self, network, optimizer):
        super(TrainOneStepCell, self).__init__(auto_prefix=False)
        self.network = network  # Network structure with loss
        self.network.set_grad()   # Required for PYNATIVE mode. If True, the inverse network requiring the computation of gradients will be generated when the forward network is executed.
        self.optimizer = optimizer   # Optimizer for parameter updates
        self.weights = self.optimizer.parameters    # Get the parameters of the optimizer
        self.grad = ops.GradOperation(get_by_list=True)   # Get the gradient of all inputs and parameters

        # Related logic of parallel computation
        self.reducer_flag = False
        self.grad_reducer = ops.identity
        self.parallel_mode = _get_parallel_mode()
        self.reducer_flag = self.parallel_mode in (ms.ParallelMode.DATA_PARALLEL, ms.ParallelMode.HYBRID_PARALLEL)
        if self.reducer_flag:
            self.mean = _get_gradients_mean()
            self.degree = _get_device_num()
            self.grad_reducer = nn.DistributedGradReducer(self.weights, self.mean, self.degree)

    def construct(self, *inputs):
        loss = self.network(*inputs)    # Run the forward network and obtain the loss
        grads = self.grad(self.network, self.weights)(*inputs) # Get the gradient of all Parameter free variables
        # grads = grad_op(grads)    # Some computing logic for the gradient can be added here, such as gradient clipping
        grads = self.grad_reducer(grads)  # Gradient aggregation
        self.optimizer(grads)    # Optimizer updates parameters
        return loss
```

The whole training process can be encapsulated into a Cell, in which the forward computation, backward gradient derivation and parameter update of the network can be realized, and we can do some special processing on the gradient after obtaining it.

#### Gradient Clipping

When there is gradient explosion or particularly large gradient during the training process, in a case that training is not stable, you can consider adding gradient clipping. Here is an example of the common use of global_norm for gradient clipping scenarios.

```python
import mindspore as ms
from mindspore import nn, ops

_grad_scale = ops.MultitypeFuncGraph("grad_scale")

@_grad_scale.register("Tensor", "Tensor")
def tensor_grad_scale(scale, grad):
    return grad * ops.cast(ops.Reciprocal()(scale), ops.dtype(grad))

class MyTrainOneStepCell(nn.TrainOneStepWithLossScaleCell):
    """
    Network training package class with gradient clip.

    Append an optimizer to the training network after that the construct function
    can be called to create the backward graph.

    Args:
        network (Cell): The training network.
        optimizer (Cell): Optimizer for updating the weights.
        sens (Number): The adjust parameter. Default value is 1.0.
        grad_clip (bool): Whether clip gradients. Default value is False.
    """

    def __init__(self, network, optimizer, scale_sense=1, grad_clip=False):
        if isinstance(scale_sense, (int, float)):
            scale_sense = ms.FixedLossScaleManager(scale_sense)
        super(MyTrainOneStepCell, self).__init__(network, optimizer, scale_sense)
        self.grad_clip = grad_clip

    def construct(self, x, img_shape, gt_bboxe, gt_label, gt_num):
        # Most are attributes and methods of base class, and refer to the corresponding base class API for details
        weights = self.weights
        loss = self.network(x, img_shape, gt_bboxe, gt_label, gt_num)
        scaling_sens = self.scale_sense
        # Start floating point overflow detection. Create and clear overflow detection status
        status, scaling_sens = self.start_overflow_check(loss, scaling_sens)
        # Multiply the gradient by a scale to prevent the gradient from overflowing
        scaling_sens_filled = ops.ones_like(loss) * ops.cast(scaling_sens, ops.dtype(loss))
        grads = self.grad(self.network, weights)(x, img_shape, gt_bboxe, gt_label, gt_num, scaling_sens_filled)
        # Calculate the true gradient value by dividing the obtained gradient by the scale
        grads = self.hyper_map(ops.partial(_grad_scale, scaling_sens), grads)
        # Gradient clipping
        if self.grad_clip:
            grads = ops.clip_by_global_norm(grads)
        # Gradient aggregation
        grads = self.grad_reducer(grads)

        # Get floating point overflow status
        cond = self.get_overflow_status(status, grads)
        # Calculate loss scaling factor based on overflow state during dynamic loss scale
        overflow = self.process_loss_scale(cond)
        # If there is no overflow, execute the optimizer to update the parameters
        if not overflow:
            self.optimizer(grads)
        return loss, cond, scaling_sens
```

#### Gradient Accumulation

Gradient accumulation is a method that data samples of the training neural network are split into several small Batch  by Batch, and are calculated in order to solve the problem that Batch size is too large due to lack of memory, or the OOM, that is, the neural network can not be trained or the network model is too large to load.

Please refer to [Gradient Accumulation](https://mindspore.cn/tutorials/experts/en/master/others/gradient_accumulation.html) for details.
