# 训练网络与梯度求导

<a href="https://gitee.com/mindspore/docs/blob/master/docs/mindspore/source_zh_cn/migration_guide/model_development/training_and_gradient.md" target="_blank"><img src="https://mindspore-website.obs.cn-north-4.myhuaweicloud.com/website-images/master/resource/_static/logo_source.png"></a>

## 自动微分

正向网络构建完成之后，MindSpore提供了[自动微分](https://mindspore.cn/tutorials/zh-CN/master/beginner/autograd.html)的接口用以计算模型的梯度结果。
在[自动求导](https://mindspore.cn/tutorials/zh-CN/master/advanced/network/derivation.html)的教程中，对各种梯度计算的场景做了一些介绍。

## 训练网络

整个训练网络包含正向网络（网络和loss函数），自动梯度求导和优化器更新。MindSpore提供了三种方式来实现这个过程。

1. 封装`Model`，使用`model.train`或者'model.fit'方法执行网络训练，如[模型训练](https://mindspore.cn/tutorials/zh-CN/master/beginner/train.html)。

2. 使用MindSpore封装好的[TrainOneStepCell](https://www.mindspore.cn/docs/zh-CN/master/api_python/nn/mindspore.nn.TrainOneStepCell.html) 和 [TrainOneStepWithLossScaleCell](https://www.mindspore.cn/docs/zh-CN/master/api_python/nn/mindspore.nn.TrainOneStepWithLossScaleCell.html) 分别用于普通的训练流程和带[loss_scale](https://mindspore.cn/tutorials/experts/zh-CN/master/others/mixed_precision.html)的训练流程。如[进阶案例：线性拟合](https://mindspore.cn/tutorials/zh-CN/master/advanced/linear_fitting.html)。

3. 自定义训练Cell。

### 自定义训练Cell

前两个方法举了官网的两个例子说明，对于自定义训练Cell，我们先复习下TrainOneStepCell里面做了什么：

```python
import mindspore as ms
from mindspore import ops, nn
from mindspore.parallel._utils import (_get_device_num, _get_gradients_mean,
                                       _get_parallel_mode)
class TrainOneStepCell(nn.Cell):
    def __init__(self, network, optimizer):
        super(TrainOneStepCell, self).__init__(auto_prefix=False)
        self.network = network  # 带loss的网络结构
        self.network.set_grad()   # PYNATIVE模式时需要，如果为True，则在执行正向网络时，将生成需要计算梯度的反向网络。
        self.optimizer = optimizer   # 优化器，用于参数更新
        self.weights = self.optimizer.parameters    # 获取优化器的参数
        self.grad = ops.GradOperation(get_by_list=True)   # 获取所有输入和参数的梯度

        # 并行计算相关逻辑
        self.reducer_flag = False
        self.grad_reducer = ops.identity
        self.parallel_mode = _get_parallel_mode()
        self.reducer_flag = self.parallel_mode in (ms.ParallelMode.DATA_PARALLEL, ms.ParallelMode.HYBRID_PARALLEL)
        if self.reducer_flag:
            self.mean = _get_gradients_mean()
            self.degree = _get_device_num()
            self.grad_reducer = nn.DistributedGradReducer(self.weights, self.mean, self.degree)

    def construct(self, *inputs):
        loss = self.network(*inputs)    # 运行正向网络，获取loss
        grads = self.grad(self.network, self.weights)(*inputs) # 获得所有Parameter自由变量的梯度
        # grads = grad_op(grads)    # 可以在这里加对梯度的一些计算逻辑，如梯度裁剪
        grads = self.grad_reducer(grads)  # 梯度聚合
        self.optimizer(grads)    # 优化器更新参数
        return loss
```

整个训练流程其实可以包装成一个Cell，在这个Cell里实现网络的正向计算，反向梯度求导和参数更新整个训练的流程，其中我们可以在获取到梯度之后，对梯度做一些特别的处理。

#### 梯度裁剪

当训练过程中遇到梯度爆炸或者梯度特别大训练不稳定的情况，可以考虑添加梯度裁剪，这里对常用的使用global_norm进行梯度裁剪的场景举例说明：

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
        # 大都数是基类的属性和方法，详情参考对应基类API
        weights = self.weights
        loss = self.network(x, img_shape, gt_bboxe, gt_label, gt_num)
        scaling_sens = self.scale_sense
        # 启动浮点溢出检测。创建并清除溢出检测状态
        status, scaling_sens = self.start_overflow_check(loss, scaling_sens)
        # 给梯度乘一个scale防止梯度下溢
        scaling_sens_filled = ops.ones_like(loss) * ops.cast(scaling_sens, ops.dtype(loss))
        grads = self.grad(self.network, weights)(x, img_shape, gt_bboxe, gt_label, gt_num, scaling_sens_filled)
        # 给求得的梯度除回scale计算真实的梯度值
        grads = self.hyper_map(ops.partial(_grad_scale, scaling_sens), grads)
        # 梯度裁剪
        if self.grad_clip:
            grads = ops.clip_by_global_norm(grads)
        # 梯度聚合
        grads = self.grad_reducer(grads)

        # 获取浮点溢出状态
        cond = self.get_overflow_status(status, grads)
        # 动态loss scale时根据溢出状态计算损失缩放系数
        overflow = self.process_loss_scale(cond)
        # 如果没有溢出，执行优化器更新参数
        if not overflow:
            self.optimizer(grads)
        return loss, cond, scaling_sens
```

#### 梯度累积

梯度累积是一种训练神经网络的数据样本按Batch拆分为几个小Batch的方式，然后按顺序计算，用以解决由于内存不足，导致Batch size过大神经网络无法训练或者网络模型过大无法加载的OOM（Out Of Memory）问题。

详情请参考[梯度累积](https://mindspore.cn/tutorials/experts/zh-CN/master/others/gradient_accumulation.html)。
