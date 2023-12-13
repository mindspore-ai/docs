# 缺失API处理策略

[![查看源文件](https://mindspore-website.obs.cn-north-4.myhuaweicloud.com/website-images/r2.3/resource/_static/logo_source.svg)](https://gitee.com/mindspore/docs/blob/r2.3/docs/mindspore/source_zh_cn/migration_guide/missing_api_processing_policy.md)

有以下方法来处理缺失API的情况。

## 1. 等价替换

在有些场景下API的功能是可以等价替换的，比如：

- Squeeze，Flatten，ExpandDims等没有实际的计算，只是改变Tensor shape的API均可以用Reshape代替；

- AdaptiveAvgPool，AdaptiveMaxPool在输出的shape是1时，与ReduceMean，ReduceMax在设置keep_dims=True时是等价的；

- MaxPool和MaxPoolWithArgmax在不使用indices的情况是等价的；

- Sort和在全排序场景下的TopK是等价的。

## 2. 使用已有API包装等价功能逻辑

对于一些缺失的API，可以基于MindSpore已有的API实现等价功能。下面举一个`sigmoid focal loss`的例子：

先来分析一下这个API的算法基础。

Focal Loss[1]是一种用来处理单阶段目标检测器训练过程中出现的正负样本、难易样本不平衡问题的方法。

常用的sigmoid focal loss的API接口是MMDetection的实现，PyTorch实现代码参考下方左侧。

参考API映射表，可以看到PyTorch代码中使用的API在MindSpore上都有对应实现，没有缺失。

根据PyTorch的实现，MindSpore的版本参考下方右侧。

<table class="colwidths-auto docutils align-default">
<tr>
<td style="text-align:center"> PyTorch </td> <td style="text-align:center"> MindSpore </td>
</tr>
<tr>
<td style="vertical-align:top"><pre>

```python
import torch.nn.functional as F

def reduce_loss(loss, reduction):
    """Reduce loss as specified.
    Args:
        loss (Tensor): Elementwise loss tensor.
        reduction (str): Options are "none", "mean" and "sum".
    Return:
        Tensor: Reduced loss tensor.
    """
    reduction_enum = F._Reduction.get_enum(reduction)
    # none: 0, elementwise_mean:1, sum: 2
    if reduction_enum == 0:
        return loss
    elif reduction_enum == 1:
        return loss.mean()
    elif reduction_enum == 2:
        return loss.sum()

def weight_reduce_loss(loss, weight=None, reduction='mean', avg_factor=None):
    if weight is not None:
        loss = loss * weight
    # if avg_factor is not specified, just reduce the loss
    if avg_factor is None:
        loss = reduce_loss(loss, reduction)
    else:
        # if reduction is mean, then average the loss by avg_factor
        if reduction == 'mean':
            loss = loss.sum() / avg_factor
        # if reduction is 'none', then do nothing, otherwise raise an error
        elif reduction != 'none':
            raise ValueError('avg_factor can not be used with reduction="sum"')
    return loss

def py_sigmoid_focal_loss(pred, target, weight=None, gamma=2.0, alpha=0.25,
                          reduction='mean', avg_factor=None):
    """PyTorch version of `Focal Loss <https://arxiv.org/abs/1708.02002>`_.
    Args:
        pred (torch.Tensor): The prediction with shape (N, C), C is the
            number of classes
        target (torch.Tensor): The learning label of the prediction.
        weight (torch.Tensor, optional): Sample-wise loss weight.
        gamma (float, optional): The gamma for calculating the modulating
            factor. Defaults to 2.0.
        alpha (float, optional): A balanced form for Focal Loss.
            Defaults to 0.25.
        reduction (str, optional): The method used to reduce the loss into
            a scalar. Defaults to 'mean'.
        avg_factor (int, optional): Average factor that is used to average
            the loss. Defaults to None.
    """
    pred_sigmoid = pred.sigmoid()
    target = target.type_as(pred)
    pt = (1 - pred_sigmoid) * target +
          pred_sigmoid * (1 - target)
    focal_weight = (alpha * target +
                    (1 - alpha) *
                    (1 - target)) * pt.pow(gamma)
    loss = F.binary_cross_entropy_with_logits(
        pred, target, reduction='none') * focal_weight
    if weight is not None:
        if weight.shape != loss.shape:
            if weight.size(0) == loss.size(0):
                # For most cases, weight is of shape (num_priors, ),
                #  which means it does not have the second axis num_class
                weight = weight.view(-1, 1)
            else:
                # Sometimes, weight per anchor per class is also needed. e.g.
                #  in FSAF. But it may be flattened of shape
                #  (num_priors x num_class, ), while loss is still of shape
                #  (num_priors, num_class).
                assert weight.numel() == loss.numel()
                weight = weight.view(loss.size(0), -1)
        assert weight.ndim == loss.ndim
    loss = weight_reduce_loss(loss, weight, reduction, avg_factor)
    return loss
```

</pre>
</td>
<td style="vertical-align:top"><pre>

```python
import mindspore as ms
from mindspore import nn, ops

class SigmoidFoaclLoss(nn.Cell):
    def __init__(self, weight=None, gamma=2.0, alpha=0.25, reduction='mean', avg_factor=None):
        super(SigmoidFoaclLoss, self).__init__()
        self.sigmoid = ops.Sigmoid()
        self.alpha = alpha
        self.gamma = gamma
        self.weight = ms.Tensor(weight) if weight is not None else weight
        self.reduction = reduction
        self.avg_factor = avg_factor
        self.binary_cross_entropy_with_logits = nn.BCEWithLogitsLoss(reduction="none")
        self.is_weight = (weight is not None)

    def reduce_loss(self, loss):
        """Reduce loss as specified.
        Args:
            loss (Tensor): Elementwise loss tensor.
        Return:
            Tensor: Reduced loss tensor.
        """
        if self.reduction == "mean":
            return loss.mean()
        elif self.reduction == "sum":
            return loss.sum()
        return loss

    def weight_reduce_loss(self, loss):
        # if avg_factor is not specified, just reduce the loss
        if self.avg_factor is None:
            loss = self.reduce_loss(loss)
        else:
            # if reduction is mean, then average the loss by avg_factor
            if self.reduction == 'mean':
                loss = loss.sum() / self.avg_factor
            # if reduction is 'none', then do nothing, otherwise raise an error
            elif self.reduction != 'none':
                raise ValueError('avg_factor can not be used with reduction="sum"')
        return loss

    def construct(self, pred, target):
        pred_sigmoid = self.sigmoid(pred)
        target = ops.cast(target, pred.dtype)

        pt = (1 - pred_sigmoid) * target +
              pred_sigmoid * (1 - target)
        focal_weight = (self.alpha * target +
                        (1 - self.alpha) *
                        (1 - target)) * ops.pow(pt, self.gamma)
        loss = self.binary_cross_entropy_with_logits(pred, target) * focal_weight

        if self.is_weight:
            weight = self.weight
            if self.weight.shape != loss.shape:
                if self.weight.shape[0] == loss.shape[0]:
                    # For most cases, weight is of shape (num_priors, ),
                    # which means it does not have the second axis num_class
                    weight = self.weight.view(-1, 1)
                elif self.weight.size == loss.size:
                    # Sometimes, weight per anchor per class is also needed.
                    # e.g. in FSAF. But it may be flattened of shape
                    # (num_priors x num_class, ), while loss is still of shape
                    # (num_priors, num_class).
                    weight = self.weight.view(loss.shape[0], -1)
                elif self.weight.ndim != loss.ndim:
                    raise ValueError(f"weight shape {self.weight.shape} is not match to loss shape {loss.shape}")
            loss = loss * weight
        loss = self.weight_reduce_loss(loss)
        return loss
```

</pre>
</td>
</tr>
</table>

然后我们做个测试：

```python
import torch
import numpy as np
np.random.seed(1)

def test_compare(pred, target, weight, gamma=2.0, alpha=0.25, reduction='mean', avg_factor=None):
    ms_s_focal_loss = SigmoidFoaclLoss(weight=weight, gamma=gamma, alpha=alpha,
                                       reduction=reduction, avg_factor=avg_factor)
    loss_ms = ms_s_focal_loss(ms.Tensor(pred), ms.Tensor(target))
    loss_pt = py_sigmoid_focal_loss(torch.from_numpy(pred), torch.from_numpy(target), weight=torch.from_numpy(weight),
                                    gamma=gamma, alpha=alpha, reduction=reduction, avg_factor=avg_factor)
    print(np.max(np.abs(loss_ms.asnumpy() - loss_pt.numpy())))

pred = np.random.uniform(-1, 1, (3, 4)).astype(np.float32)
target = np.random.uniform(-1, 1, (3, 4)).astype(np.float32)
weight = np.random.uniform(0, 1, (3,)).astype(np.float32)

test_compare(pred, target, weight, gamma=2.0, alpha=0.25, reduction='mean', avg_factor=None)
test_compare(pred, target, weight, gamma=1.0, alpha=0.5, reduction='sum', avg_factor=None)
test_compare(pred, target, weight, gamma=2.0, alpha=0.25, reduction='mean', avg_factor=0.3)
test_compare(pred, target, weight, gamma=2.0, alpha=0.25, reduction='none', avg_factor=None)
```

可以看到最后的误差在1e-5以下，是合理的精度误差：

```text
6.891787e-08
1.4305115e-06
2.8014183e-06
3.799796e-07
```

## 3. 自定义算子

当有些情况无法使用已有的API进行包装，或者用Cell封装的方式性能非常差，这个时候就需要使用自定义算子，详情请参考Custom算子的[使用指南](https://www.mindspore.cn/tutorials/experts/zh-CN/r2.3/operation/op_custom.html)。

除了可以自己迁移实现API，也可以利用`Custom`算子的`aot`开发方式调用PyTorch Aten的算子进行快速验证，请参考[基于自定义算子接口调用第三方算子库](https://www.mindspore.cn/docs/zh-CN/r2.3/migration_guide/use_third_party_op.html)。

**注意，PyTorch实现的算子迁移到GPU和CPU上比较方便，这里展示的也大多是GPU和CPU的，Ascend的算子由于需要使用TBE进行算子开发，门槛较高，推荐使用官方实现的算子进行包装。**

## 4. 社区求助

在MindSpore的[Gitee](https://gitee.com/mindspore/mindspore/issues)上提交issue建议开发缺失API。

[1] Lin, T. Y. , et al. "Focal Loss for Dense Object Detection." IEEE Transactions on Pattern Analysis & Machine Intelligence PP.99(2017):2999-3007.
