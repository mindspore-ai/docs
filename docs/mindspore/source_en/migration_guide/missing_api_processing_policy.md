# Missing API Processing Policy

[![View Source On Gitee](https://mindspore-website.obs.cn-north-4.myhuaweicloud.com/website-images/r2.3/resource/_static/logo_source_en.svg)](https://gitee.com/mindspore/docs/blob/r2.3/docs/mindspore/source_en/migration_guide/missing_api_processing_policy.md)

You can use the following methods to process the missing API:

## 1. Use equivalent replacement

In some scenarios, API functions can be equivalently replaced. For example:

- As Squeeze, Flatten, and ExpandDims do not perform actual calculation, APIs with only Tensor shape changed can be replaced by Reshape.

- When the output shape of AdaptiveAvgPool and AdaptiveMaxPool is 1, AdaptiveAvgPool and AdaptiveMaxPool are equivalent to ReduceMean and ReduceMax when `keep_dims` is set to `True`.

- MaxPool and MaxPoolWithArgmax are equivalent when indices are not used.

- Sort is equivalent to TopK in the full sorting scenario.

## 2. Use existing APIs to package equivalent function logic

For some missing APIs, equivalent functions can be implemented based on existing MindSpore APIs. The following is an example of `sigmoid focal loss`:

First, let's analyze the algorithm basis of the API.

Focal Loss[1] is a method used to deal with the imbalance of positive and negative references and difficult references during the training of a single-phase target detector.

Generally, the sigmoid focal loss API is implemented by MMDetection. The following shows how PyTorch implements this API.

According to the API mapping table, the APIs used in the code have corresponding implementations on MindSpore.

Implement the MindSpore version by referring to the preceding PyTorch code.

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

Then, perform a test.

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

The final error is less than 1e-5, which is a reasonable accuracy error.

```text
6.891787e-08
1.4305115e-06
2.8014183e-06
3.799796e-07
```

## 3. Customize operators

When existing APIs cannot be used for packaging, or the performance of cell encapsulation is poor, you need to customize operators. For details, see [Custom Operators](https://www.mindspore.cn/tutorials/experts/en/r2.3/operation/op_custom.html).

In addition to migrating APIs, you can also use the `aot` development mode of the `Custom` operator to call the PyTorch Aten operator for quick verification. For details, see [Using Third-party Operator Libraries Based on Customized Interfaces](https://www.mindspore.cn/docs/en/r2.3/migration_guide/use_third_party_op.html).

**Note that it is convenient to migrate operators implemented by PyTorch to the GPU and CPU. Most of the operators displayed here are GPU and CPU operators. Ascend operators need to use the TBE for operator development, which has high requirements. Therefore, you are advised to use officially implemented operators for packaging.**

## 4. Seek help from the community

Commit an issue on [MindSpore Gitee](https://gitee.com/mindspore/mindspore/issues) to suggest developing missing APIs.

[1] Lin, T. Y. , et al. "Focal Loss for Dense Object Detection." IEEE Transactions on Pattern Analysis & Machine Intelligence PP.99(2017):2999-3007.
