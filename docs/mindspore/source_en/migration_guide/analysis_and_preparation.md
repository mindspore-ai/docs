# Model Analysis and Preparation

<a href="https://gitee.com/mindspore/docs/blob/master/docs/mindspore/source_en/migration_guide/analysis_and_preparation.md" target="_blank"><img src="https://mindspore-website.obs.cn-north-4.myhuaweicloud.com/website-images/master/resource/_static/logo_source_en.png"></a>

## Obtaining Sample Code

When you obtain a paper to implement migration on MindSpore, you need to find the reference code that has been implemented in other frameworks. In principle, the reference code must meet at least one of the following requirements:

1. The author opens the paper to the public.
2. The implementation is starred and forked by many developers, which means it is widely recognized.
3. The code is new and maintained by developers.
4. The PyTorch reference code is preferred.

If the results are not reproducible in the reference project or the version information is missing, check the project issue for information.

If a new paper has no reference implementation, you can refer to [Constructing MindSpore Network](https://www.mindspore.cn/docs/en/master/migration_guide/model_development/model_development.html).

## Analyzing Algorithm and Network Structure

First, when reading the paper and reference code, you need to analyze the network structure to organize the code writing. The following shows the general network structure of YOLOX.

| Module| Implementation|
| ---- | ---- |
| backbone | CSPDarknet (s, m, l, x)|
| neck | FPN |
| head | Decoupled Head |

Second, analyze the innovative points of the migration algorithm and record the tricks used during the training, for example, data augmentation added during data processing, shuffle, optimizer, learning rate attenuation policy, and parameter initialization. You can prepare a checklist and fill in the corresponding items during analysis.

For example, the following records some tricks used by the YOLOX network during training.

<table>
    <tr>
        <th>Trick</th>
        <th>Record</th>
   </tr>
    <tr>
        <td rowspan="2">Data augmentation</td>
        <td >Mosaic, including random scaling, crop, and layout</td>
    </tr>
    <tr>
        <td >MixUp</td>
    </tr>
    <tr>
        <td >Learning rate attenuation policy</td>
        <td >Multiple attenuation modes are available. By default, the COS learning rate attenuation is used. </td>
    </tr>
    <tr>
        <td >Optimizer parameters</td>
        <td >SGD momentum=0.9, nesterov=True, and no weight decay</td>
    </tr>
    <tr>
        <td >Training parameters</td>
        <td >epoch: 300; batchsize: 8</td>
    </tr>
    <tr>
        <td >Network structure optimization points</td>
        <td >Decoupled Head; Anchor Free; SimOTA</td>
    </tr>
    <tr>
        <td >Training process optimization points </td>
        <td >EMA; Data augmentation is not performed for the last 15 epochs; mixed precision </td>
    </tr>
</table>

**Note that the tricks used in the code are mainly reproduced. The tricks mentioned in some papers may not be useful.**

In addition, you need to determine whether the paper can be implemented by modifying the existing MindSpore model. If yes, you can greatly reduce the development workload. For example, WGAN-PG can be developed based on WGAN.
[MindSpore models](https://gitee.com/mindspore/models) is a model repository. It covers mainstream models in multiple fields, such as machine vision, natural language processing, voice, and recommendation system. You can check whether there are required models from the repository.

## Reproducing Paper Implementation

After obtaining the reference code, you need to reproduce the accuracy of the reference implementation and obtain the performance data of the reference implementation. This has the following advantages:

1. Identify some issues in advance.

    - Check whether the third-party repository used by the reference code depends on a version to identify version adaptation problems in advance.
    - Check whether the dataset can be obtained. Some datasets are private or the author adds some datasets to the public dataset. This problem can be found at the reproduction reference implementation stage.
    - Check whether the reference implementation can reproduce the accuracy of the paper. Some official reference implementations may not reproduce the accuracy of the paper. In this case, detect the problem in time, replace the reference implementation, or adjust the accuracy baseline.

3. Obtain some reference data for the MindSpore migration process.

    - Obtain the loss decrease trend to check whether the training convergence trend on MindSpore is normal.
    - Obtain the parameter file for conversion and inference verification. For details, see [Inference and Training Process](https://www.mindspore.cn/docs/en/master/migration_guide/model_development/training_and_evaluation_procession.html).
    - Obtain the performance baseline for performance tuning. For details, see [Debugging and Tuning](https://www.mindspore.cn/docs/en/master/migration_guide/debug_and_tune.html).

## Analyzing API Compliance

The API missing analysis here refers to APIs in the network execution diagram, including MindSpore [operators](https://www.mindspore.cn/docs/en/master/api_python/mindspore.ops.html) and advanced encapsulated APIs, and excluding the APIs used in data processing. You are advised to use third-party APIs, such as NumPy, OpenCV, Pandas, and PIL, to replace APIs used in data processing.

### Querying the API Mapping Table

Take the PyTorch code migration as an example. After obtaining the reference code implementation, you can filter keywords such as `torch`, `nn`, and `ops` to obtain the used APIs. If the method of another repository is invoked, you need to manually analyze the API. Then, check the [PyTorch and MindSpore API Mapping Table](https://www.mindspore.cn/docs/en/master/note/api_mapping/pytorch_api_mapping.html).
Alternatively, the [API](https://www.mindspore.cn/docs/en/master/api_python/mindspore.ops.html) searches for the corresponding API implementation.

Generally the training process of a network contains forward calculation, backward gradient calculation and parameter update. In some special scenarios, another gradient calculation is needed for the gradient, such as [Gradient Penalty](https://arxiv.org/pdf/1704.00028.pdf), and this kind of scenario uses the second order gradient calculation. For scenarios where second-order gradient calculations are used in the network requires additional analysis of the second-order support of the APIs, the derivative links of the network need to be analyzed by code walk-through, and all APIs within the second-order derivative links need to support second order. The second-order support case can be viewed in [MindSpore gradient section source code](https://gitee.com/mindspore/mindspore/tree/master/mindspore/python/mindspore/ops/_grad) to see if its first-order Grad has a corresponding of the bprop function definition.

For example, if the network second-order derivative links contain StridedSlice slicing operation, you can look up [array_ops gradient definition file](https://gitee.com/mindspore/mindspore/blob/master/mindspore/python/mindspore/ops/_grad/grad_array_ops.py) in the [reverse registration code of StridedSliceGrad](https://gitee.com/mindspore/mindspore/blob/master/mindspore/python/mindspore/ops/_grad/grad_array_ops.py#L867). If it exists, the current version of MindSpore StridedSlice slicing operation supports second-order gradient calculation.

For details about the mapping of other framework APIs, see the [API naming and function description](https://www.mindspore.cn/docs/en/master/api_python/mindspore.html). For APIs with the same function, the names of MindSpore may be different from those of other frameworks. The parameters and functions of APIs with the same name may also be different from those of other frameworks. For details, see the official description.

If the corresponding API is not found, see specific missing API processing policy.

### Missing API Processing Policy

You can use the following methods to process the missing API:

#### 1. Use equivalent replacement

In some scenarios, API functions can be equivalently replaced. For example:

- As Squeeze, Flatten, and ExpandDims do not perform actual calculation, APIs with only Tensor shape changed can be replaced by Reshape.

- When the output shape of AdaptiveAvgPool and AdaptiveMaxPool is 1, AdaptiveAvgPool and AdaptiveMaxPool are equivalent to ReduceMean and ReduceMax when `keep_dims` is set to `True`.

- MaxPool and MaxPoolWithArgmax are equivalent when indices are not used.

- Sort is equivalent to TopK in the full sorting scenario.

#### 2. Use existing APIs to package equivalent function logic

For some missing APIs, equivalent functions can be implemented based on existing MindSpore APIs. The following is an example of `sigmoid focal loss`:

First, let's analyze the algorithm basis of the API.

Focal Loss[1] is a method used to deal with the imbalance of positive and negative references and difficult references during the training of a single-phase target detector.

Generally, the sigmoid focal loss API is implemented by MMDetection. The following shows how PyTorch implements this API.

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


def py_sigmoid_focal_loss(pred,
                          target,
                          weight=None,
                          gamma=2.0,
                          alpha=0.25,
                          reduction='mean',
                          avg_factor=None):
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
    pt = (1 - pred_sigmoid) * target + pred_sigmoid * (1 - target)
    focal_weight = (alpha * target + (1 - alpha) *
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

According to the API mapping table, the APIs used in the code have corresponding implementations on MindSpore.

Implement the MindSpore version by referring to the preceding PyTorch code.

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
        pt = (1 - pred_sigmoid) * target + pred_sigmoid * (1 - target)
        focal_weight = (self.alpha * target + (1 - self.alpha) * (1 - target)) * ops.pow(pt, self.gamma)
        loss = self.binary_cross_entropy_with_logits(pred, target) * focal_weight
        if self.is_weight:
            weight = self.weight
            if self.weight.shape != loss.shape:
                if self.weight.shape[0] == loss.shape[0]:
                    # For most cases, weight is of shape (num_priors, ),
                    #  which means it does not have the second axis num_class
                    weight = self.weight.view(-1, 1)
                elif self.weight.size == loss.size:
                    # Sometimes, weight per anchor per class is also needed. e.g.
                    #  in FSAF. But it may be flattened of shape
                    #  (num_priors x num_class, ), while loss is still of shape
                    #  (num_priors, num_class).
                    weight = self.weight.view(loss.shape[0], -1)
                elif self.weight.ndim != loss.ndim:
                    raise ValueError(f"weight shape {self.weight.shape} is not match to loss shape {loss.shape}")
            loss = loss * weight
        loss = self.weight_reduce_loss(loss)
        return loss
```

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

#### 3. Customize operators

When existing APIs cannot be used for packaging, or the performance of cell encapsulation is poor, you need to customize operators. For details, see [Custom Operators](https://www.mindspore.cn/tutorials/experts/en/master/operation/op_custom.html).

In addition to migrating APIs, you can also use the `aot` development mode of the `Custom` operator to call the PyTorch Aten operator for quick verification. For details, see [Using Third-party Operator Libraries Based on Customized Interfaces](https://www.mindspore.cn/docs/en/master/migration_guide/use_third_party_op.html).

**Note that it is convenient to migrate operators implemented by PyTorch to the GPU and CPU. Most of the operators displayed here are GPU and CPU operators. Ascend operators need to use the TBE for operator development, which has high requirements. Therefore, you are advised to use officially implemented operators for packaging.**

#### 4. Seek help from the community

Commit an issue on [MindSpore Gitee](https://gitee.com/mindspore/mindspore/issues) to suggest developing missing APIs.

## Analyzing Function Compliance

During continuous delivery of MindSpore, some functions are restricted. If restricted functions are involved during network migration, some measures can be taken to avoid the impact of function restrictions.

### Dynamic shape

To know dynamic shape, you need to know what is a static shape.
Static shape indicates that the shape of a tensor does not change during network execution.
For example, on the ResNet50 network, if the input shape of an image is always `224*224`, the shapes of the output Tesnor of the four residual modules are `B*64*56*56`, `B*128*28*28`, `B*256*14*14`, and `B*512*7*7` respectively in the network training phase. `B` indicates `BatchSize`, which is also fixed during the training. In this case, all shapes on the network are static and no dynamic shape is available.
If the input shape may no+t be `224*224`, the shape of the output tensor of the four residual modules varies with the input shape. In this case, the shape is dynamic instead of static. Generally, dynamic shape is introduced due to the following reasons:

#### Input shape not fixed

For example, the input image has different shapes, and the audio label has different lengths. In this case, dynamic shapes are introduced.

In this scenario, you can read the code to check whether the output shape of data processing is fixed, or directly print the output shape of data processing for comparison.

```python
for batch_idx, (data, target) in enumerate(data_loader):
    print(batch_idx, data.shape, target.shape)
    print("="*20)
```

#### APIs that cause shape changes during network execution

During network execution, some operations may cause tensor shape changes.

The common APIs that cause this scenario are as follows:

| API | Description| Dynamic Shape Scenario|
| ---- | ----- | ------- |
| StridedSlice/Slice | Specifies a slice. You can also use [start_idx:end_idx] during programming.| The slice subscript is a variable.|
| TopK | Obtains the first K data.| The value of K is not fixed.|
| Gather | Obtains the slice consisting of the elements corresponding to the tensor index on the specified axis.| The index length is not fixed.|
| UnsortedSegmentX | Specifies computation of an input tensor, including UnsortedSegmentSum and UnsortedSegmentMax.| The segment is not fixed.|
| Sampler | Specifies sampler-related operations, such as where and random.choice.| The sample quantity is not fixed.|
| ReduceX | Specifies a reduction operation, such as ReduceSum and ReduceMean.| The axis is not fixed.|
| Transpose | Performs transformation based on the axis.| The axis is not fixed.|
| Unique | Deduplicates data.| Dynamic shape is introduced when this API is used.|
| MaskedSelect | Obtains the value of mask based on the Boolean type.| Dynamic shape is introduced when this API is used.|

For example:

```python
import numpy as np
import mindspore as ms
np.random.seed(1)
x = ms.Tensor(np.random.uniform(0, 1, (10)).astype(np.float32))
k = ms.Tensor(np.random.randint(1, 10), ms.int64)
print(k)
print(x[:k].shape)
# 6
# (6,)
```

During network training, there is a slicing operation `x[:k]`. Here, k is not a constant. As a result, the shape of `x[:k]` changes with the value of k, and the shape of all subsequent operations related to `x[:k]` is uncertain.

#### Shape changes introduced by different branches of control flows

The output of some control flows on the network may be different. When the condition control items of the control flows are not fixed, dynamic shape may be triggered. For example:

```python
import numpy as np
import mindspore as ms
from mindspore import ops
np.random.seed(1)
x = ms.Tensor(np.random.uniform(0, 1, (10)).astype(np.float32))
cond = (x > 0.5).any()

if cond:
    y = ops.masked_select(x, x > 0.5)
else:
    y = ops.zeros_like(x)
print(x)
print(cond)
print(y)

# [4.17021990e-01 7.20324516e-01 1.14374816e-04 3.02332580e-01
#  1.46755889e-01 9.23385918e-02 1.86260208e-01 3.45560730e-01
#  3.96767467e-01 5.38816750e-01]
# True
# [0.7203245  0.53881675]
```

In this process, there are two dynamic shapes. One is that the shape of the `masked_select` result is dynamic if `cond=True`. The other is the control flow. Because `cond` is uncertain, the shape output of the two branches of the control flow is different, which also causes the dynamic shape.

Generally, the dynamic shape can be analyzed at the algorithm and code layers, or the tensor related to the reference code can be directly printed for judgment. If dynamic shape exists, we will introduce the workaround in [Network Body and Loss Setup](https://www.mindspore.cn/docs/en/master/migration_guide/model_development/model_and_loss.html).

#### Sparsity

A [sparse tensor](https://matteding.github.io/2019/04/25/sparse-matrices/) is a special tensor in which the value of the most significant element is zero.

In some scenarios (such as recommendation systems, molecular dynamics, graph neural networks), the data is sparse. If you use common dense tensors to represent the data, you may introduce many unnecessary calculations, storage, and communication costs. In this case, it is better to use sparse tensor to represent the data.

MindSpore now supports the most commonly used [CSR and COO data formats](https://www.mindspore.cn/tutorials/en/master/beginner/tensor.html#sparse-tensor). Currently, only a limited number of sparse operators are supported, and most sparse features are restricted. In this case, you are advised to check whether the corresponding operator supports sparse computing. If the operator does not support sparse computing, convert it into a common operator.
After the operator is converted into a dense operator, the video memory used increases. Therefore, the batch size implemented by referring to may not be used for training. In this case, you can use [Gradient Accumulation](https://www.mindspore.cn/tutorials/experts/en/master/others/gradient_accumulation.html) to simulate large batch training.

## MindSpore Function/Feature Recommendation

### [Dynamic and Static Graphs](https://www.mindspore.cn/tutorials/en/master/advanced/compute_graph.html)

Currently, there are two execution modes of a mainstream deep learning framework: a static graph mode (Graph) and a dynamic graph mode (PyNative).

- In static graph mode, when the program is built and executed, the graph structure of the neural network is generated first, and then the computation operations involved in the graph are performed. Therefore, in static graph mode, the compiler can achieve better execution performance by using technologies such as graph optimization, which facilitates large-scale deployment and cross-platform running.

- In dynamic graph mode, the program is executed line by line according to the code writing sequence. In the forward execution process, the backward execution graph is dynamically generated according to the backward propagation principle. In this mode, the compiler delivers the operators in the neural network to the device one by one for computing, facilitating users to build and debug the neural network model.

### [Calling the Custom Class](https://www.mindspore.cn/tutorials/experts/en/master/network/ms_class.html)

In static graph mode, you can use `ms_class` to modify a custom class. You can create and call an instance of the custom class, and obtain its attributes and methods.

`ms_class` is applied to the static graph mode to expand the support scope of static graph compilation syntax. In dynamic graph mode, that is, PyNative mode, the use of `ms_class` does not affect the execution logic of PyNative mode.

### [Automatic Differential](https://www.mindspore.cn/tutorials/en/master/beginner/autograd.html)

Automatic differentiation can calculate a derivative value of a derivative function at a certain point, which is a generalization of backward propagation algorithms. The main problem solved by automatic differential is to decompose a complex mathematical operation into a series of simple basic operations. This function shields a large number of derivative details and processes from users, greatly reducing the threshold for using the framework.

### [Mixed Precision](https://www.mindspore.cn/tutorials/zh-CN/master/advanced/mixed_precision.html)

Generally, when a neural network model is trained, the default data type is FP32. In recent years, to accelerate training time, reduce memory occupied during network training, and store a trained model with same precision, more and more mixed-precision training methods are proposed in the industry. The mixed-precision training herein means that both single precision (FP32) and half precision (FP16) are used in a training process.

### [Auto Augmentation](https://www.mindspore.cn/tutorials/experts/en/master/dataset/augment.html)

MindSpore not only allows you to customize data augmentation, but also provides an automatic data augmentation mode to automatically perform data augmentation on images based on specific policies.

### [Multi Dimensional](https://www.mindspore.cn/tutorials/experts/en/master/parallel/multi_dimensional.html)

With the development of deep learning, the model scale becomes larger and larger. For example, in the NLP field, the number of parameters has increased from 100 million in BERT to 170 billion in GPT-3, and then to 200 billion in PanGu-Alpha. Currently, the industry even proposes millions of parameters. It can be seen that the parameter scale has an exponential growth trend in recent years. On the other hand, with the development of technologies in fields such as big data and the Internet, datasets that can be used for model training also increase rapidly. For example, the size of datasets in scenarios such as recommendation and natural language processing can reach TB level.

### [Gradient Accumulation Algorithm](https://www.mindspore.cn/tutorials/experts/en/master/others/gradient_accumulation.html)

Gradient accumulation is a method of splitting data samples for training neural networks into several small batches by batch and then calculating the batches in sequence. The purpose is to solve the out of memory (OOM) problem that the neural network cannot be trained or the network model cannot be loaded due to insufficient memory.

### [Summary](https://www.mindspore.cn/mindinsight/docs/en/master/summary_record.html)

Scalars, images, computational graphs, training optimization processes, and model hyperparameters during training are recorded in files and can be viewed on the web page.

### [Debugger](https://www.mindspore.cn/mindinsight/docs/en/master/debugger.html)

The MindSpore debugger is a debugging tool provided for graph mode training. It can be used to view and analyze the intermediate results of graph nodes.

### [Golden Stick](https://www.mindspore.cn/golden_stick/docs/en/master/index.html)

MindSpore Golden Stick is a model compression algorithm set jointly designed and developed by Huawei Noah's team and Huawei MindSpore team. It contains basic quantization and pruning methods.

## Differences Between MindSpore and PyTorch APIs

When migrating the network from PyTorch to MindSpore, pay attention to the differences between MindSpore and [typical PyTorch APIs](https://www.mindspore.cn/docs/en/master/migration_guide/typical_api_comparision.html).

[1] Lin, T. Y. , et al. "Focal Loss for Dense Object Detection." IEEE Transactions on Pattern Analysis & Machine Intelligence PP.99(2017):2999-3007.
