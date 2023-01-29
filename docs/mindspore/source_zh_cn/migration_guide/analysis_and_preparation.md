# 模型分析与准备

<a href="https://gitee.com/mindspore/docs/blob/r1.10/docs/mindspore/source_zh_cn/migration_guide/analysis_and_preparation.md" target="_blank"><img src="https://mindspore-website.obs.cn-north-4.myhuaweicloud.com/website-images/r1.10/resource/_static/logo_source.png"></a>

## 获取参考代码

我们拿到一篇论文，需要在MindSpore上进行迁移实现时，优先需要找到在其他框架已经实现好的参考代码，原则上这个参考代码需要符合以下要求中的至少一项：

1. 论文原作者开源的实现；
2. 大众普遍认可的实现(star数，fork数较多)；
3. 比较新的代码，有开发者对代码进行维护；
4. 优先考虑PyTorch的参考代码。

如果是全新的论文，无可参考实现，请参考[MindSpore网络搭建](https://www.mindspore.cn/docs/zh-CN/r1.10/migration_guide/model_development/model_development.html)进行开发。

## 分析算法及网络结构

在阅读论文及参考代码时，首先需要分析网络结构，用以组织代码编写。比如下面是YOLOX的大致网络结构：

| 模块 | 实现 |
| ---- | ---- |
| backbone | CSPDarknet(s,m,l,x等) |
| neck | FPN |
| head | Decoupled Head |

其次需要分析迁移算法的创新点，记录在训练过程中使用了哪些trick，如数据处理加了哪些数据增强，是否有shuffle，使用了什么优化器，学习率衰减策略，参数初始化方式等。可以整理一个checklist，在分析过程中可以填写相应项来记录。

比如这里记录了YOLOX网络在训练时使用的一些trick：

<table>
    <tr>
        <th>trick</th>
        <th>记录</th>
   </tr>
    <tr>
        <td rowspan="2">数据增强</td>
        <td >mosaic，包含随机缩放，随机剪裁，随机排布 </td>
    </tr>
    <tr>
        <td >MixUp</td>
    </tr>
    <tr>
        <td >学习率衰减策略</td>
        <td >多种衰减方式供选择，默认使用cos学习率衰减</td>
    </tr>
    <tr>
        <td >优化器参数</td>
        <td >带动量SGD momentum=0.9，nesterov=True，无weight decay</td>
    </tr>
    <tr>
        <td >训练参数</td>
        <td >epoch：300；batchsize：8</td>
    </tr>
    <tr>
        <td >网络结构优化点</td>
        <td >Decoupled Head；Anchor Free；SimOTA</td>
    </tr>
    <tr>
        <td >训练流程优化点</td>
        <td >EMA；后15epoch不做数据增强；混合精度</td>
    </tr>
</table>

**注意，以复现代码中使用的trick为主，有些论文里提到的不一定有用。**

此外，需要判断论文是否能通过在MindSpore已有模型上做少量修改来实现，若是，可以在已有模型的基础上进行开发，这样能极大的减少开发的工作量。比如WGAN-PG可以基于WGAN进行开发。
[MindSpore models](https://gitee.com/mindspore/models)是MindSpore的模型仓库，当前已经覆盖了机器视觉、自然语言处理、语音、推荐系统等多个领域的主流模型，可以从中查找是否有需要的模型。

## 复现论文实现

获取到参考代码后，需要复现下参考实现的精度，获取参考实现的性能数据。这样做有几点好处：

1. 提前识别一些问题：

    - 判断参考代码使用的三方库是否有版本依赖，提前识别版本适配问题；
    - 判断数据集是否能获取的到，有的数据集是私有的或者原作者在公开数据集上加了自己的部分数据集，在复现参考实现阶段就可以发现这种问题；
    - 参考实现是否能复现论文精度，有的官方的参考实现也不一定能复现论文的精度，当出现这种情况时要及时发现问题，更换参考实现或者调整精度基线。

3. 获取一些参考数据作为MindSpore迁移过程的参考：

    - 获取loss下降趋势，帮助验证MindSpore上训练收敛趋势是否ok；
    - 获取参数文件，用于进行转换，进行推理验证，详细过程参考[推理及训练流程](https://www.mindspore.cn/docs/zh-CN/r1.10/migration_guide/model_development/training_and_evaluation_procession.html)；
    - 获取性能基线，在做性能优化时有一个基础目标，如需做性能优化，请参考[调试调优](https://www.mindspore.cn/docs/zh-CN/r1.10/migration_guide/debug_and_tune.html)。

## 分析API满足度

这里分析的API缺失专指网络执行图中的API，包含MindSpore的[算子](https://www.mindspore.cn/docs/zh-CN/r1.10/api_python/mindspore.ops.html)及高级封装API，不包括数据处理中使用的API。数据处理过程中使用的API建议使用三方的实现代替，如numpy，opencv，pandas，PIL等。

### 查询API映射表

以PyTorch的代码迁移为例，拿到参考代码实现后，可以通过过滤`torch`，`nn`，`ops`等关键字获取使用的API接口，如调用了其他库的方法，需要手动分析。然后对照[PyTorch与MindSpore API 映射](https://www.mindspore.cn/docs/zh-CN/r1.10/note/api_mapping/pytorch_api_mapping.html)
或者[API](https://www.mindspore.cn/docs/zh-CN/r1.10/api_python/mindspore.ops.html) 查找对应的API实现。

其他框架API的映射可以参考API命名与功能描述。注意，针对相同功能的API，MindSpore的命名可能与其他框架不同，同名API参数与功能也可能与其他框架有区别，均以官方描述为准。

如果没有找到对应的API接口，需要用具体的策略来处理API缺失的问题。

### 缺失API处理策略

有以下方法来处理缺失API的情况。

#### 1. 等价替换

在有些场景下API的功能是可以等价替换的，比如：

- Squeeze，Flatten，ExpandDims等没有实际的计算，只是改变Tensor shape的API均可以用Reshape代替；

- AdaptiveAvgPool，AdaptiveMaxPool在输出的shape是1时，与ReduceMean，ReduceMax在设置keep_dims=True时是等价的；

- MaxPool和MaxPoolWithArgmax在不使用indices的情况是等价的；

- Sort和在全排序场景下的TopK是等价的。

#### 2. 使用已有API包装等价功能逻辑

对于一些缺失的API，可以基于MindSpore已有的API实现等价功能。下面举一个`sigmoid focal loss`的例子：

先来分析一下这个API的算法基础。

Focal Loss[1]是一种用来处理单阶段目标检测器训练过程中出现的正负样本、难易样本不平衡问题的方法。

常用的sigmoid focal loss的API接口是MMDetection的实现，我们来看一下使用PyTorch是怎么实现的：

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

参考API映射表，可以看到代码中使用的API MindSpore上都有对应实现，没有缺失。

参考上面的PyTorch代码，实现下MindSpore的版本：

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

#### 3. 自定义算子

当有些情况无法使用已有的API进行包装，或者用Cell封装的方式性能非常差，这个时候就需要使用自定义算子，详情请参考Custom算子的[使用指南](https://www.mindspore.cn/tutorials/experts/zh-CN/r1.10/operation/op_custom.html)。

除了可以自己迁移实现API，也可以利用`Custom`算子的`aot`开发方式调用PyTorch Aten的算子进行快速验证，请参考[基于自定义算子接口调用第三方算子库](https://www.mindspore.cn/docs/zh-CN/r1.10/migration_guide/use_third_party_op.html)。

**注意，PyTorch实现的算子迁移到GPU和CPU上比较方便，这里展示的也大多是GPU和CPU的，Ascend的算子由于需要使用TBE进行算子开发，门槛较高，推荐使用官方实现的算子进行包装。**

#### 4. 社区求助

在MindSpore的[Gitee](https://gitee.com/mindspore/mindspore/issues)上提交issue建议开发缺失API。

## 分析功能满足度

MindSpore在持续交付中，部分功能存在限制，在网络迁移过程中涉及受限功能使用的情况，可采取一些措施来避免功能限制的影响。

### 动态shape

想要了解动态shape，需要先了解什么是静态shape。
静态shape指在网路执行阶段Tensor的shape没有发生变化。
比如resnet50网络如果保证图片的输入shape一直是`224*224`的，那么在网络训练阶段，四个残差模块的输出Tensor的shape分别是`B*64*56*56`，`B*128*28*28`，`B*256*14*14`，`B*512*7*7`，`B`指`BatchSize`，在训练过程中也是固定的，此时网络中全部是静态的shape，没有动态shape。
如果输入的shape不一定是`224*224`的，那么四个残差模块输出Tensor的shape将会随输入shape变化，此时就不是静态shape，而是动态shape了。一般动态shape引入的原因有：

#### 输入shape不固定

比如输入图片需要有不同的shape，音频的label需要不同长度，这都会引入动态shape；

这种场景可以读代码分析数据处理的输出shape是否固定，也可以直接打印数据处理输出的shape，进行对比：

```python
for batch_idx, (data, target) in enumerate(data_loader):
    print(batch_idx, data.shape, target.shape)
    print("="*20)
```

#### 网络执行过程中有引发shape变化的API

在网络执行过程中可能有一些操作会引起Tensor的shape变化。

引起这种场景常见的API有：

| API | 功能描述 | 引发动态shape场景 |
| ---- | ----- | ------- |
| StridedSlice/Slice | 切片，用户编程时也可以使用 [start_idx:end_idx]这种方式 | 当切片下标是变量时 |
| TopK | 取前K大 | 当K取值不定时 |
| Gather | 取Tensor在指定 axis 上索引对应的元素组成的切片 | 当index长度不定时 |
| UnsortedSegmentX | 包含UnsortedSegmentSum，UnsortedSegmentMax等沿分段计算输入Tensor的某个计算 | 当分段不固定时 |
| Sampler | 取样器相关操作，比如where，random.choice等 | 当抽取数量不固定时 |
| ReduceX | ReduceSum，ReduceMean等归约操作 | 当axis不固定时 |
| Transpose | 根据轴进行变换 | 当变化轴不定时 |
| Unique | 去重 | 使用就会引入动态shape |
| MaskedSelect | 根据bool型的mask取值 | 使用就会引入动态shape |
| NonZero | 计算非零元素的下标 | 使用就会引入动态shape |

比如：

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

在网络训练时有个切片的操作`x[:k]`这里的k不是一个常量，会导致`x[:k]`的shape随k的值改变，导致后续所有和`x[:k]`相关的操作的shape不确定。

#### 控制流不同分支引入shape上的变化

网络中可能会有一些控制流的输出是不一样的，而当控制流的条件控制项不是固定的时，可能会引发动态shape，比如：

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

在这个过程其实有两个地方有动态shape，一个是`cond=True`时`masked_select`结果的shape是动态，另外是控制流，由于cond不定，控制流两个分支的shape输出不同也会造成动态shape。

动态shape一般可以从算法、代码层面进行分析，也可以直接打印参考代码相关Tensor进行判断。如果存在动态shape，我们在[网络主体和loss搭建](https://www.mindspore.cn/docs/zh-CN/r1.10/migration_guide/model_development/model_and_loss.html)篇章有规避策略的介绍。

#### 稀疏

[稀疏张量](https://matteding.github.io/2019/04/25/sparse-matrices/) 是一种特殊张量，其中绝大部分元素的值为零。

在某些应用场景中（比如推荐系统、分子动力学、图神经网络等），数据的特征是稀疏的，若使用普通张量表征这些数据会引入大量不必要的计算、存储和通讯开销。在这种时候就可以使用稀疏张量来表征这些数据。

MindSpore现在已经支持最常用的[CSR和COO两种稀疏数据格式](https://www.mindspore.cn/tutorials/zh-CN/r1.10/beginner/tensor.html#%E7%A8%80%E7%96%8F%E5%BC%A0%E9%87%8F)。但是由于目前支持稀疏算子有限，大部分稀疏的特性还存在限制，在此情况下，建议优先查找对应的算子是否支持稀疏计算，如不支持的话需要转换成普通算子。
由于转换成稠密算子后使用的显存会增加，可能不能使用参考实现的batch size进行训练，此时可以使用 [梯度累积](https://www.mindspore.cn/tutorials/experts/zh-CN/r1.10/others/gradient_accumulation.html) 来模拟大batch训练。

## MindSpore好用功能/特性推荐

### [动态图与静态图](https://www.mindspore.cn/tutorials/zh-CN/r1.10/advanced/compute_graph.html)

目前主流的深度学习框架有静态图(Graph)和动态图(PyNative)两种执行模式。

- 静态图模式下，程序在编译执行时，首先生成神经网络的图结构，然后再执行图中涉及的计算操作。因此，在静态图模式下，编译器可以通过使用图优化等技术来获得更好的执行性能，有助于规模部署和跨平台运行。

- 动态图模式下，程序按照代码的编写顺序逐行执行，在执行正向过程中根据反向传播的原理，动态生成反向执行图。这种模式下，编译器将神经网络中的各个算子逐一下发到设备进行计算操作，方便用户编写和调试神经网络模型。

### [调用自定义类](https://www.mindspore.cn/tutorials/experts/zh-CN/r1.10/network/ms_class.html)

在静态图模式下，通过使用ms_class修饰自定义类，用户可以创建、调用该自定义类的实例，并且可以获取其属性和方法。

ms_class应用于静态图模式，扩充完善静态图编译语法的支持范围。在动态图模式即PyNative模式下，ms_class的使用不影响PyNative模式的执行逻辑。

### [自动微分](https://www.mindspore.cn/tutorials/zh-CN/r1.10/beginner/autograd.html)

自动微分能够计算可导函数在某点处的导数值，是反向传播算法的一般化。自动微分主要解决的问题是将一个复杂的数学运算分解为一系列简单的基本运算，该功能对用户屏蔽了大量的求导细节和过程，大大降低了框架的使用门槛。

### [混合精度](https://www.mindspore.cn/tutorials/experts/zh-CN/r1.10/others/mixed_precision.html)

通常我们训练神经网络模型的时候，默认使用的数据类型为单精度FP32。近年来，为了加快训练时间、减少网络训练时候所占用的内存，并且保存训练出来的模型精度持平的条件下，业界提出越来越多的混合精度训练的方法。这里的混合精度训练是指在训练的过程中，同时使用单精度（FP32）和半精度（FP16）。

### [自动数据增强](https://www.mindspore.cn/tutorials/experts/zh-CN/r1.10/dataset/augment.html)

MindSpore除了可以让用户自定义数据增强的使用，还提供了一种自动数据增强方式，可以基于特定策略自动对图像进行数据增强处理。

### [多维度混合并行](https://www.mindspore.cn/tutorials/experts/zh-CN/r1.10/parallel/multi_dimensional.html)

随着深度学习的发展，模型规模越来越大。如NLP领域，短短几年时间，参数量就从BERT的亿级，发展到GPT-3的1700亿，再到盘古alpha 2000亿，以及当前业界甚至提出百万亿级。由此可以看出，近年来参数规模呈指数增长趋势。另一方面，随着大数据、互联网等领域相关技术的发展，可供模型训练的数据集也极速扩增，例如推荐、自然语言处理等场景的数据集可达数TB。

### [梯度累积](https://www.mindspore.cn/tutorials/experts/zh-CN/r1.10/others/gradient_accumulation.html)

梯度累积是一种训练神经网络的数据样本按Batch拆分为几个小Batch的方式，然后按顺序计算。目的是为了解决由于内存不足，导致Batch size过大神经网络无法训练或者网络模型过大无法加载的OOM（Out Of Memory）问题。

### [Summary](https://www.mindspore.cn/mindinsight/docs/zh-CN/r1.10/summary_record.html)

训练过程中的标量、图像、计算图、训练优化过程以及模型超参等信息记录到文件中，通过可视化界面供用户查看。

### [调试器](https://www.mindspore.cn/mindinsight/docs/zh-CN/r1.10/debugger.html)

MindSpore调试器是为图模式训练提供的调试工具，可以用来查看并分析计算图节点的中间结果。

### [Golden Stick](https://www.mindspore.cn/golden_stick/docs/zh-CN/r0.2/index.html)

MindSpore Golden Stick是华为诺亚团队和华为MindSpore团队联合设计开发的一个模型压缩算法集。包含基本的量化和剪枝方法。

## 与PyTorch典型接口区别

在PyTorch往MindSpore进行网络迁移时，需要注意[与PyTorch典型接口区别](https://www.mindspore.cn/docs/zh-CN/r1.10/migration_guide/typical_api_comparision.html)。

[1] Lin, T. Y. , et al. "Focal Loss for Dense Object Detection." IEEE Transactions on Pattern Analysis & Machine Intelligence PP.99(2017):2999-3007.
