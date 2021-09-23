# 自定义Metrics验证模型推理精度

`Linux` `Ascend` `GPU` `CPU` `模型加载` `初级` `中级` `高级`

<!-- TOC -->

- [自定义Metrics验证模型推理精度](#自定义Metrics验证模型推理精度)
    - [概述](#概述)
    - [自定义Metrics步骤](#自定义Metrics步骤)
    - [在Model中使用Metrics](#在Model中使用Metrics)

<!-- /TOC -->

<a href="https://gitee.com/mindspore/docs/blob/r1.5/docs/mindspore/programming_guide/source_zh_cn/self_define_metric.md" target="_blank"><img src="https://gitee.com/mindspore/docs/raw/r1.5/resource/_static/logo_source.png"></a>
&nbsp;&nbsp;

## 概述

当训练任务结束，常常需要评价函数(Metrics)来评估模型的好坏。不同的训练任务往往需要不同的Metrics函数。例如，对于二分类问题，常用的评价指标有precision(准确率)、recall(召回率)等， 而对于多分类任务，可使用宏平均(Macro)和微平均(Micro)来评估。MindSpore中提供的Metrics有：`nn.Accuracy`、`nn.Pecision`、`nn.MAE`、`nn.Topk`、`nn.MSE`等，详情可参考：[Metric](https://www.mindspore.cn/docs/api/zh-CN/r1.5/api_python/mindspore.nn.html#metrics) 。虽然MindSpore提供了大部分常见任务的评价指标，但是无法满足所有任务的需求。因此使用者可针对具体的任务自定义Metrics来评估训练的模型。

以下通过示例来介绍如何自定义Metrics以及如何在`nn.Model`中使用。

## 自定义Metrics步骤

所有的metrics都需要继承`nn.Metric`父类，用户只需要重新实现父类中的`clear`、`update`和`eval`即可。其中`clear`用于初始化相关内部参数；`update`接收网络的预测和真值标签，更新内部变量；`eval`用于计算相关指标并返回计算结果。下面我们以简单的`MAE`为例，介绍这三个函数。

示例代码如下：

```python
import numpy as np
import mindspore
from mindspore import nn, Tensor
from mindspore.nn import Metric
from mindspore.nn import rearrange_inputs

class MyMAE(Metric):

    def __init__(self):
        super(MyMAE, self).__init__()
        self.clear()

    def clear(self):
        """Clears the internal evaluation result."""
        self._abs_error_sum = 0
        self._samples_num = 0

    @rearrange_inputs
    def update(self, *inputs):
        if len(inputs) != 2:
            raise ValueError('Mean absolute error need 2 inputs (y_pred, y), but got {}'.format(len(inputs)))
        y_pred = self._convert_data(inputs[0])
        y = self._convert_data(inputs[1])
        abs_error_sum = np.abs(y.reshape(y_pred.shape) - y_pred)
        self._abs_error_sum += abs_error_sum.sum()
        self._samples_num += y.shape[0]

    def eval(self):
        if self._samples_num == 0:
            raise RuntimeError('Total samples num must not be 0.')
        return self._abs_error_sum / self._samples_num
```

- `clear`方法初始化了两个变量，其中`_abs_error_sum`用于保存误差和，`_samples_num`用于累计数据量。
- `update`方法接收网络预测输出和标签，计算误差，并更新`_abs_error_sum`和`_samples_num`。其中`_convert_data`用于将`Tensor`转换为numpy, 后续可利用numpy做相关计算。如果用户评估网络有多个输出，但只用两个输出进行评估，此时可以使用装饰器`rearrange_inputs`中的`set_indexes`方法指定评估网络输出中哪些用于计算评估指标。update一般在每个step进行计算并更新统计值。
- `eval`方法计算最终的指标并返回结果。eval一般在一个epoch结束后计算最终的评估结果。

下面我们使用上面定义的MAE，计算几组数据的MAE指标。

```python
x = Tensor(np.array([[0.1, 0.2, 0.6, 0.9],[0.1, 0.2, 0.6, 0.9]]), mindspore.float32)
y = Tensor(np.array([[0.1, 0.25, 0.7, 0.9],[0.1, 0.25, 0.7, 0.9]]), mindspore.float32)
error = MyMAE()
error.clear()
error.update(x, y)
result = error.eval()
print(result)

```

```python
0.14999

```

下面的例子展示了装饰器`rearrange_inputs`的用法。假设网络共有三个输出，其中第0， 2个输出可用来计算指标。

```python
x = Tensor(np.array([[0.1, 0.2, 0.6, 0.9],[0.1, 0.2, 0.6, 0.9]]), mindspore.float32)
y = Tensor(np.array([[0.1, 0.25, 0.7, 0.9],[0.1, 0.25, 0.7, 0.9]]), mindspore.float32)
z = Tensor(np.array([[0.1, 0.25, 0.7, 0.9],[0.1, 0.25, 0.7, 0.9]]), mindspore.float32)
error = MyMAE().set_indexes([0, 2])
error.clear()
error.update(x, y, z)
result = error.eval()
print(result)

```

```python
0.1499919

```

## 在Model中使用Metrics

`mindspore.Model`是用于训练和测试的高层API，可以将自定义或MindSpore已有的Metrics作为参数传入，Model能够自动调用传入的Metrics进行评估。`mindspore.Model`的信息，请参考[Model](https://www.mindspore.cn/docs/api/zh-CN/r1.5/api_python/mindspore/mindspore.Model.html#mindspore.Model)。

```python
model = Model(network, loss, metrics={"MyMAE":MyMAE()})
output = model.eval(eval_dataset)

```


