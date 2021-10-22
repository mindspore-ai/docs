
# 自定义评价指标

`Ascend` `GPU` `CPU` `进阶` `模型评估`

<!-- TOC -->

- [自定义评价指标](#自定义评价指标)
    - [Metrics自定义方法](#metrics自定义方法)
        - [导入Metric模块](#导入metric模块)
        - [定义Metrics](#定义metrics)
        - [在框架中导入Metrics](#在框架中导入metrics)

<!-- /TOC -->

<a href="https://gitee.com/mindspore/docs/blob/r1.5/tutorials/source_zh_cn/intermediate/custom/metric.md" target="_blank"><img src="https://gitee.com/mindspore/docs/raw/r1.5/resource/_static/logo_source.png"></a>

评价指标（Metrics）可以用来评估模型结果的可信度。

MindSpore提供了多种Metrics评估指标，如：`accuracy`、`loss`、`precision`、`recall`、`F1`等，完整的`Metrics`功能可参考[API](https://www.mindspore.cn/docs/api/zh-CN/r1.5/api_python/mindspore.nn.html#metrics)。

用户也可根据需求，自行开发并使用Metrics。

## Metrics自定义方法

通过Class实现一个自定义的 Metric 功能，其中包含以下四部分：

- `init`：初始化，同时进行输入的校验。
- `clear`：变量初始化。
- `update`：进行中间过程的计算。
- `eval`：计算得到最后的输出值。

下面以相似度计算函数`Dice`为例，讲解 Metrics 的开发过程。

### 导入Metric模块

```python
import numpy as np
from mindspore.nn import Metric, rearrange_inputs
```

### 定义Metrics

`Dice`实际上计算了两个样本间的相似度，数学公式可以表达为：

$$ dice = \frac{2 \times (pred \bigcap  true)}{pred \bigcup true} $$

Dice的输入为两个尺度相同的Tensor, list或numpy，一个为预测值，一个为实际值。最后输出两个Tensor间的相似度计算值。其中为防止计算过程中分母为零，引入参数smooth，默认值为1e-5。

代码实现为：

```python
class Dice(Metric):

    def __init__(self, smooth=1e-5):
        """调用super进行初始化"""
        super(Dice, self).__init__()
        self.smooth = smooth
        # 调用clear清空变量
        self.clear()

    def clear(self):
        """清除内部计算结果，变量初始化"""
        self._dice_coeff_sum = 0
        self._samples_num = 0

    @rearrange_inputs
    def update(self, *inputs):
        """更新内部计算结果"""

        # 校验输入的数量，y_pred为预测值，y为实际值
        if len(inputs) != 2:
            raise ValueError('Dice need 2 inputs (y_pred, y), but got {}'.format(len(inputs)))
        # 将输入的数据格式变为numpy array
        y_pred = inputs[0].asnumpy()
        y = inputs[1].asnumpy()
        # 参数计算
        self._samples_num += y.shape[0]

        # 校验输入的shape是否一致
        if y_pred.shape != y.shape:
            raise RuntimeError('y_pred and y should have same the dimension, but the shape of y_pred is{}, '
                               'the shape of y is {}.'.format(y_pred.shape, y.shape))

        # 根据公式实现Dice的过程计算
        intersection = np.dot(y_pred.flatten(), y.flatten())
        unionset = np.dot(y_pred.flatten(), y_pred.flatten()) + np.dot(y.flatten(), y.flatten())

        single_dice_coeff = 2 * float(intersection) / float(unionset + self.smooth)
        self._dice_coeff_sum += single_dice_coeff


    def eval(self):
        """进行Dice计算"""
        if self._samples_num == 0:
            raise RuntimeError('Total samples num must not be 0.')

        return self._dice_coeff_sum / float(self._samples_num)

```

### 在框架中导入Metrics

在同级目录中的[\_\_init\_\_.py](https://gitee.com/mindspore/mindspore/blob/r1.5/mindspore/nn/metrics/__init__.py)文件中，添加已经定义好的[Dice](https://gitee.com/mindspore/mindspore/blob/r1.5/mindspore/nn/metrics/dice.py)。可以点击[链接](https://gitee.com/mindspore/mindspore/tree/r1.5/mindspore/nn/metrics)查看文件的具体位置，Metrics在框架中位于`mindspore/nn/metrics/`目录下：

```text
__all__ = [
…
    "Dice",
…
]

__factory__ = {
…
    'dice': Dice,
…
}
```
