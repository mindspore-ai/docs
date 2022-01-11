# 使用度量方法

<a href="https://gitee.com/mindspore/docs/blob/r1.6/docs/xai/docs/source_zh_cn/using_benchmarks.md" target="_blank"><img src="https://gitee.com/mindspore/docs/raw/r1.6/resource/_static/logo_source.png"></a>

## 什么是度量方法

度量方法是用来为热力图好坏评分的一些算法，目前 MindSpore XAI 为图片分类场景提供四个度量方法：`Robustness`、`Faithfulness`、`ClassSensitivity`和`Localization`。

## 准备

以下教程参考了 [using_benchmarks.py](https://gitee.com/mindspore/xai/blob/r1.6/examples/using_benchmarks.py) 。

请参阅 [下载教程数据集及模型](https://www.mindspore.cn/xai/docs/zh-CN/r1.6/using_explainers.html#id4) 以下载所有本教程所需的文件。

下载教程数据集及模型后，我们要加载一张样本图片，一个训练好的分类器，一个解释器和一张热力图(可选)：

```python
# 必须先把当前目录切换到 xai/examples/
from mindspore import context, load_checkpoint, load_param_into_net
from mindspore_xai.explanation import GradCAM

from common.resnet import resnet50
from common.dataset import load_image_tensor

# 只支持 PYNATIVE_MODE
context.set_context(mode=context.PYNATIVE_MODE)

# 有20个类
num_classes = 20

# 加载训练好的分类器
net = resnet50(num_classes)
param_dict = load_checkpoint("xai_examples_data/ckpt/resnet50.ckpt")
load_param_into_net(net, param_dict)

# [1, 3, 224, 224] Tensor
boat_image = load_image_tensor('xai_examples_data/test/boat.jpg')

# 解释器
grad_cam = GradCAM(net, layer='layer4')

# 5 是'boat'类的ID
saliency = grad_cam(boat_image, targets=5)
```

## 使用 Robustness

`Robustness`是最简单的度量方法，它把随机噪声加入图片作推理并输出最高的召回率作为评分：

```python
from mindspore.nn import Softmax
from mindspore_xai.explanation import Robustness

# 分类器使用 Softmax 作为激活函数
robustness = Robustness(num_classes, activation_fn=Softmax())
# 可以省略 'saliency' 参数
score = robustness.evaluate(grad_cam, boat_image, targets=5, saliency=saliency)
```

如果输入的是一个 1xCx224x224 的图片Tensor，那返回的`score`就是一个只有一个数值的一维Tensor 。

## 使用 Faithfulness 及 ClassSensitivity

使用`Faithfulness`及`ClassSensitivity`的方法跟`Robustness`的使用方法十分相似，但`ClassSensitivity`是全类评分，不能指定`targets` 。

## 使用 Localization

如果有物体的范围或界框，可以使用`Localization`作评分：

```python
import numpy as np
import mindspore as ms
from mindspore import Tensor
from mindspore_xai.explanation import Localization

# 左上角：80,66 到 右下角：223,196 是一条船的界框
mask = np.zeros([1, 1, 224, 224])
mask[:, :, 66:196, 80:223] = 1

mask = Tensor(mask, dtype=ms.float32)

localization = Localization(num_classes)

score = localization.evaluate(grad_cam, boat_image, targets=5, mask=mask)
```
