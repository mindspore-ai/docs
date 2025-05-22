# 实现模型故障注入评估模型容错性

[![查看源文件](https://mindspore-website.obs.cn-north-4.myhuaweicloud.com/website-images/master/resource/_static/logo_source.svg)](https://gitee.com/mindspore/docs/blob/master/docs/mindarmour/docs/source_zh_cn/fault_injection.md)

## 概述

在过去十几年，人工智能的应用越来越广。其中也包括safety critical和security critical的领域中，
例如自动驾驶、智能安防、智慧医疗等等。在这些领域中发生故障可能会导致严重的生命财产损失，
因此确保AI模型性能的可靠性和使用过程中的可靠性是至关重要的。

为了保障AI模型在各种故障模式下的可靠性和可用性，对其组件进行严格的测试验证非常重要。
本模块可模拟各类故障场景，实现对模型可靠性的度量评估。

本例会实现一个简单的模型故障注入及容错性评估的功能，整体流程如下：

1. 下载公开数据集和模型参数文件。
2. 构建数据集和模型。
3. 调用故障注入模块。
4. 查看结果。

> 你可以在这里找到完整可运行的样例代码：<https://gitee.com/mindspore/mindarmour/blob/master/examples/reliability/model_fault_injection.py>。

## 准备环节

确保已经正确安装了MindSpore。如果没有，可以通过 [MindSpore安装页面](https://www.mindspore.cn/install) 进行安装。  

### 下载数据集

MNIST手写数据集包含60,000个样本的训练集和10,000个样本的测试集。
> 数据集下载页面：<http://yann.lecun.com/exdb/mnist/>。

将数据集下载并解压到本地路径下，目录结构如下：

```text
- data_path
    - train
        - train-images-idx3-ubyte
        - train-labels-idx1-ubyte
    - test
        - t10k-images-idx3-ubyte
        - t10k-labels-idx1-ubyte
```

### 下载模型参数文件

下载对应模型的参数文件，或者自己训练。
> 参数文件下载页面：<https://www.mindspore.cn/hub>。

### 导入Python库&模块

在使用前，需要导入需要的Python库。

```python
import numpy as np
import mindspore as ms

from mindspore.train import Model
from mindarmour.reliability import FaultInjector
from examples.common.networks.lenet5.lenet5_net import LeNet5
from examples.common.dataset.data_processing import generate_mnist_dataset
```

## 构建数据集和模型

以MNIST数据集和LeNet5模型为例。

构建MNIST数据集：

```python
DATA_FILE = 'PATH_TO_MNIST/'
ds_eval = generate_mnist_dataset(DATA_FILE, batch_size=64)
test_images = []
test_labels = []
for data in ds_eval.create_tuple_iterator(output_numpy=True):
    images = data[0].astype(np.float32)
    labels = data[1]
    test_images.append(images)
    test_labels.append(labels)
ds_data = np.concatenate(test_images, axis=0)
ds_label = np.concatenate(test_labels, axis=0)
```

构建LeNet5网络：

```python
ckpt_path = 'PATH_TO_CHECKPOINT/'
net = LeNet5()
param_dict = ms.load_checkpoint(ckpt_path)
ms.load_param_into_net(net, param_dict)
model = Model(net)
```

## 设置参数及初始化故障注入模块

设置故障注入参数，示例代码如下：

```python
fi_type = ['bitflips_designated', 'precision_loss']
fi_mode = ['single_layer', 'all_layer']
fi_size = [1, 2]
```

初始化故障注入模块：

```python
fi = FaultInjector(model=model, fi_type=fi_type, fi_mode=fi_mode, fi_size=fi_size)
```

参数含义：

- `model(Model)`：需要评估的模型。
- `fi_type(list)`: 注入的故障类型，目前支持8种故障类型，分别为`bitflips_random`、 `bitflips_designated`、 `random`、 `zeros`、 `NaN`、 `INF`、 `anti_activation`和`precision_loss`。
    - `bitflips_random`: 随机反转一位比特位。
    - `bitflips_designated`: 反转关键比特位，关键比特位指对数值影响最大的比特位。
    - `random`: 数值随机，随机范围是[-1, 1]。
    - `zeros`: 数值置零，用零替换原始数值。
    - `NaN`: 数值非数，用NaN替换原始数值。
    - `INF`: 数值无穷，用INF替换原始数值。
    - `anti_activation`: 反激活，反转原始数值符号。
    - `precision_loss`: 原始数值保留一位小数。
- `fi_mode(list)`：故障注入的模式，有两种可选模式，分别是`single_layer` 随机一层注入故障或者`all_layer`每层都注入故障。
- `fi_size(list)`：每次注入故障的具体数量，对于`zeros`、`anti_activation` 和 `precision_loss` 类型故障则代表为张量元素总量的百分比。

## 评估模型的容错性

完成模块初始化后，调用故障注入方法`kick_off`评估模型：

```python
results = fi.kick_off(ds_data, ds_label, iter_times=100)
```

- `ds_data(numpy.ndarray)`：测试数据，将在此数据集上评估模型对于注入故障的容错性。
- `ds_label(numpy.ndarray)`：数据标签，与测试数据对应。
- `iter_times(int)`：每种故障参数评估次数，决定数据批大小。

调用方法`metrics`统计结果：

```python
result_summary = fi.metrics()
```

返回值：

- `results(list)`：每种故障参数下模型的评估结果。
- `result_summary(list)`：按故障模式分别统计评估结果的最大值，最小值和均值。

## 查看结果

```python
for result in results:
    print(result)
for result in result_summary:
    print(result)
```

结果如下所示：

```text
{'original_acc': 0.9797676282051282}
{'type': 'bitflips_designated', 'mode': 'single_layer', 'size': 1, 'acc': 0.7028245192307693, 'SDC': 0.2769431089743589}
{'type': 'bitflips_designated', 'mode': 'single_layer', 'size': 2, 'acc': 0.5052083333333334, 'SDC': 0.4745592948717948}
{'type': 'bitflips_designated', 'mode': 'all_layer', 'size': 1, 'acc': 0.2077323717948718, 'SDC': 0.7720352564102564}
{'type': 'bitflips_designated', 'mode': 'all_layer', 'size': 2, 'acc': 0.15745192307692307, 'SDC': 0.8223157051282051}
{'type': 'precision_loss', 'mode': 'single_layer', 'size': 1, 'acc': 0.9795673076923077, 'SDC': 0.00020032051282048435}
{'type': 'precision_loss', 'mode': 'single_layer', 'size': 2, 'acc': 0.9797676282051282, 'SDC': 0.0}
{'type': 'precision_loss', 'mode': 'all_layer', 'size': 1, 'acc': 0.9794671474358975, 'SDC': 0.00030048076923072653}
{'type': 'precision_loss', 'mode': 'all_layer', 'size': 2, 'acc': 0.9795673076923077, 'SDC': 0.00020032051282048435}
single_layer_acc_mean:0.791842 single_layer_acc_max:0.979768 single_layer_acc_min:0.505208
single_layer_SDC_mean:0.187926 single_layer_SDC_max:0.474559 single_layer_SDC_min:0.000000
all_layer_acc_mean:0.581055 all_layer_acc_max:0.979567 all_layer_acc_min:0.157452
all_layer_SDC_mean:0.398713 all_layer_SDC_max:0.822316 all_layer_SDC_min:0.000200
```

- `original_acc`: 模型原始准确率。
- `SDC(Silent Data Corruption)`: 代表性能下降值，为原始准确率减去当前故障准确率。
- `single_layer_acc_mean/max/min`: 单层故障模式下，准确率的均值/最大值/最小值。
- `single_layer_SDC_mean/max/min`: 单层故障模式下，SDC的均值/最大值/最小值。
- `all_layer_acc_mean/max/min`: 每层故障模式下，准确率的均值/最大值/最小值。
- `all_layer_SDC_mean/max/min`: 每层故障模式下，SDC的均值/最大值/最小值。
