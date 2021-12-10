# 实现图像数据概念漂移检测应用

<!-- /TOC -->

- [实现图像数据概念漂移检测应用](#实现时序数据概念漂移检测应用)
    - [概述](#概述)
    - [准备环节](#准备环节)
        - [准备数据集](#准备数据集)
        - [导入Python库&模块](#导入python库模块)
    - [加载数据](#加载数据)
    - [加载神经网络模型](#加载神经网络模型)
    - [初始化图像概念漂移检测模块](#初始化图像概念漂移检测模块)
    - [获得最优概念漂移检测阈值](#获得最优概念漂移检测阈值)
    - [执行概念漂移检测](#执行图像概念漂移检测)
    - [查看结果](#查看结果)

<!-- /TOC -->

<a href="https://gitee.com/mindspore/docs/blob/master/docs/mindarmour/docs/source_zh_cn/concept_drift_images.md" target="_blank"><img src="https://gitee.com/mindspore/docs/raw/master/resource/_static/logo_source.png"></a>

## 概述

图像数据的概念漂移（Concept Drift）是AI学习领域的一种重要数据现象，表现为在线推理的图像数据（实时分布）
与训练阶段（历史分布）不一致，也称作Out-of-Distribution（OOD）。例如，基于MNIST数据集进行训练获得神经网络模型，但实际的测试数据为CIFAR-10
数据环境，那么，CIFAR-10数据集则是OOD样本。

本示例提出一种检测图像数据分布变化的方法，整体流程如下：

1. 加载公开数据集或使用用户自定义数据。
2. 加载神经网络模型。
3. 初始化图像概念漂移类参数。
4. 获得最优概念漂移检测阈值。
5. 执行概念漂移检测函数。
6. 查看结果。

> 你可以在这里找到完整可运行的样例代码：<https://gitee.com/mindspore/mindarmour/blob/master/examples/reliability/concept_drift_check_images_lenet.py>。

## 准备环节

确保已经正确安装了MindSpore。如果没有，可以通过[MindSpore安装页面](https://www.mindspore.cn/install)进行安装。  

### 准备数据集

示例中用到公开图像数据集MNIST和CIFAR-10。
> 数据集下载页面：<http://yann.lecun.com/exdb/mnist/>，<http://www.cs.toronto.edu/~kriz/cifar.html>。


### 导入Python库&模块

在使用前，需要导入需要的Python库。

```python
import numpy as np
from mindspore import Tensor
from mindspore import Model
from mindarmour.utils import LogUtil
from mindspore import Model, nn, context
from examples.common.networks.lenet5.lenet5_net_for_fuzzing import LeNet5
from mindspore import load_checkpoint, load_param_into_net
from mindarmour.reliability.concept_drift.concept_drift_check_images import OodDetectorFeatureCluster
```

## 加载数据

1. 将MNIST数据集作为训练集`ds_train`，这里`ds_train`只包含image数据，不包含label。 

2. 将MNIST和CIFAR-10的混合数据集作为测试集`ds_test`，这里`ds_test`只包含image数据，不包含label。

3. 将另一组MNIST和CIFAR-10的混合数据集作为验证样本，记作`ds_eval`，这里`ds_eval`只包含image数据，不包含label。
`ds_eval`另行标记，其中非OOD样本标记为0，OOD样本标记为1，`ds_eval`的标记单独记作`ood_label`。

```python
ds_train = np.load('/dataset/concept_train_lenet.npy')
ds_test = np.load('/dataset/concept_test_lenet2.npy')
ds_eval = np.load('/dataset/concept_test_lenet1.npy')
```

`ds_train(numpy.ndarray)`: 训练集，只包含image数据。   
`ds_test(numpy.ndarray)`: 测试集，只包含image数据。  
`ds_eval(numpy.ndarray)`: 验证集，只包含image数据。  



## 加载神经网络模型

利用训练集`ds_train`以及`ds_train`所对应的分类`label`，训练神经网络LeNet，并加载模型。这里，我们直接导入已训练好的模型文件。

此处的`label`区别于前文提到的`ood_label`。`label`表示样本的分类标签，`ood_label`表示样本是否属于OOD的标签。

```python
ckpt_path = '../../dataset/trained_ckpt_file/checkpoint_lenet-10_1875.ckpt'
net = LeNet5()
load_dict = load_checkpoint(ckpt_path)
load_param_into_net(net, load_dict)
model = Model(net)
```


`ckpt_path(str)`: 模型文件路径。


## 初始化图像概念漂移检测模块

导入概念漂移检测模块，并初始化。

```python
detector = OodDetectorFeatureCluster(model, ds_train, n_cluster=10, layer='output[:Tensor]')
```

`model(Model)`: 神经网络模型，由训练集`ds_train`和其分类标签训练所得。   
`ds_train(numpy.ndarray)`: 训练集，只包含image数据。   
`n_cluster(int)`: 特征聚类数目。  
`layer(str)`: 神经网络用于提取特征的层的名称。

## 获取最优概念漂移检测阈值

基于验证集`ds_eval` 和其OOD标签`ood_label`，获得最优概念漂移检测阈值。

这里验证集`ds_eval`可人为构造，例如由50%的MNIST数据集和50%的CIFAR-10数据集组成，因此，OOD标签`ood_label`可以得知前50%标签值为0，后50%标签值为1。

```python
num = int(len(ds_eval) / 2)
ood_label = np.concatenate((np.zeros(num), np.ones(num)), axis=0)  # ID data = 0, OOD data = 1
optimal_threshold = detector.get_optimal_threshold(ood_label, ds_eval)
```


`ds_eval(numpy.ndarray)`: 验证集，只包含image数据。  
`ood_label(numpy.ndarray)`: 验证集`ds_eval`的OOD标签，非OOD样本标记为0，OOD样本标记为1。   


当然，如果用户很难获得`ds_eval`和OOD标签`ood_label`，`optimal_threshold`值可以人为灵活设定，`optimal_threshold`值是[0，1]之间的浮点数。


## 执行概念漂移检测


```python
result = detector.ood_predict(optimal_threshold, ds_test)
```


`ds_test(numpy.ndarray)`: 测试集，只包含image数据。  
`optimal_threshold(float)`: 最优阈值。可通过执行`detector.get_optimal_threshold(ood_label, ds_eval)`获得。
但如果用户很难获得ds_eval和OOD标签`ood_label`，`optimal_threshold`值可以人为灵活设定，`optimal_threshold`值是[0，1]之间的浮点数。


## 查看结果

```python
print(result)
```

`result(numpy.ndarray)`: 由元素0和1构成的一维数组，对应了`ds_test`的OOD检测结果。
例如`ds_test`是由5个MNIST和5个CIFAR-10数据组成的数据集，那么检测结果为[0,0,0,0,0,1,1,1,1,1]。