# 纵向联邦-基于差分隐私的标签保护

<a href="https://gitee.com/mindspore/docs/blob/master/docs/federated/docs/source_zh_cn/secure_vertical_federated_learning_with_DP.md" target="_blank"><img src="https://mindspore-website.obs.cn-north-4.myhuaweicloud.com/website-images/master/resource/_static/logo_source.png"></a>

注：这是一个实验特性，未来有可能被修改或删除。

## 背景

纵向联邦学习（vFL）是联邦学习（FL）的一大重要分支。当不同的参与方拥有来自相同一批用户但属性不同的数据时，他们便可使用vFL进行协同训练。在vFL中，拥有属性的参与方都会持有一个下层网络（Bottom Model），他们分别将属性输入下层网络，得到中间结果（embedding），发送给拥有标签的参与方（简称leader方，如下图参与方B，而不拥有标签的被称作follower方，如下图参与方A），leader方使用embedding和标签来训练上层网络，再将算得的梯度回传给各个参与方用以训练下层网络。由此可见，vFL不需要任何参与方上传自己的原始数据即可协同训练模型。

![image.png](./images/vfl_1.png)

然而，leader方传回给follower方的梯度中包含的信息较多，让follower方能够从梯度反推出leader方持有的标签。在这样的背景下，我们需要对vFL的训练提供更强的隐私保证来规避标签泄露的风险。

差分隐私是一种严格基于统计学/信息论的隐私定义，能够确保任何一个个体数据的改变不会让算法的输出有很大的区别（通常通过随机变量分布的重叠实现）。从而在理论上就严格确保了，算法的结果没有可能反推出个体的数据。本设计方案基于label differential privacy，在纵向联邦学习训练时为leader参与方的标签提供差分隐私保证，从而使攻击者无法从回传的梯度反推出数据的标签信息。

## 算法实现

我们采用了一种轻量级的label dp实现方式：训练时，在使用leader参与方的标签数据之前，将一定比例的标签进行随机翻转后再进行训练。由于随机性的引入，攻击者若想反推标签，最多只能反推出随机翻转/扰动之后的标签，增加了反推出原始标签的难度，满足差分隐私保证。在实际应用时，我们可以调整隐私参数`eps`（可以理解为随机翻转标签的比例）来满足不同的场景需求，可以在需要高隐私时使用较小的`eps`，需要高精度时使用较大的`eps`。

![image.png](./images/label_dp.png)

此方案基于randomized response算法，在vFL的leader训练前将用户的标签随机翻转/扰乱后再进行训练，实际实现时分为binary标签和onehot标签两种情况：

### binary标签保护

1. 根据预设的隐私参数eps，计算翻转概率$p = \frac{1}{1 + e^{eps}}$。
2. 以概率$p$翻转每个标签。

### onehot标签保护

1. 对于n个类的标签，计算$p_1 = \frac{e^{eps}}{n - 1 + e^{eps}}$，$p_2 = \frac{1}{n - 1 + e^{eps}}$。
2. 根据以下概率随机扰乱标签：维持当前标签不变的概率为$p_1$；改成其他n - 1个类里的任意一个的概率都为$p_2$。

## 快速体验

我们以[Wide&Deep纵向联邦学习案例](https://gitee.com/mindspore/federated/tree/master/example/splitnn_criteo)中的单线程案例为例，介绍如何在一个纵向联邦模型中加入label dp保护。

### 前置需要

以下操作皆可参考[Wide&Deep纵向联邦学习案例](https://gitee.com/mindspore/federated/tree/master/example/splitnn_criteo)：

1. 在 Python 环境中安装MindSpore1.8.1或其更高版本，请参考[MindSpore官网安装指引](https://www.mindspore.cn/install)。
2. 安装MindSpore Federated及所依赖Python库。
3. 准备criteo数据集。

### 启动脚本

1. 下载federated仓

   ```bash
   git clone https://gitee.com/mindspore/federated.git
   ```

2. 进入脚本所在文件夹

   ```bash
   cd federated/example/splitnn_criteo
   ```

3. 运行脚本

   ```bash
   sh run_vfl_train_label_dp.sh
   ```

### 查看结果

在训练日志`log_local_gpu.txt`查看模型训练的loss变化：

```sh
INFO:root:epoch 0 step 100/2582 wide_loss: 0.588637 deep_loss: 0.589756
INFO:root:epoch 0 step 200/2582 wide_loss: 0.561055 deep_loss: 0.562271
INFO:root:epoch 0 step 300/2582 wide_loss: 0.556246 deep_loss: 0.557509
INFO:root:epoch 0 step 400/2582 wide_loss: 0.557931 deep_loss: 0.559055
INFO:root:epoch 0 step 500/2582 wide_loss: 0.553283 deep_loss: 0.554257
INFO:root:epoch 0 step 600/2582 wide_loss: 0.549618 deep_loss: 0.550489
INFO:root:epoch 0 step 700/2582 wide_loss: 0.550243 deep_loss: 0.551095
INFO:root:epoch 0 step 800/2582 wide_loss: 0.549496 deep_loss: 0.550298
INFO:root:epoch 0 step 900/2582 wide_loss: 0.549224 deep_loss: 0.549974
INFO:root:epoch 0 step 1000/2582 wide_loss: 0.547547 deep_loss: 0.548288
INFO:root:epoch 0 step 1100/2582 wide_loss: 0.546989 deep_loss: 0.547737
INFO:root:epoch 0 step 1200/2582 wide_loss: 0.552165 deep_loss: 0.552862
INFO:root:epoch 0 step 1300/2582 wide_loss: 0.546926 deep_loss: 0.547594
INFO:root:epoch 0 step 1400/2582 wide_loss: 0.558071 deep_loss: 0.558702
INFO:root:epoch 0 step 1500/2582 wide_loss: 0.548258 deep_loss: 0.548910
INFO:root:epoch 0 step 1600/2582 wide_loss: 0.546442 deep_loss: 0.547072
INFO:root:epoch 0 step 1700/2582 wide_loss: 0.549062 deep_loss: 0.549701
INFO:root:epoch 0 step 1800/2582 wide_loss: 0.546558 deep_loss: 0.547184
INFO:root:epoch 0 step 1900/2582 wide_loss: 0.542755 deep_loss: 0.543386
INFO:root:epoch 0 step 2000/2582 wide_loss: 0.543118 deep_loss: 0.543774
INFO:root:epoch 0 step 2100/2582 wide_loss: 0.542587 deep_loss: 0.543265
INFO:root:epoch 0 step 2200/2582 wide_loss: 0.545770 deep_loss: 0.546451
INFO:root:epoch 0 step 2300/2582 wide_loss: 0.554520 deep_loss: 0.555198
INFO:root:epoch 0 step 2400/2582 wide_loss: 0.551129 deep_loss: 0.551790
INFO:root:epoch 0 step 2500/2582 wide_loss: 0.545622 deep_loss: 0.546315
...
```

## 深度体验

我们以[Wide&Deep纵向联邦学习案例](https://gitee.com/mindspore/federated/tree/master/example/splitnn_criteo)中的单线程案例为例，介绍在纵向联邦模型中加入label dp保护的具体操作方法。

### 前置需要

和[快速体验](#快速体验)相同：安装MindSpore、安装MindSpore Federated、准备数据集。

### 方案一：调用FLModel类中集成的label dp功能

MindSpore Federated纵向联邦学习框架采用`FLModel`（参见[纵向联邦学习模型训练接口](https://www.mindspore.cn/federated/docs/zh-CN/master/vertical/vertical_federated_FLModel.html)）和yaml文件（参见[纵向联邦学习yaml详细配置项](https://www.mindspore.cn/federated/docs/zh-CN/master/vertical/vertical_federated_yaml.html)），建模纵向联邦学习的训练过程。

我们在`FLModel`类中集成了label dp功能。使用者在正常完成整个纵向联邦学习的训练过程建模后（关于vFL训练的详细介绍可以参见[纵向联邦学习模型训练 - 盘古α大模型跨域训练](https://www.mindspore.cn/federated/docs/zh-CN/master/split_pangu_alpha_application.html)），只需在标签方的yaml文件中，在`privacy`模块下加入`label_dp`子模块（若没有`privacy`模块则需使用者输入添加），并在`label_dp`模块内设定`eps`参数（差分隐私参数$\epsilon$，使用者可以根据实际需求设置此参数的值），即可让模型享受label dp保护：

```yaml
privacy:
  ...
  ...
  ...
  label_dp:
    eps: 1.0
```

### 方案二：直接调用LabelDP类

使用者也可以直接调用`LabelDP`类，更加灵活地使用label dp功能。`LabelDP`类集成在`mindspore_federated.privacy`模块中，使用者可以先指定`eps`的值定义一个`LabelDP`对象，然后将标签组作为参数传入这个对象，对象的`__call__`函数中会自动识别当前传入的是onehot还是binary标签，输出一个经过label dp处理后的标签组。可参见以下范例：

```python
# make private a batch of binary labels
import numpy as np
import mindspore
from mindspore import Tensor
from mindspore_federated.privacy import LabelDP
label_dp = LabelDP(eps=0.0)
label = Tensor(np.zero(5, 1), dtype=mindspore.float32)
dp_label = label_dp(label)

# make private a batch of onehot labels
label = Tensor(np.hstack((np.ones((5, 1)), np.zeros((5, 2)))), dtype=mindspore.float32)
dp_label = label_dp(label)
print(dp_label)
```
