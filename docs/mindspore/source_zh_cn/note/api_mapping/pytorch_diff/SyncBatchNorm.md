# 比较与torch.nn.SyncBatchNorm的差异

[![查看源文件](https://mindspore-website.obs.cn-north-4.myhuaweicloud.com/website-images/master/resource/_static/logo_source.svg)](https://gitee.com/mindspore/docs/blob/master/docs/mindspore/source_zh_cn/note/api_mapping/pytorch_diff/SyncBatchNorm.md)

## torch.nn.SyncBatchNorm

```text
class torch.nn.SyncBatchNorm(
    num_features,
    eps=1e-05,
    momentum=0.1,
    affine=True,
    track_running_stats=True,
    process_group=None
)(input) -> Tensor
```

更多内容详见[torch.nn.SyncBatchNorm](https://pytorch.org/docs/1.8.1/generated/torch.nn.SyncBatchNorm.html)。

## mindspore.nn.SyncBatchNorm

```text
class mindspore.nn.SyncBatchNorm(
    num_features,
    eps=1e-5,
    momentum=0.9,
    affine=True,
    gamma_init='ones',
    beta_init='zeros',
    moving_mean_init='zeros',
    moving_var_init='ones',
    use_batch_statistics=None,
    process_groups=None
)(x) -> Tensor
```

更多内容详见[mindspore.nn.SyncBatchNorm](https://mindspore.cn/docs/zh-CN/master/api_python/nn/mindspore.nn.SyncBatchNorm.html)。

## 差异对比

PyTorch：对输入的数据进行跨设备同步批归一化。

MindSpore：MindSpore此API实现功能与PyTorch基本一致。MindSpore输入仅支持二维和四维。MindSpore中momentum参数默认值为0.9，与PyTorch的momentum转换关系为1-momentum，默认值行为与PyTorch相同；训练以及推理时的参数更新策略和PyTorch有所不同。

| 分类 | 子类   | PyTorch             | MindSpore            | 差异                                                         |
| ---- | ------ | ------------------- | -------------------- | ------------------------------------------------------------ |
| 参数 | 参数1  | num_features        | num_features         | -                                                            |
|      | 参数2  | eps                 | eps                  | -                                                            |
|      | 参数3  | momentum            | momentum             | 功能一致，但PyTorch中的默认值是0.1，MindSpore中是0.9，与PyTorch的momentum转换关系为1-momentum，默认值行为与PyTorch相同        |
|      | 参数4  | affine              | affine               | -                                                            |
|      | 参数5  | track_running_stats              | use_batch_statistics               | 功能一致，不同值对应的默认方式不同                                |
|      | 参数6  | -                   | gamma_init           |    PyTorch无此参数，MindSpore可以初始化参数gamma的值    |
|      | 参数7  | -                   | beta_init            |    PyTorch无此参数，MindSpore可以初始化参数beta的值     |
|      | 参数8  | -                   | moving_mean_init     |    PyTorch无此参数，MindSpore可以初始化参数moving_mean的值    |
|      | 参数9  | -                   | moving_var_init      |    PyTorch无此参数，MindSpore可以初始化参数moving_var的值     |
|      | 参数10  | process_group        | process_groups      |    -     |
| 输入 | 单输入 | input               | x                    | 接口输入，MindSpore只支持二维和四维输入 |

详细区别如下：
BatchNorm是CV领域比较特殊的正则化方法，它在训练和推理的过程中有着不同计算流程，通常由算子属性控制。MindSpore和PyTorch的 BatchNorm在这一点上使用了两种不同的参数组。

- 差异一

  `torch.nn.SyncBatchNorm` 在不同参数下的状态

  |training|track_running_stats|状态|
  |----|----|--------------------------------------|
  |True|True|期望中训练的状态，running_mean 和 running_var 会跟踪整个训练过程中 batch 的统计特性，而每组输入数据用当前 batch 的 mean 和 var 统计特性做归一化，然后再更新 running_mean 和 running_var。|
  |True|False|每组输入数据会根据当前 batch 的统计特性做归一化，但不会有 running_mean 和 running_var 参数了。|
  |False|True|期望中推理的状态，BN 使用 running_mean 和 running_var 做归一化，并且不会对其进行更新。|
  |False|False|效果同第二点，只不过处于推理状态，不会学习 weight 和 bias 两个参数。一般不采用该状态。|

  `mindspore.nn.SyncBatchNorm` 在不同参数下的状态

  |use_batch_statistics|状态|
  |----|--------------------------------------|
  |True|期望中训练的状态，moving_mean 和 moving_var 会跟踪整个训练过程中 batch 的统计特性，而每组输入数据用当前 batch 的 mean 和 var 统计特性做归一化，然后再更新 moving_mean 和 moving_var。
  |Fasle|期望中推理的状态，BN 使用 moving_mean 和 moving_var 做归一化，并且不会对其进行更新。
  |None|自动设置 use_batch_statistics。如果是训练，use_batch_statistics=True，如果是推理，use_batch_statistics=False。

  通过比较可以发现，`mindspore.nn.SyncBatchNorm`  相比 `torch.nn.SyncBatchNorm`，少了两种冗余的状态，仅保留了最常用的训练和推理两种状态。

- 差异二

  在PyTorch中，网络默认是训练模式，而MindSpore默认是推理模式（`is_training`为False），需要通过 `net.set_train()` 方法将网络调整为训练模式，此时才会在训练期间去对参数 `mean` 和 `variance` 进行计算，否则，在推理模式下，参数会尝试从checkpoint去加载。

- 差异三

  BatchNorm系列算子的momentum参数在MindSpore和PyTorch表示的意义相反，关系为：
  $$momentum_{pytorch} = 1 - momentum_{mindspore}$$
