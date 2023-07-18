# 比较与torch.nn.SyncBatchNorm的差异

<a href="https://gitee.com/mindspore/docs/blob/r2.1/docs/mindspore/source_zh_cn/note/api_mapping/pytorch_diff/SyncBatchNorm.md" target="_blank"><img src="https://mindspore-website.obs.cn-north-4.myhuaweicloud.com/website-images/r2.1/resource/_static/logo_source.png"></a>

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

更多内容详见[mindspore.nn.SyncBatchNorm](https://mindspore.cn/docs/zh-CN/r2.1/api_python/nn/mindspore.nn.SyncBatchNorm.html)。

## 差异对比

PyTorch：对输入的数据进行跨设备同步批归一化。

MindSpore：MindSpore此API实现功能与PyTorch基本一致。MindSpore输入仅支持二维和四维。MindSpore中momentum参数默认值为0.9，与PyTorch的momentum转换关系为1-momentum，默认值行为与PyTorch相同；训练以及推理时的参数更新策略和PyTorch有所不同，详细区别请参考[与PyTorch典型区别-BatchNorm](https://www.mindspore.cn/docs/zh-CN/r2.1/migration_guide/typical_api_comparision.html#nn.BatchNorm2d)。

| 分类 | 子类   | PyTorch             | MindSpore            | 差异                                                         |
| ---- | ------ | ------------------- | -------------------- | ------------------------------------------------------------ |
| 参数 | 参数1  | num_features        | num_features         | -                                                            |
|      | 参数2  | eps                 | eps                  | -                                                            |
|      | 参数3  | momentum            | momentum             | 功能一致，但PyTorch中的默认值是0.1，MindSpore中是0.9，与PyTorch的momentum转换关系为1-momentum，默认值行为与PyTorch相同        |
|      | 参数4  | affine              | affine               | -                                                            |
|      | 参数5  | track_running_stats              | use_batch_statistics               | 功能一致，不同值对应的默认方式不同，详细区别请参考[与PyTorch典型区别-nn.BatchNorm2d](https://www.mindspore.cn/docs/zh-CN/r2.1/migration_guide/typical_api_comparision.html#nn.BatchNorm2d)                                |
|      | 参数6  | -                   | gamma_init           |    PyTorch无此参数，MindSpore可以初始化参数gamma的值    |
|      | 参数7  | -                   | beta_init            |    PyTorch无此参数，MindSpore可以初始化参数beta的值     |
|      | 参数8  | -                   | moving_mean_init     |    PyTorch无此参数，MindSpore可以初始化参数moving_mean的值    |
|      | 参数9  | -                   | moving_var_init      |    PyTorch无此参数，MindSpore可以初始化参数moving_var的值     |
|      | 参数10  | process_group        | process_groups      |    -     |
| 输入 | 单输入 | input               | x                    | 接口输入，MindSpore只支持二维和四维输入 |
