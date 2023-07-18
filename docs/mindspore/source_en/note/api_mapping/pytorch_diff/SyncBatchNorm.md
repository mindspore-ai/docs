# Differences with torch.nn.SyncBatchNorm

<a href="https://gitee.com/mindspore/docs/blob/r2.1/docs/mindspore/source_en/note/api_mapping/pytorch_diff/SyncBatchNorm.md" target="_blank"><img src="https://mindspore-website.obs.cn-north-4.myhuaweicloud.com/website-images/r2.1/resource/_static/logo_source_en.png"></a>

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

For more information, see [torch.nn.SyncBatchNorm](https://pytorch.org/docs/1.8.1/generated/torch.nn.SyncBatchNorm.html).

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

For more information, see [mindspore.nn.SyncBatchNorm](https://mindspore.cn/docs/en/r2.1/api_python/nn/mindspore.nn.SyncBatchNorm.html).

## Differences

PyTorch: Perform cross-device synchronous batch normalization of input data.

MindSpore: MindSpore API is basically the same as PyTorch, and the MindSpore input only supports 2D and 4D. The default value of momentum in MindSpore is 0.9, and the conversion relationship with PyTorch momentum is 1-momentum, with the same default behavior as PyTorch. The training and the parameter update strategy during inference are different from PyTorch. Please refer to [Typical Differences with PyTorch - BatchNorm](https://www.mindspore.cn/docs/en/r2.1/migration_guide/typical_api_comparision.html#nn-batchnorm2d).

| Categories | Subcategories | PyTorch      | MindSpore     | Differences   |
| ---------- | ------------- | ------------ | ---------     | ------------- |
| Parameters | Parameter 1  | num_features        | num_features         | -                                                            |
|      | Parameter 2  | eps                 | eps                  | -                                                            |
|      | Parameter 3  | momentum            | momentum             | Consistent functionality, but the default value is 0.1 in PyTorch and 0.9 in MindSpore. The conversion relationship with PyTorch momentum is 1-momentum with the same default behavior as PyTorch        |
|      | Parameter 4  | affine              | affine               | -                                                            |
|      | Parameter 5  | track_running_stats              | use_batch_statistics               | Consistent function. Different values correspond to different default methods. Please refer to [Typical Differences with PyTorch - nn.BatchNorm2d](https://www.mindspore.cn/docs/en/r2.1/migration_guide/typical_api_comparision.html#nn-batchnorm2d) for detailed differences comparison                               |
|      | Parameter 6  | -                   | gamma_init           |    PyTorch does not have this parameter, and MindSpore can initialize the value of the parameter gamma    |
|      | Parameter 7  | -                   | beta_init            |    PyTorch does not have this parameter, and MindSpore can initialize the value of the parameter beta     |
|      | Parameter 8  | -                   | moving_mean_init     |    PyTorch does not have this parameter, and MindSpore can initialize the value of the parameter moving_mean    |
|      | Parameter 9  | -                   | moving_var_init      |    PyTorch does not have this parameter, and MindSpore can initialize the value of the parameter moving_var     |
|      | Parameter 10  | process_group        | process_groups      |    -     |
| Input | Single input | input               | x                    | Interface input. MindSpore only supports 2-D and 4-D input |


