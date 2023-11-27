# Differences with torch.nn.SyncBatchNorm

[![View Source On Gitee](https://mindspore-website.obs.cn-north-4.myhuaweicloud.com/website-images/r2.3/resource/_static/logo_source_en.svg)](https://gitee.com/mindspore/docs/blob/r2.3/docs/mindspore/source_en/note/api_mapping/pytorch_diff/SyncBatchNorm.md)

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

For more information, see [mindspore.nn.SyncBatchNorm](https://mindspore.cn/docs/en/r2.3/api_python/nn/mindspore.nn.SyncBatchNorm.html).

## Differences

PyTorch: Perform cross-device synchronous batch normalization of input data.

MindSpore: MindSpore API is basically the same as PyTorch, and the MindSpore input only supports 2D and 4D. The default value of momentum in MindSpore is 0.9, and the conversion relationship with PyTorch momentum is 1-momentum, with the same default behavior as PyTorch. The training and the parameter update strategy during inference are different from PyTorch.

| Categories | Subcategories | PyTorch      | MindSpore     | Differences   |
| ---------- | ------------- | ------------ | ---------     | ------------- |
| Parameters | Parameter 1  | num_features        | num_features         | -                                                            |
|      | Parameter 2  | eps                 | eps                  | -                                                            |
|      | Parameter 3  | momentum            | momentum             | Consistent functionality, but the default value is 0.1 in PyTorch and 0.9 in MindSpore. The conversion relationship with PyTorch momentum is 1-momentum with the same default behavior as PyTorch        |
|      | Parameter 4  | affine              | affine               | -                                                            |
|      | Parameter 5  | track_running_stats              | use_batch_statistics               | Consistent function. Different values correspond to different default methods.                               |
|      | Parameter 6  | -                   | gamma_init           |    PyTorch does not have this parameter, and MindSpore can initialize the value of the parameter gamma    |
|      | Parameter 7  | -                   | beta_init            |    PyTorch does not have this parameter, and MindSpore can initialize the value of the parameter beta     |
|      | Parameter 8  | -                   | moving_mean_init     |    PyTorch does not have this parameter, and MindSpore can initialize the value of the parameter moving_mean    |
|      | Parameter 9  | -                   | moving_var_init      |    PyTorch does not have this parameter, and MindSpore can initialize the value of the parameter moving_var     |
|      | Parameter 10  | process_group        | process_groups      |    -     |
| Input | Single input | input               | x                    | Interface input. MindSpore only supports 2-D and 4-D input |

The detailed differences are as follows:
BatchNorm is a special regularization method in the CV field. It has different computation processes during training and inference and is usually controlled by operator attributes. BatchNorm of MindSpore and PyTorch uses two different parameter groups at this point.

- Difference 1

`torch.nn.SyncBatchNorm` status under different parameters

|training|track_running_stats|Status|
|:----|:----|:--------------------------------------|
|True|True|Expected training status. `running_mean` and `running_var` trace the statistical features of the batch in the entire training process. Each group of input data is normalized based on the mean and var statistical features of the current batch, and then `running_mean` and `running_var` are updated.|
|True|False|Each group of input data is normalized based on the statistics feature of the current batch, but the `running_mean` and `running_var` parameters do not exist.|
|False|True|Expected inference status. The BN uses `running_mean` and `running_var` for normalization and does not update them.|
|False|False|The effect is the same as that of the second status. The only difference is that this is the inference status and does not learn the weight and bias parameters. Generally, this status is not used.|

`mindspore.nn.SyncBatchNorm` status under different parameters

|use_batch_statistics|Status|
|:----|:--------------------------------------|
|True|Expected training status. `moving_mean` and `moving_var` trace the statistical features of the batch in the entire training process. Each group of input data is normalized based on the mean and var statistical features of the current batch, and then `moving_mean` and `moving_var` are updated.
|Fasle|Expected inference status. The BN uses `moving_mean` and `moving_var` for normalization and does not update them.
|None|`use_batch_statistics` is automatically set. For training, set `use_batch_statistics` to `True`. For inference, `set use_batch_statistics` to `False`.

Compared with `torch.nn.SyncBatchNorm`, `mindspore.nn.SyncBatchNorm` does not have two redundant states and retains only the most commonly used training and inference states.

- Difference 2

In PyTorch, the network is in training mode by default, while in MindSpore, it is in inference mode by default (`is_training` is False). You need to use the `net.set_train()` method in MindSpore to switch the network to training mode. In this case, the parameters `mean` and `variance` are calculated during the training. Otherwise, in inference mode, the parameters are loaded from the checkpoint.

- Difference 3

The meaning of the momentum parameter of the BatchNorm series operators in MindSpore is opposite to that in PyTorch. The relationship is as follows:

$$momentum_{pytorch} = 1 - momentum_{mindspore}$$
