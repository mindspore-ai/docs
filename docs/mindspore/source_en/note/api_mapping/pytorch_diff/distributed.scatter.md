# Differences with torch.distributed.scatter

[![View Source On Gitee](https://mindspore-website.obs.cn-north-4.myhuaweicloud.com/website-images/r2.4.1/resource/_static/logo_source_en.svg)](https://gitee.com/mindspore/docs/blob/r2.4.1/docs/mindspore/source_en/note/api_mapping/pytorch_diff/distributed.scatter.md)

## torch.distributed.scatter

```python
torch.distributed.scatter(
    tensor,
    scatter_list=None,
    src=0,
    group=None,
    async_op=False
)
```

For more information, see [torch.distributed.scatter](https://pytorch.org/docs/1.8.1/distributed.html#torch.distributed.scatter)。

## mindspore.communication.comm_func.scatter_tensor

```python
mindspore.communication.comm_func.scatter_tensor(tensor, src=0, group=GlobalComm.WORLD_COMM_GROUP)
```

For more information, see [mindspore.communication.comm_func.scatter_tensor](https://www.mindspore.cn/docs/en/r2.4.1/api_python/communication/mindspore.communication.comm_func.scatter_tensor.html#mindspore.communication.comm_func.scatter_tensor)。

## Differences

API function of MindSpore is not consistent with that of PyTorch.

PyTorch：`tensor` will store the scattered result, the tensors to be gatherd `scatter_list`, the rank(global rank) `src` of the process, the communication `group` to work on, and the async op flag `async_op`. The process will scatter the tensors and store in `tensor`. The return is a async work handle if `async_op` is True, otherwise is `None`. The async op flag `async_op` and the working device lists `device_ids` are not supported.

MindSpore: The inputs contains the `tensor` to be scattered, the rank(global rank) `src` of the process, the communication `group` to work on, and it will return the tensor. The dimension 0 of data is equal to the dimension of input tensor divided by `src`, and the other dimension keep the same.

|Parameters | Parameters 1 | tensor | tensor |PyTorch：`tensor` will store the scattered result. MindSpore：the tensor to be scattered |
| | Parameters 2 | scatter_list | - | PyTorch：the tensors to be scattered, MindSpore does not have this parameter|
| | Parameters 3 | src | src |No difference|
| | Parameters 4 | group | group |No difference|
| | Parameters 5 | async_op | - |PyTorch: the async op flag. MindSpore: does not have this parameter.  |