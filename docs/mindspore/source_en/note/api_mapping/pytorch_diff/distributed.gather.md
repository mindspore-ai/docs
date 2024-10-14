# Differences with torch.distributed.gather

[![View Source On Gitee](https://mindspore-website.obs.cn-north-4.myhuaweicloud.com/website-images/r2.4.0/resource/_static/logo_source.svg)](https://gitee.com/mindspore/docs/blob/r2.4.0/docs/mindspore/source_zh_cn/note/api_mapping/pytorch_diff/distributed.gather.md)

## torch.distributed.gather

```python
torch.distributed.gather(
    tensor,
    gather_list=None,
    dst=0,
    group=None,
    async_op=False
)
```

For more information, see [torch.distributed.gather](https://pytorch.org/docs/1.8.1/distributed.html#torch.distributed.gather)。

## mindspore.communication.comm_func.gather_into_tensor

```python
mindspore.communication.comm_func.gather_into_tensor(tensor, dst=0, group=GlobalComm.WORLD_COMM_GROUP)
```

For more information, see [mindspore.communication.comm_func.gather_into_tensor](https://www.mindspore.cn/docs/en/r2.4.0/api_python/communication/mindspore.communication.comm_func.gather_into_tensor.html#mindspore.communication.comm_func.gather_into_tensor)。

## Differences

API function of MindSpore is not consistent with that of PyTorch.

PyTorch：`tensor` will store the gathered result, the tensors to be gatherd `gather_list`, the rank(global rank) `dst` of the process, the communication `group` to work on, and the async op flag `async_op`. The process will gather the tensors and store in `tensor`. The return is a async work handle if `async_op` is True, otherwise is `None`.

MindSpore: The inputs contains the `tensor` to be gathered, the rank(global rank) `dst` of the process, the communication `group` to work on, and it will return the tensor. The dimension 0 of this tensor is equal to sum of the dimension of input tensor, and the other dimension keep the same. The async op flag `async_op` and the working device lists `device_ids` are not supported.

| Categories | Subcategories |PyTorch | MindSpore | Difference |
| --- | --- | --- | --- |---|
|Parameters | Parameters 1 | tensor | tensor |PyTorch：`tensor` will store the gathered result. MindSpore：the tensor to be gathered |
| | Parameters 2 | gather_list | - | PyTorch：the tensors to be gatherd, MindSpore does not have this parameter|
| | Parameters 3 | dst | dst |No difference|
| | Parameters 4 | group | group |No difference|
| | Parameters 5 | async_op | - |PyTorch: the async op flag. MindSpore: does not have this parameter.  |
