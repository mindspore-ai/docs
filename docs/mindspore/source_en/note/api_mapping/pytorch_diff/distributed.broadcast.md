# Differences with torch.distributed.broadcast

[![View Source On Gitee](https://mindspore-website.obs.cn-north-4.myhuaweicloud.com/website-images/master/resource/_static/logo_source_en.svg)](https://gitee.com/mindspore/docs/blob/master/docs/mindspore/source_en/note/api_mapping/pytorch_diff/distributed.broadcast.md)

## torch.distributed.broadcast

```python
torch.distributed.broadcast(
    tensor,
    src=0,
    group=None,
    async_op=False
)
```

For more information, see [torch.distributed.broadcast](https://pytorch.org/docs/1.8.1/distributed.html#torch.distributed.broadcast)。

## mindspore.communication.comm_func.broadcast

```python
mindspore.communication.comm_func.broadcast(tensor, src=0, group=GlobalComm.WORLD_COMM_GROUP)
```

For more information, see [mindspore.communication.comm_func.broadcast](https://www.mindspore.cn/docs/en/master/api_python/communication/mindspore.communication.comm_func.broadcast.html#mindspore.communication.comm_func.broadcast)。

## Differences

API function of MindSpore is not consistent with that of PyTorch.

PyTorch：the inputs contains the `tensor` to be broadcast or received, the rank(global rank) `src` of the process that broadcast the tensor, the communication `group` to work on, and the async op flag `async_op`. The process will broadcast the tensor if it is the `src` process, otherwise it will receive the Tensor. The return is a async work handle if `async_op=True`, otherwise is `None`.

MindSpore: The inputs contains the `tensor` to be broadcast, the rank(global rank) `src` of the process that broadcast the tensor, the communication `group` to work on, and it will return the tensor with the same shape with the broadcasted tensor. The async op flag `async_op` and the working device lists `device_ids` are not supported.

| Categories | Subcategories |PyTorch | MindSpore | Difference |
| --- | --- | --- | --- |---|
|Parameters | Parameters 1 | tensor | tensor |PyTorch：tensor to be broadcast in `src` process，otherwise to be received. MindSpore：tensor to be broadcasted |
| | Parameters 2 | src | src |No difference|
| | Parameters 3 | group | group |No difference|
| | Parameters 4 | async_op | - |PyTorch: the async op flag. MindSpore: does not have this parameter. |
