# Differences with torch.distributed.all_reduce

[![View Source On Gitee](https://mindspore-website.obs.cn-north-4.myhuaweicloud.com/website-images/r2.4.0/resource/_static/logo_source_en.svg)](https://gitee.com/mindspore/docs/blob/r2.4.0/docs/mindspore/source_en/note/api_mapping/pytorch_diff/all_reduce.md)

## torch.distributed.all_reduce

```python
torch.distributed.all_reduce(
    tensor,
    op=ReduceOp.SUM,
    group=None,
    async_op=False
)
```

For more information, see [torch.distributed.all_reduce](https://pytorch.org/docs/1.8.1/distributed.html#torch.distributed.all_reduce).

## mindspore.communication.comm_func.all_reduce

```python
mindspore.communication.comm_func.all_reduce(
    tensor,
    op=ReduceOp.SUM,
    group=GlobalComm.WORLD_COMM_GROUP,
    async_op=False
)
```

For more information, see [mindspore.communication.comm_func.all_reduce](https://www.mindspore.cn/docs/en/r2.4.0/api_python/communication/mindspore.communication.comm_func.all_reduce.html#mindspore.communication.comm_func.all_reduce).

## Differences

PyTorch: The inputs of this interface are the `tensor`, the all_reduce operation `op`, the communication group `group` and the async op flag `async_op`. After the all_reduce operation specified by op, Pytorch writes the result back to the input `tensor`. The return is an async work handle if `async_op=True`, otherwise is `None`.

MindSpore: The inputs of this interface are the `tensor`, the all_reduce operation `op`, the communication group `group` and the async op flag `async_op`. After the all_reduce operation specified by op, Mindspore returns the result `tensor` which has the same shape as the input `tensor` and an async work handle if `async_op=True`, otherwise is `None`.

| Class      | Sub-class     |PyTorch | MindSpore | Difference                                                                                                                                                                                                                                                                            |
|------------|---------------| --- |---------|---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|
| Parameters | Parameter 1   | tensor | tensor  | PyTorch: the input tensor, and the output is written back to it after all_reduce operation. Mindpore does not write the output result into the input tensor.                                                                                                                          |
|            | Parameter 2   | op | op      | No difference                                                                                                                                                                                                                                                                         |
|            | Parameter 3   | group | group   | No difference                                                                                                                                                                                                                                                                         |
|            | Parameter 4   | async_op | async_op       | No difference                                                                                                                                                                                                                                                                         |
| Returns    | Single return | async_work_handle | tuple(tensor, CommHandle) | PyTorch: An async work handle is returned, if async_op is set to True. None, if not async_op or if not part of the group.</br> MindSpore: returns a tuple.The tuple contains an output tensor after all_reduce operation and a CommHandle. When `async_op` is False, the CommHandle will be None. |
