# Differences with torch.distributed.reduce_scatter

[![View Source On Gitee](https://mindspore-website.obs.cn-north-4.myhuaweicloud.com/website-images/master/resource/_static/logo_source_en.svg)](https://gitee.com/mindspore/docs/blob/master/docs/mindspore/source_en/note/api_mapping/pytorch_diff/reduce_scatter_tensor.md)

## torch.distributed.reduce_scatter

```python
torch.distributed.reduce_scatter(
    output,
    input_list,
    op=ReduceOp.SUM,
    group=None,
    async_op=False
)
```

For more information, see [torch.distributed.reduce_scatter](https://pytorch.org/docs/1.8.1/distributed.html#torch.distributed.reduce_scatter).

## mindspore.communication.comm_func.reduce_scatter_tensor

```python
mindspore.communication.comm_func.reduce_scatter_tensor(
    tensor,
    op=ReduceOp.SUM,
    group=None,
    async_op=False
)
```

For more information, see [mindspore.communication.comm_func.reduce_scatter_tensor](https://www.mindspore.cn/docs/en/master/api_python/communication/mindspore.communication.comm_func.reduce_scatter_tensor.html#mindspore.communication.comm_func.reduce_scatter_tensor).

## Differences

PyTorch: The inputs of this interface are the `output` which will hold the scattered result, `input_list` contains a list of tensors to be reduced and scattered. Each process provides the same number of tensors (with the same size).`group` is the communication group and the async op flag `async_op`. The return is an async work handle if `async_op=True`, otherwise is `None`.

MindSpore: The inputs of this interface are the input tensor `tensor`, the communication group `group` and the async op flag `async_op`. The first dimension of the input `tensor` can be divided by N(the number of devices in the communication domain). It returns an async work handle if `async_op=True`, otherwise is `None`.

| Class     | Sub-class     | PyTorch           | MindSpore                 | Difference                                                                                                                                                                                                                                                                                                          |
|-----------|---------------|-------------------|---------------------------|---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|
| Parameters | Parameter 1   | output            | -                         | PyTorch: the output after reduce_scatter Operation. MindSpore: does not have this parameter.                                                                                                                                                                                                                        |
|           | Parameter 2   | input_list        | tensor                    | PyTorch:input_list contains a list of tensors to be reduced and scattered.  MindSpore: the input tensor which the first dimension can be divided by N.                                                                                                                                                              |
|           | Parameter 3   | op                | op                        | No difference                                                                                                                                                              |
|           | Parameter 4   | group             | group                     | No difference                                                                                                                                                                                                                                                                                                       |
|           | Parameter 5   | async_op          | async_op                  | No difference                                                                                                                                                                                                                                                                                                       |
| Returns   | Single return | async_work_handle | tuple(tensor, CommHandle) | PyTorch: An async work handle, if async_op is set to True. None, if not async_op or if not part of the group.</br> MindSpore: returns a tuple. The tuple contains an output tensor after reduce_scatter_tensor operation and an async work handle `CommHandle`. When `async_op` is False, the `CommHandle` will be None. |
