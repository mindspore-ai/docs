# Differences with torch.distributed.reduce_scatter

[![View Source On Gitee](https://mindspore-website.obs.cn-north-4.myhuaweicloud.com/website-images/r2.4.0/resource/_static/logo_source_en.svg)](https://gitee.com/mindspore/docs/blob/r2.4.0/docs/mindspore/source_en/note/api_mapping/pytorch_diff/reduce_scatter_tensor.md)

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
import mindspore.communication as comm
return_tensor = comm.comm_func.reduce_scatter_tensor(
    tensor,
    op=ReduceOp.SUM,
    group=None,
    async_op=False
)
```

For more information, see [mindspore.communication.comm_func.reduce_scatter_tensor](https://www.mindspore.cn/docs/en/r2.4.0/api_python/communication/mindspore.communication.comm_func.reduce_scatter_tensor.html#mindspore.communication.comm_func.reduce_scatter_tensor).

## Differences

PyTorch: This interface has four inputs:`output` will hold the scattered result, `input_list` contains a list of tensors to be reduced and scattered. Each process provides the same number of tensors (with the same size).`group` is the communication group and the async op flag `async_op`.  The return is a async work handle if `async_op=True`, otherwise is `None`.

MindSpore: This interface has three inputs and an output, the input tensor `tensor`, the communication group `group` and the async op flag `async_op`. The first dimension can be divided by N(the number of devices in the communication domain), and the rest of the dimensions is the same as the input tensor.

| Class     | Sub-class     |PyTorch | MindSpore | Difference                                                                                                                                          |
|-----------|---------------| --- |-----------|-----------------------------------------------------------------------------------------------------------------------------------------------------|
| Parameters | Parameter 1   | output | -         | PyTorch: the output after reduce_scatter Operation. MindSpore: does not have this parameter.                                                        |
|           | Parameter 2   | input_list | tensor    | PyTorch:input_list contains a list of tensors to be reduced and scattered.  MindSpore: the input tensor which the first dimension can be divided by N. |
|           | Parameter 3   | group | group     | No difference                                                                                                                                       |
|           | Parameter 4   | async_op | async_op    | No difference                                                                                 |
| Returns   | Single return | - | tensor    | PyTorch: does not have a return. MindSpore: returns the output tensor after reduce_scatter_tensor Operation.                                        |
