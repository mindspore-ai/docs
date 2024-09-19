# Differences with torch.distributed.reduce

[![View Source On Gitee](https://mindspore-website.obs.cn-north-4.myhuaweicloud.com/website-images/master/resource/_static/logo_source_en.svg)](https://gitee.com/mindspore/docs/blob/master/docs/mindspore/source_en/note/api_mapping/pytorch_diff/reduce.md)

## torch.distributed.reduce

```python
torch.distributed.reduce(
    tensor,
    dst,
    op=ReduceOp.SUM,
    group=None,
    async_op=False
)
```

For more information, see [torch.distributed.reduce](https://pytorch.org/docs/1.8.1/distributed.html#torch.distributed.reduce).

## mindspore.communication.comm_func.reduce

```python
from mindspore.communication.comm_func import reduce
return_tensor = reduce(
    tensor,
    dst,
    op=ReduceOp.SUM,
    group=None,
)
```

For more information, see [mindspore.communication.comm_func.reduce](https://www.mindspore.cn/docs/en/master/api_python/communication/mindspore.communication.comm_func.reduce.html#mindspore.communication.comm_func.reduce).

## Differences

PyTorch: This interface has five inputs:`tensor` will store the reduced result of the destination process `dst` and all tensors must have the same shape.  Only the `dst` process will store the reduced result, other processes’ tensors remain unchanged.`group` is the communication group and the async op flag `async_op`.  The return is a async work handle if `async_op=True`, otherwise is `None`.

MindSpore: This interface has four inputs and an output, the input `tensor` of each process must have the same shape and will not store the reduced result, and the communication group `group`  and `op` are same as pytorch. This interface currently does not support the configuration of `async_op`. The return `output` is a tensor which store the reduced result of the destination process.Only the output of `dst` process will store the reduced result, other processes’ tensors remain a zero tensor with the same shape of input tensor.

| Class      | Sub-class     | PyTorch  | MindSpore | Difference                                                                                                                                                                  |
|------------|---------------|----------|-----------|-----------------------------------------------------------------------------------------------------------------------------------------------------------------------------|
| Parameters | Parameter 1   | tensor   | tensor    | PyTorch: the input tensor will store the reduced result of the destination process. MindSpore: the input tensor will not store the reduced result of the destination process |
|            | Parameter 2   | dst      | dst       | No difference                                                                                                                                                               |
|            | Parameter 3   | op       | op        | No difference                                                                                                                                                               |
|            | Parameter 4   | group | group     | No difference                                                                                                                                                               |
|            | Parameter 5   | async_op | -         | PyTorch: the async op flag. MindSpore: does not have this parameter.                                                                                                        |
| Returns    | Single return | -| tensor    | PyTorch: does not have a return. MindSpore: returns the tensor after reduce operation.                                                                              |
