# Differences with torch.distributed.all_reduce

[![View Source On Gitee](https://mindspore-website.obs.cn-north-4.myhuaweicloud.com/website-images/master/resource/_static/logo_source_en.svg)](https://gitee.com/mindspore/docs/blob/master/docs/mindspore/source_en/note/api_mapping/pytorch_diff/all_reduce.md)

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
from mindspore.communication.comm_func import all_reduce
return_tensor = all_reduce(
    tensor,
    op=ReduceOp.SUM,
    group=GlobalComm.WORLD_COMM_GROUP,
    async_op=False
)
```

For more information, see [mindspore.communication.comm_func.all_reduce](https://www.mindspore.cn/docs/en/master/api_python/communication/mindspore.communication.comm_func.all_reduce.html#mindspore.communication.comm_func.all_reduce).

## Differences

PyTorch: The inputs are the tensor broadcasted by the current process `tensor`, the all_reduce operation `op`, the communication group `group` and the async op flag `async_op`. After the all_reduce operation, the output is written back to `tensor`. The return is a async work handle if `async_op=True`, otherwise is `None`.

MindSpore: The inputs of this interface are the `tensor`, the all_reduce operation `op`, the communication group `group` and the async op flag `async_op`. The output `tensor` has the same shape as input `tensor`, and is generated after the all_reduce operation configured by `op` in the communication group `group`.

| Class      | Sub-class     |PyTorch | MindSpore | Difference                                                                                                                                           |
|------------|---------------| --- |---------|------------------------------------------------------------------------------------------------------------------------------------------------------|
| Parameters | Parameter 1   | tensor | tensor  | PyTorch: the input tensor, and the output is written back to it after all_reduce operation. Mindpore does not write the output result into the input |
|            | Parameter 2   | op | op      | No difference                                                                                                                                        |
|            | Parameter 3   | group | group   | No difference                                                                                                                                        |
|            | Parameter 4   | async_op | async_op       | No difference                                                                                                                                  |
| Returns    | Single return | - | tensor | PyTorch: does not have a return. MindSpore: returns the output tensor after all_reduce operation.                                                    |
