# Differences with torch.distributed.all_reduce

[![View Source On Gitee](https://mindspore-website.obs.cn-north-4.myhuaweicloud.com/website-images/r2.3/resource/_static/logo_source_en.svg)](https://gitee.com/mindspore/docs/blob/r2.3/docs/mindspore/source_en/note/api_mapping/pytorch_diff/all_reduce.md)

## torch.distributed.all_reduce

```python
torch.distributed.all_reduce(
    tensor,
    op=<ReduceOp.SUM: 0>,
    group=None,
    async_op=False
)
```

For more information, see [torch.distributed.all_reduce](https://pytorch.org/docs/1.8.1/distributed.html#torch.distributed.all_reduce).

## mindspore.ops.AllReduce

```python
class mindspore.ops.AllReduce(
    op=ReduceOp.SUM,
    group=GlobalComm.WORLD_COMM_GROUP
)(input_x)
```

For more information, see [mindspore.ops.AllReduce](https://mindspore.cn/docs/en/r2.3/api_python/ops/mindspore.ops.AllReduce.html#mindspore.ops.AllReduce).

## Differences

PyTorch: The inputs are the tensor broadcasted by the current process `tensor`, the AllReduce operation `op`, the communication group `group` and the async op flag `async_op`. After the AllReduce operation, the output is written back to `tensor`. The return is a async work handle if `async_op=True`, otherwise is `None`.

MindSpore: The input of this interface is `input_x` that is a `tensor`. The output `tensor` has the same shape as `input_x`, and is generated after the AllReduce operation configured by `op` in the communication group `group`. This interface currently not support the configuration of `async_op`.

| Class | Sub-class |PyTorch | MindSpore | Difference |
| --- | --- | --- | --- |---|
|Parameters | Parameter 1 | tensor | - |PyTorch: the input tensor, and the output is written back to it after AllReduce operation. MindSpore does not have this parameter|
| | Parameter 2 | op | op |No difference|
| | Parameter 3 | group | group |No difference|
| | Parameter 4 | async_op | - |PyTorch: the async op flag. MindSpore does not have this parameter|
| Input | Single input | - | input_x | PyTorch: not applied. MindSpore: the input tensor of AllReduce. |
