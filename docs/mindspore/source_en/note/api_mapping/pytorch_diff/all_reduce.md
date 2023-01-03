# Function Differences with torch.distributed.all_reduce

<a href="https://gitee.com/mindspore/docs/blob/r2.0.0-alpha/docs/mindspore/source_en/note/api_mapping/pytorch_diff/all_reduce.md" target="_blank"><img src="https://mindspore-website.obs.cn-north-4.myhuaweicloud.com/website-images/master/resource/_static/logo_source_en.png"></a>

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
mindspore.ops.AllReduce(
    op=ReduceOp.SUM,
    group=GlobalComm.WORLD_COMM_GROUP
)(input_x)
```

For more information, see [mindspore.ops.AllReduce](https://mindspore.cn/docs/en/r2.0.0-alpha/api_python/ops/mindspore.ops.AllReduce.html#mindspore.ops.AllReduce).

## Differences

PyTorch: The inputs are the tensor broadcasted by the current process `tensor`, the AllReduce operation `op`, the communication group `group` and the async op flag `async_op`. After the AllReduce operation, the output is written back to `tensor`. The return is a async work handle if `async_op=True`, otherwise is `None`.

MindSpore: The input of this interface is `input_x` that is a `tensor`. The output `tensor` has the same shape as `input_x`, after the AllReduce operation configured by `op` in the communication group `group`. This interface currently not support the configuration of `async_op`.

| Class | Sub-class |PyTorch | MindSpore | Difference |
| --- | --- | --- | --- |---|
|Param | Param 1 | tensor | - |PyTorch: the input tensor, and the output is written back to it. MindSpore does not have this parameter|
| | Param 2 | op | op |No difference|
| | Param 3 | group | group |No difference|
| | Param 4 | async_op | - |PyTorch: the async op flag. MindSpore does not have this parameter|
