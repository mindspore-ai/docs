# Function Differences with torch.distributed.all_gather

[![View Source On Gitee](https://mindspore-website.obs.cn-north-4.myhuaweicloud.com/website-images/r2.0/resource/_static/logo_source_en.png)](https://gitee.com/mindspore/docs/blob/r2.0/docs/mindspore/source_en/note/api_mapping/pytorch_diff/all_gather.md)

## torch.distributed.all_gather

```python
torch.distributed.all_gather(
    tensor_list,
    tensor,
    group=None,
    async_op=False
)
```

For more information, see [torch.distributed.all_gather](https://pytorch.org/docs/1.8.1/distributed.html#torch.distributed.all_gather).

## mindspore.ops.AllGather

```python
class mindspore.ops.AllGather(group=GlobalComm.WORLD_COMM_GROUP)(input_x)
```

For more information, see [mindspore.ops.AllGather](https://mindspore.cn/docs/en/r2.0/api_python/ops/mindspore.ops.AllGather.html#mindspore.ops.AllGather).

## Differences

PyTorch: The inputs are the tensor broadcasted by the current process `tensor`, the communication group `group` and the async op flag `async_op`. The output is `tensor_list` after AllGather op, whose type is `list[Tensor]` and the length is the number of devices in the communication group. The return is a async work handle if `async_op=True`, otherwise is `None`.

MindSpore: The input of this interface is `input_x` that is a `tensor`. The output is a `tensor`, whose first dimension equals the number of devices in the communication group. This interface currently does not support the configuration of `async_op`.

| Class | Sub-class |PyTorch | MindSpore | Difference |
| --- | --- | --- | --- |---|
|Param | Param 1 | tensor_list | - |PyTorch: the output after AllGather. MindSpore does not have this parameter|
| | Param 2 | tensor | - |PyTorch: the tensor broadcasted by the current process. MindSpore does not have this parameter |
| | Param 3 | group | group |No dfference|
| | Param 4 | async_op | - |PyTorch: the async op flag. MindSpore does not have this parameter|
| Input | Single input | - | input_x | PyTorch: not applied. MindSpore: the input tensor of AllGather. |
