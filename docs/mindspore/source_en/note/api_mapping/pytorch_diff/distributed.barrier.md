# Differences with torch.distributed.barrier

[![View Source On Gitee](https://mindspore-website.obs.cn-north-4.myhuaweicloud.com/website-images/master/resource/_static/logo_source_en.svg)](https://gitee.com/mindspore/docs/blob/master/docs/mindspore/source_en/note/api_mapping/pytorch_diff/distributed.barrier.md)

## torch.distributed.barrier

```python
torch.distributed.barrier(
    group=None,
    async_op=False,
    device_ids=None
)
```

For more information, see [torch.distributed.barrier](https://pytorch.org/docs/1.8.1/distributed.html#torch.distributed.barrier)。

## mindspore.communication.comm_func.barrier

```python
mindspore.communication.comm_func.barrier(group=GlobalComm.WORLD_COMM_GROUP)
```

For more information, see [mindspore.communication.comm_func.barrier](https://www.mindspore.cn/docs/en/master/api_python/communication/mindspore.communication.comm_func.barrier.html#mindspore.communication.comm_func.barrier)。

## Differences

API function of MindSpore is not consistent with that of PyTorch.

PyTorch: The inputs contains the communication group `group`, the async op flag `async_op`. And the working device lists `device_ids`. The return is a async work handle if `async_op=True`, otherwise is `None`.

MindSpore：The inputs contains the communication group `group`. The async op flag `async_op` and the working device lists `device_ids` are not supported.

| Categories | Subcategories |PyTorch | MindSpore | Difference |
| --- | --- | --- | --- |---|
|Parameters | Parameter 1 | group | group | No difference |
| | Parameter 2 | async_op | - |PyTorch: the async op flag. MindSpore: does not have this parameter. |
| | Parameter 3 | device_ids | - |PyTorch： the working device lists. MindSpore: does not have this parameter. |
