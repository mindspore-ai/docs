# 比较与torch.distributed.barrier的差异

[![查看源文件](https://mindspore-website.obs.cn-north-4.myhuaweicloud.com/website-images/r2.4.10/resource/_static/logo_source.svg)](https://gitee.com/mindspore/docs/blob/r2.4.10/docs/mindspore/source_zh_cn/note/api_mapping/pytorch_diff/distributed.barrier.md)

## torch.distributed.barrier

```python
torch.distributed.barrier(
    group=None,
    async_op=False,
    device_ids=None
)
```

更多内容详见[torch.distributed.barrier](https://pytorch.org/docs/1.8.1/distributed.html#torch.distributed.barrier)。

## mindspore.communication.comm_func.barrier

```python
mindspore.communication.comm_func.barrier(group=GlobalComm.WORLD_COMM_GROUP)
```

更多内容详见[mindspore.communication.comm_func.barrier](https://www.mindspore.cn/docs/zh-CN/r2.4.10/api_python/communication/mindspore.communication.comm_func.barrier.html#mindspore.communication.comm_func.barrier)。

## 差异对比

PyTorch：该接口输入通信域group，异步操作标志async_op，以及生效的设备id列表。当async_op=True时，返回异步work句柄，否则返回为空。

MindSpore：该接口输入通信域group。当前该接口不支持async_op以及device_ids的配置。

| 分类 | 子类 |PyTorch | MindSpore | 差异 |
| --- | --- | --- | --- |---|
|参数 | 参数1 | group | group | - |
| | 参数2 | async_op | - |PyTorch：异步操作标志，MindSpore无此参数 |
| | 参数3 | device_ids | - |PyTorch：生效的设备id列表，MindSpore无此参数 |
