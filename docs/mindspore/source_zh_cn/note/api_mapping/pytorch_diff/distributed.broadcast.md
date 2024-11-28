# 比较与torch.distributed.broadcast的差异

[![查看源文件](https://mindspore-website.obs.cn-north-4.myhuaweicloud.com/website-images/r2.4.1/resource/_static/logo_source.svg)](https://gitee.com/mindspore/docs/blob/r2.4.1/docs/mindspore/source_zh_cn/note/api_mapping/pytorch_diff/distributed.broadcast.md)

## torch.distributed.broadcast

```python
torch.distributed.broadcast(
    tensor,
    src=0,
    group=None,
    async_op=False
)
```

更多内容详见[torch.distributed.broadcast](https://pytorch.org/docs/1.8.1/distributed.html#torch.distributed.broadcast)。

## mindspore.communication.comm_func.broadcast

```python
mindspore.communication.comm_func.broadcast(tensor, src=0, group=GlobalComm.WORLD_COMM_GROUP)
```

更多内容详见[mindspore.communication.comm_func.broadcast](https://www.mindspore.cn/docs/zh-CN/r2.4.1/api_python/communication/mindspore.communication.comm_func.broadcast.html#mindspore.communication.comm_func.broadcast)。

## 差异对比

PyTorch：该接口输入将被广播或用于接收的`Tensor`、源进程编号src、通信域group及异步操作标志async_op。若该进程为源进程，则tensor为待广播的Tensor；否则为用于接受数据的Tensor。当async_op=True时，返回异步work句柄，否则返回为空。

MindSpore：该接口输入待广播的tensor，源进程编号src，通信域group，输出tensor，维度与广播的tensor一致。当前该接口不支持async_op的配置。

| 分类 | 子类 |PyTorch | MindSpore | 差异 |
| --- | --- | --- | --- |---|
|参数 | 参数1 | tensor | tensor |PyTorch：目标进程为待广播Tensor，其他进程为广播后的输出，MindSpore：待广播的tensor |
| | 参数2 | src | src |-|
| | 参数3 | group | group |-|
| | 参数4 | async_op | - |PyTorch：异步操作标志，MindSpore无此参数 |
