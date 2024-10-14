# 比较与torch.distributed.scatter的差异

[![查看源文件](https://mindspore-website.obs.cn-north-4.myhuaweicloud.com/website-images/r2.4.0/resource/_static/logo_source.svg)](https://gitee.com/mindspore/docs/blob/r2.4.0/docs/mindspore/source_zh_cn/note/api_mapping/pytorch_diff/distributed.scatter.md)

## torch.distributed.scatter

```python
torch.distributed.scatter(
    tensor,
    scatter_list=None,
    src=0,
    group=None,
    async_op=False
)
```

更多内容详见[torch.distributed.scatter](https://pytorch.org/docs/1.8.1/distributed.html#torch.distributed.scatter)。

## mindspore.communication.comm_func.scatter_tensor

```python
mindspore.communication.comm_func.scatter_tensor(tensor, src=0, group=GlobalComm.WORLD_COMM_GROUP)
```

更多内容详见[mindspore.communication.comm_func.scatter_tensor](https://www.mindspore.cn/docs/zh-CN/r2.4.0/api_python/communication/mindspore.communication.comm_func.scatter_tensor.html#mindspore.communication.comm_func.scatter_tensor)。

## 差异对比

PyTorch：该接口输入当前进程的tensor、散射列表scatter_list、发送源的进程编号src、通信域group及异步操作标志async_op，进行scatter操作后输出tensor，类型为Tensor。当async_op=True时，返回异步work句柄，否则返回为空。

MindSpore：该接口输入待散射的tensor，发送源的进程编号src，通信域group，输出tensor，第一维等于输入数据第0维除以src，其余维度与输入tensor一致。当前该接口不支持async_op的配置。

| 分类 | 子类 |PyTorch | MindSpore | 差异 |
| --- | --- | --- | --- |---|
|参数 | 参数1 | tensor | tensor |PyTorch：进行scatter操作后的输出，MindSpore：待散射的tensor |
| | 参数2 | scatter_list | - | PyTorch：待散射tensor列表，MindSpore无此参数|
| | 参数3 | src | src |-|
| | 参数4 | group | group |-|
| | 参数5 | async_op | - |PyTorch：异步操作标志，MindSpore无此参数 |
