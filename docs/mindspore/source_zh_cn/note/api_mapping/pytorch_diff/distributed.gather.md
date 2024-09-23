# 比较与torch.distributed.gather的差异

[![查看源文件](https://mindspore-website.obs.cn-north-4.myhuaweicloud.com/website-images/master/resource/_static/logo_source.svg)](https://gitee.com/mindspore/docs/blob/master/docs/mindspore/source_zh_cn/note/api_mapping/pytorch_diff/distributed.gather.md)

## torch.distributed.gather

```python
torch.distributed.gather(
    tensor,
    gather_list=None,
    dst=0,
    group=None,
    async_op=False
)
```

更多内容详见[torch.distributed.gather](https://pytorch.org/docs/1.8.1/distributed.html#torch.distributed.gather)。

## mindspore.communication.comm_func.gather_into_tensor

```python
mindspore.communication.comm_func.gather_into_tensor(tensor, dst=0, group=GlobalComm.WORLD_COMM_GROUP)
```

更多内容详见[mindspore.communication.comm_func.gather_into_tensor](https://www.mindspore.cn/docs/zh-CN/master/api_python/communication/mindspore.communication.comm_func.gather_into_tensor.html#mindspore.communication.comm_func.gather_into_tensor)。

## 差异对比

PyTorch：该接口输入当前进程的`tensor`、聚合列表`gather_list`、目标的进程编号`dst`、通信域`group`及异步操作标志`async_op`，进行gather操作后结果保存在`tensor`中。当`async_op`为True时，返回异步work句柄，否则返回为空。

MindSpore：该接口输入待聚合的tensor，目标进程编号dst，通信域group，输出tensor，第一维等于输入数据第0维求和，其余维度与输入tensor一致。当前该接口不支持async_op的配置。

| 分类 | 子类 |PyTorch | MindSpore | 差异 |
| --- | --- | --- | --- |---|
|参数 | 参数1 | tensor | tensor |PyTorch：进行gather操作后的输出，MindSpore：待聚合的tensor |
| | 参数2 | gather_list | - | PyTorch：待聚合tensor列表，MindSpore无此参数|
| | 参数3 | dst | dst |-|
| | 参数4 | group | group |-|
| | 参数5 | async_op | - |PyTorch：异步操作标志，MindSpore无此参数 |
