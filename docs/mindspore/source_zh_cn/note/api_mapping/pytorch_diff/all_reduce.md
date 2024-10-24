# 比较与torch.distributed.all_reduce的差异

[![查看源文件](https://mindspore-website.obs.cn-north-4.myhuaweicloud.com/website-images/r2.4.0/resource/_static/logo_source.svg)](https://gitee.com/mindspore/docs/blob/r2.4.0/docs/mindspore/source_zh_cn/note/api_mapping/pytorch_diff/all_reduce.md)

## torch.distributed.all_reduce

```python
torch.distributed.all_reduce(
    tensor,
    op=ReduceOp.SUM,
    group=None,
    async_op=False
)
```

更多内容详见[torch.distributed.all_reduce](https://pytorch.org/docs/1.8.1/distributed.html#torch.distributed.all_reduce)。

## mindspore.communication.comm_func.all_reduce

```python
import mindspore.communication as comm
return_tensor = comm.comm_func.all_reduce(
    tensor,
    op=ReduceOp.SUM,
    group=GlobalComm.WORLD_COMM_GROUP,
    async_op=False
)
```

更多内容详见[mindspore.communication.comm_func.all_reduce](https://www.mindspore.cn/docs/zh-CN/r2.4.0/api_python/communication/mindspore.communication.comm_func.all_reduce.html#mindspore.communication.comm_func.all_reduce)。

## 差异对比

PyTorch：该接口输入tensor、操作类型op、通信域group及异步操作标志async_op，按op指定的操作进行all_reduce操作后，PyTorch将结果写回输入`tensor`。当async_op=True时，返回异步work句柄，否则返回为空。

MindSpore：该接口输入tensor、操作类型op和通信域group及异步操作标志async_op。按op指定的操作进行all_reduce操作后，MindSpore返回结果`tensor`，其shape与输入`tensor`一致。当async_op=True时，返回异步work句柄，否则返回为空。

| 分类 | 子类   |PyTorch | MindSpore | 差异                                                                                                                                                            |
|----|------| --- |-----------|---------------------------------------------------------------------------------------------------------------------------------------------------------------|
| 参数 | 参数1  | tensor | -         | PyTorch：输入tensor，进行all_reduce操作后将结果写回tensor，MindSpore无此参数                                                                                                     |
|    | 参数2  | op | op        | 一致                                                                                                                                                            |
|    | 参数3  | group | group     | 一致                                                                                                                                                            |
|    | 参数4  | async_op | async_op  | 一致                                                                                                                                                            |
| 返回值 | 单返回值 | async_work_handle | tuple(tensor, CommHandle)   | PyTorch：如果`async_op`设置为 True，返回异步工作句柄。如果`async_op`为False或不是组的一部分，则为None。</br> MindSpore：返回一个元组。该元组包含了一个all_reduce操作后返回的张量以及一个异步返回句柄。当`async_op`为False，该异步返回句柄为None。 |
