# 比较与torch.distributed.all_reduce的功能差异

<a href="https://gitee.com/mindspore/docs/blob/r2.0.0-alpha/docs/mindspore/source_zh_cn/note/api_mapping/pytorch_diff/all_reduce.md" target="_blank"><img src="https://mindspore-website.obs.cn-north-4.myhuaweicloud.com/website-images/r2.0.0-alpha/resource/_static/logo_source.png"></a>

## torch.distributed.all_reduce

```python
torch.distributed.all_reduce(
    tensor,
    op=<ReduceOp.SUM: 0>,
    group=None,
    async_op=False
)
```

更多内容详见[torch.distributed.all_reduce](https://pytorch.org/docs/1.8.1/distributed.html#torch.distributed.all_reduce)。

## mindspore.ops.AllReduce

```python
class mindspore.ops.AllReduce(
    op=ReduceOp.SUM,
    group=GlobalComm.WORLD_COMM_GROUP
)(input_x)
```

更多内容详见[mindspore.ops.AllReduce](https://mindspore.cn/docs/zh-CN/r2.0.0-alpha/api_python/ops/mindspore.ops.AllReduce.html#mindspore.ops.AllReduce)。

## 使用方式

PyTorch：该接口输入tensor、操作类型op、通信域group及异步操作标志async_op，按op指定的操作进行AllReduce操作后，将结果写回tensor。当async_op=True时，返回异步work句柄，否则返回为空。

MindSpore：该接口输入tensor input_x，输出在通信域group中进行op指定的AllGather操作后得到的tensor，shape与输入tensor一致。当前该接口不支持async_op的配置。

| 分类 | 子类 |PyTorch | MindSpore | 差异 |
| --- | --- | --- | --- |---|
|参数 | 参数1 | tensor | - |PyTorch：输入tensor，进行AllReduce操作后将结果写回tensor，MindSpore无此参数 |
| | 参数2 | op | op | 一致|
| | 参数3 | group | group |一致|
| | 参数4 | async_op | - |PyTorch：异步操作标志，MindSpore无此参数 |
