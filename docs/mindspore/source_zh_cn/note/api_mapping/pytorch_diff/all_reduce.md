# 比较与torch.distributed.all_reduce的差异

[![查看源文件](https://mindspore-website.obs.cn-north-4.myhuaweicloud.com/website-images/master/resource/_static/logo_source.svg)](https://gitee.com/mindspore/docs/blob/master/docs/mindspore/source_zh_cn/note/api_mapping/pytorch_diff/all_reduce.md)

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
from mindspore.communication.comm_func import all_reduce
return_tensor = all_reduce(
    tensor,
    op=ReduceOp.SUM,
    group=GlobalComm.WORLD_COMM_GROUP,
    async_op=False
)
```

更多内容详见[mindspore.communication.comm_func.all_reduce](https://www.mindspore.cn/docs/en/master/api_python/communication/mindspore.communication.comm_func.all_reduce.html#mindspore.communication.comm_func.all_reduce)。

## 差异对比

PyTorch：该接口输入tensor、操作类型op、通信域group及异步操作标志async_op，按op指定的操作进行all_reduce操作后，将结果写回tensor。当async_op=True时，返回异步work句柄，否则返回为空。

MindSpore：该接口输入tensor、操作类型op和通信域group及异步操作标志async_op。输出在通信域group中进行op指定的all_reduce操作后得到的tensor，shape与输入tensor一致。

| 分类 | 子类   |PyTorch | MindSpore | 差异                                                        |
|----|------| --- |-----------|-----------------------------------------------------------|
| 参数 | 参数1  | tensor | -         | PyTorch：输入tensor，进行all_reduce操作后将结果写回tensor，MindSpore无此参数 |
|    | 参数2  | op | op        | 一致                                                        |
|    | 参数3  | group | group     | 一致                                                        |
|    | 参数4  | async_op | async_op    | 一致                                                         |
| 返回值 | 单返回值 | - | tensor    | PyTorch：没有返回值。 MindSpore：返回all_reduce操作后返回的张量。            |
