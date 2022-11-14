# 比较与torch.distributed.all_gather的功能差异

<a href="https://gitee.com/mindspore/docs/blob/master/docs/mindspore/source_zh_cn/note/api_mapping/pytorch_diff/all_gather.md" target="_blank"><img src="https://mindspore-website.obs.cn-north-4.myhuaweicloud.com/website-images/master/resource/_static/logo_source.png"></a>

## torch.distributed.all_gather

```python
torch.distributed.all_gather(
    tensor_list,
    tensor,
    group,
    async_op=False
)
```

更多内容详见[torch.distributed.all_gather](https://pytorch.org/docs/1.8.1/distributed.html#torch.distributed.all_gather)。

## mindspore.ops.AllGather

```python
mindspore.ops.AllGather(group)
```

更多内容详见[mindspore.ops.AllGather](https://mindspore.cn/docs/zh-CN/master/api_python/ops/mindspore.ops.AllGather.html#mindspore.ops.AllGather)。

## 使用方式

PyTorch：该接口输入当前进程广播的tensor、通信域group及异步操作标志async_op，进行AllGather操作后输出tensor_list，类型为list[Tensor]，长度为通信域中设备数量N。当async_op=True时，返回异步work句柄，否则返回为空。

MindSpore：该接口输出为tensor，第一维为通信域中设备数量N，其余维度与输入tensor一致，而不是像PyTorch对应接口输出list[Tensor]。当前该接口不支持async_op的配置。

| 分类 | 子类 |PyTorch | MindSpore | 差异 |
| --- | --- | --- | --- |---|
|参数 | 参数1 | tensor_list | - |PyTorch:进行AllGather操作后的输出，MindSpore无此参数 |
| | 参数2 | tensor | - | PyTorch:当前进程广播的tensor，MindSpore无此参数|
| | 参数3 | group | group |一致|
| | 参数4 | async_op | - |PyTorch：异步操作标志，MindSpore无此参数 |