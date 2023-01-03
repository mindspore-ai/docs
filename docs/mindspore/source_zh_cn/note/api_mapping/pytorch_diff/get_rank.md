# 比较与torch.distributed.get_rank的功能差异

<a href="https://gitee.com/mindspore/docs/blob/r2.0.0-alpha/docs/mindspore/source_zh_cn/note/api_mapping/pytorch_diff/get_rank.md" target="_blank"><img src="https://mindspore-website.obs.cn-north-4.myhuaweicloud.com/website-images/master/resource/_static/logo_source.png"></a>

## torch.distributed.get_rank

```python
torch.distributed.get_rank(group=None)
```

更多内容详见[torch.distributed.get_rank](https://pytorch.org/docs/1.8.1/distributed.html#torch.distributed.get_rank)。

## mindspore.communication.get_rank

```python
mindspore.communication.get_rank(group=GlobalComm.WORLD_COMM_GROUP)
```

更多内容详见[mindspore.communication.get_rank](https://mindspore.cn/docs/zh-CN/r2.0.0-alpha/api_python/mindspore.communication.html#mindspore.communication.get_rank)。

## 使用方式

PyTorch：该接口输入当前通信组group，输出为调用该接口进程的对应rank，当前进程不在group中时返回-1。

MindSpore：该接口输入当前通信组group，输出为调用该接口进程的对应rank。由于get_rank方法应该在init方法之后使用，故正常情况下不会出现当前进程不在group中的情况；若get_rank在init前使用，会抛出error。

| 分类 | 子类 |PyTorch | MindSpore | 差异 |
| --- | --- | --- | --- |---|
|参数  | 参数1 | group | group |一致|
