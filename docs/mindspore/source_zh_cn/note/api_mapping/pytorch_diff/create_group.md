# 比较与torch.distributed.new_group的功能差异

<a href="https://gitee.com/mindspore/docs/blob/master/docs/mindspore/source_zh_cn/note/api_mapping/pytorch_diff/create_group.md" target="_blank"><img src="https://mindspore-website.obs.cn-north-4.myhuaweicloud.com/website-images/master/resource/_static/logo_source.png"></a>

## torch.distributed.new_group

```python
torch.distributed.new_group(
    ranks=None,
    timeout=datetime.timedelta(0, 1800),
    backend=None
)
```

更多内容详见[torch.distributed.new_group](https://pytorch.org/docs/1.8.1/distributed.html#torch.distributed.new_group)。

## mindspore.communication.create_group

```python
mindspore.communication.create_group(group, rank_ids)
```

更多内容详见[mindspore.communication.create_group](https://mindspore.cn/docs/zh-CN/master/api_python/mindspore.communication.html#mindspore.communication.create_group)。

## 使用方式

PyTorch：该接口传入待构建通信域rank列表，指定backend创建指定的通信域，并返回创建的通信域。

MindSpore：该接口传入group名字，以及待构建通信域rank列表，创建一个以传入的group名字为key的通信域，不返回任何值。

| 分类 | 子类 |PyTorch | MindSpore | 差异 |
| --- | --- | --- | --- |---|
|参数 | 参数1 | - | group | PyTorch无此参数，MindSpore：group名字|
| | 参数2 | ranks | rank_ids | 功能一致，参数名有差异|
| | 参数3 | timeout | - |PyTorch：超时阈值，MindSpore无此参数且需在调用该接口之前配置相应环境变量|
| | 参数4 | backend | - |PyTorch：集合通信后端，MindSpore无此参数且需在调用该接口之前配置相应环境变量 |