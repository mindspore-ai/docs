# 比较与torch.distributed.new_group的功能差异

<a href="https://gitee.com/mindspore/docs/blob/master/docs/mindspore/migration_guide/source_zh_cn/api_mapping/pytorch_diff/create_group.md" target="_blank"><img src="https://mindspore-website.obs.cn-north-4.myhuaweicloud.com/website-images/master/resource/_static/logo_source.png"></a>

## torch.distributed.new_group

```python
torch.distributed.new_group(
    ranks=None,
    timeout=datetime.timedelta(0, 1800),
    backend=None
)
```

更多内容详见[torch.distributed.new_group](https://pytorch.org/docs/1.5.0/distributed.html#torch.distributed.new_group)。

## mindspore.communication.create_group

```python
mindspore.communication.create_group(group, rank_ids)
```

更多内容详见[mindspore.communication.create_group](https://mindspore.cn/docs/api/zh-CN/master/api_python/mindspore.communication.html#mindspore.communication.create_group)。

## 使用方式

PyTorch：该接口传入待构建通信域rank列表，指定backend创建指定的通信域，并返回创建的通信域。

MindSpore：该接口传入group名字，以及待构建通信域rank列表，创建一个以传入的group名字为key的通信域，不返回任何值。