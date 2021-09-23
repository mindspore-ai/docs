# 比较与torch.distributed.new_group的功能差异

<a href="https://gitee.com/mindspore/docs/blob/r1.5/docs/mindspore/migration_guide/source_zh_cn/api_mapping/pytorch_diff/create_group.md" target="_blank"><img src="https://gitee.com/mindspore/docs/raw/r1.5/resource/_static/logo_source.png"></a>

## torch.distributed.new_group

```python
torch.distributed.new_group(
    ranks=None,
    timeout=datetime.timedelta(0, 1800),
    backend=None,
    pg_options=None
)
```

## mindspore.communication.create_group

```python
mindspore.communication.create_group(group, rank_ids)
```

## 使用方式

PyTorch: 该接口传入待构建通信域rank列表，指定backend创建指定的通信域，并返回创建的通信域。

MindSpore：该接口传入group名字，以及待构建通信域rank列表，创建一个以传入的group名字为key的通信域，不返回任何值。