# Comparing the Function Differences with torch.distributed.new_group

<a href="https://gitee.com/mindspore/docs/blob/master/docs/mindspore/migration_guide/source_en/api_mapping/pytorch_diff/create_group.md" target="_blank"><img src="https://mindspore-website.obs.cn-north-4.myhuaweicloud.com/website-images/master/resource/_static/logo_source_en.png"></a>

## torch.distributed.new_group

```python
torch.distributed.new_group(
    ranks=None,
    timeout=datetime.timedelta(0, 1800),
    backend=None
)
```

For more information, see [torch.distributed.new_group](https://pytorch.org/docs/1.5.0/distributed.html#torch.distributed.new_group).

## mindspore.communication.create_group

```python
mindspore.communication.create_group(group, rank_ids)
```

For more information, see [mindspore.communication.create_group](https://mindspore.cn/docs/api/en/master/api_python/mindspore.communication.html#mindspore.communication.create_group).

## Differences

PyTorch: This interface passes in the rank list of the communication domain to be constructed, specifies the backend to create the specified communication domain, and returns the created communication domain.

MindSpore：The interface passes in the group name and the rank list of the communication domain to be constructed, creates a communication domain with the incoming group name as the key, and does not return any value.