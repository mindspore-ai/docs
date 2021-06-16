# Comparing the Function Differences with torch.distributed.new_group

## torch.distributed.new_group

```python
torch.distributed.new_group(
    ranks=None,
    timeout=datetime.timedelta(0, 1800),
    backend=None
)
```

## mindspore.communication.create_group

```python
mindspore.communication.create_group(group, rank_ids)
```

## Differences

PyTorch: This interface passes in the rank list of the communication domain to be constructed, specifies the backend to create the specified communication domain, and returns the created communication domain.

MindSpore：The interface passes in the group name and the rank list of the communication domain to be constructed, creates a communication domain with the incoming group name as the key, and does not return any value.