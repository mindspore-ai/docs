# Differences with torch.nn.ModuleDict

[![View Source On Gitee](https://mindspore-website.obs.cn-north-4.myhuaweicloud.com/website-images/r2.3/resource/_static/logo_source_en.svg)](https://gitee.com/mindspore/docs/blob/r2.3/docs/mindspore/source_en/note/api_mapping/pytorch_diff/CellDict.md)

## torch.nn.ModuleDict

```text
class torch.nn.ModuleDict(modules=None)
```

For more information, see [torch.nn.ModuleDict](https://pytorch.org/docs/1.8.1/generated/torch.nn.ModuleDict.html).

## mindspore.nn.CellDict

```text
class mindspore.nn.CellDict(*args, **kwargs)
```

For more information, see [mindspore.nn.CellDict](https://www.mindspore.cn/docs/en/r2.3/api_python/nn/mindspore.nn.CellDict.html).

## Differences

PyTorch: ModuleDict is a Module dictionary that can be used like a regular Python dictionary.

MindSpore: MindSpore API implementation is basically the same as PyTorch. The types of Cells supported by CellDict are inconsistent with ModuleDict in two ways. First, compared to ModuleDict, CellDict does not support the storage of CellDict, CellList and SequentialCell derived from Cell, and see code example 1; Second, CellDict does not support Cell with the storage type of None, and see code example 2.

| Categories | Subcategories |PyTorch | MindSpore | Difference |
| --- | ---   | ---   | ---        |---  |
| Parameters | Parameter 1  | modules | args  | The parameter names are different, but the meanings of the parameters are the same, both are used to initialize the ModuleDict or CellDict iterable object |
|      | Parameter 2  |         | kwargs | MindSpore is reserved for keyword parameters to be expanded, and PyTorch does not have this parameter |

### Code Example 1

```python
# PyTorch
from torch import nn

linear_p = nn.ModuleList([nn.Linear(2, 2)])

class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.module_dict = nn.ModuleDict({'conv': nn.Conv2d(1, 1, 3), 'linear': linear_p})

    def forward(self):
        return self.module_dict.items()

net = Net()
modules = net()
for item in modules:
    print(item[0])
    print(item[1])
# conv
# Conv2d(1, 1, kernel_size=(3, 3), stride=(1, 1))
# linear
# ModuleList(
#   (0): Linear(in_features=2, out_features=2, bias=True)
# )
```

### Code Example 2

```python
# PyTorch
from torch import nn

class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.module_dict = nn.ModuleDict({'conv': None, 'pool': None})

    def forward(self):
        return self.module_dict.items()

net = Net()
modules = net()
for item in modules:
    print(item[0])
    print(item[1])
# conv
# None
# pool
# None
```
