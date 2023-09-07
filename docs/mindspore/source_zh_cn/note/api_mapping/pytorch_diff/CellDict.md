# 比较与torch.nn.ModuleDict的差异

[![查看源文件](https://mindspore-website.obs.cn-north-4.myhuaweicloud.com/website-images/master/resource/_static/logo_source.png)](https://gitee.com/mindspore/docs/blob/master/docs/mindspore/source_zh_cn/note/api_mapping/pytorch_diff/CellDict.md)

## torch.nn.ModuleDict

```text
class torch.nn.ModuleDict(modules=None)
```

更多内容详见[torch.nn.ModuleDict](https://pytorch.org/docs/1.8.1/generated/torch.nn.ModuleDict.html)。

## mindspore.nn.CellDict

```text
class mindspore.nn.CellDict(*args, **kwargs)
```

更多内容详见[mindspore.nn.CellDict](https://www.mindspore.cn/docs/zh-CN/master/api_python/nn/mindspore.nn.CellDict.html)。

## 差异对比

PyTorch：ModuleDict是一个Module字典，可以像使用普通Python字典一样使用它。

MindSpore：MindSpore此API实现功能与PyTorch基本一致。MindSpore在功能上有一点与PyTorch不一致，
相比于ModuleDict, CellDict不支持存储从Cell派生而来的CellDict、CellList以及SequentialCell。

| 分类 | 子类   | PyTorch | MindSpore  | 差异 |
| ---- | ------ | -------| -----------| ------|
| 参数 | 参数1  | modules | args  | 参数名不同，参数含义相同，均是用于初始化ModuleDict或CellDict的可迭代对象 |
|      | 参数2  |         | kwargs | MindSpore为待扩展的关键字参数预留，PyTorch无该参数 |

### 代码示例1

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
