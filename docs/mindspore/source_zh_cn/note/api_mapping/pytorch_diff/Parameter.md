# 比较与torch.nn.parameter.Parameter的功能差异

<a href="https://gitee.com/mindspore/docs/blob/master/docs/mindspore/source_zh_cn/note/api_mapping/pytorch_diff/Parameter.md" target="_blank"><img src="https://mindspore-website.obs.cn-north-4.myhuaweicloud.com/website-images/master/resource/_static/logo_source.svg"></a>

## torch.nn.parameter.Parameter

```python
torch.nn.parameter.Parameter(data=None, requires_grad=True)
```

更多内容详见[torch.nn.parameter.Parameter](https://pytorch.org/docs/1.8.1/generated/torch.nn.parameter.Parameter.html#torch.nn.parameter.Parameter)。

## mindspore.Parameter

```python
mindspore.Parameter(default_input, name=None, requires_grad=True, layerwise_parallel=False, parallel_optimizer=True)
```

更多内容详见[mindspore.Parameter](https://mindspore.cn/docs/zh-CN/master/api_python/mindspore/mindspore.Parameter.html#mindspore.Parameter)。

## 差异对比

PyTorch：PyTorch中一种特殊的Tensor，是Tensor的一个子类，与普通Tensor不同，Parameter会被自动的注册为模型参数，从而被优化器更新。

MindSpore：MindSpore的Parameter也是一种特殊的Tensor，与PyTorch不同，MindSpore的Parameter与Tensor均继承与C接口的Tensor_。

其次，MindSpore的 `requires_grad` 参数与PyTorch也有差异，该参数在PyTorch中为一个后端级别的参数，该参数设为 ``False`` 时，代表不需要对该张量计算梯度，构建计算图时不会包含该张量，也不会记录每个操作的梯度信息，在推理阶段等场景可以提高计算效率；而MindSpore的该参数则为一个前端级别的参数，参数设为 ``False`` 时，MindSpore的自动微分机制在后端依旧会对该张量进行梯度计算，只会在前端侧对该参数以属性的形式进行展示并使用，例如，MindSpore中的 `trainable_params` 方法则会屏蔽掉Parameter中 `requires_grad` 为 ``False`` 的属性。

此外，MindSpore的Parameter相比PyTorch多了一个name参数，该参数会作为一个属性与Parameter进行强绑定，在后端执行图编译时或者执行ckpt保存时均会使用此参数，该参数可以手动指定，若不指定，则会触发MindSpore的Parameter自动命名机制对Parameter进行命名。

最后，MindSpore的Parameter直接打印时无法查看到里面实际包含的值，需要使用Parameter.value()方法来查看实际的值。

| 分类 | 子类 |PyTorch | MindSpore | 差异 |
| --- | --- | --- | --- |---|
|参数 | 参数1 | data |default_input|一致|
| | 参数2 |-|name|差异见上文|
| | 参数3 |requires_grad|requires_grad|差异见上文|
| | 参数4 |  |layerwise_parallel|MindSpore并行相关特有参数，torch无此参数|
| | 参数5 |  |parallel_optimizer|MindSpore并行相关特有参数，torch无此参数|

### 代码示例

```python
import numpy as np
from mindspore import Parameter, Tensor

a = Parameter(Tensor(np.ones((1, 2), dtype=np.float32)))
print(a)
# Parameter (name=Parameter, shape=(1, 2), dtype=Float32, requires_grad=True)
print(a.value())
# [[1. 1.]]

import torch

b = torch.nn.parameter.Parameter(torch.tensor(np.ones((1, 2), dtype=np.float32)))
print(b.data)
# tensor([[1., 1.]])
```
