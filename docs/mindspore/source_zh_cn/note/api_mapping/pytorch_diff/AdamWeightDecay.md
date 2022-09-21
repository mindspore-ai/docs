# 比较与torch.optim.AdamW的功能差异

<a href="https://gitee.com/mindspore/docs/blob/r1.9/docs/mindspore/source_zh_cn/note/api_mapping/pytorch_diff/AdamWeightDecay.md" target="_blank"><img src="https://mindspore-website.obs.cn-north-4.myhuaweicloud.com/website-images/r1.9/resource/_static/logo_source.png"></a>

## torch.optim.AdamW

```python
class torch.optim.AdamW(
    params,
    lr=0.001,
    betas=(0.9, 0.999),
    eps=1e-08,
    weight_decay=0.01,
    amsgrad=False,
    maximize=False,
    foreach=None,
    capturable=False
)
```

更多内容详见[torch.optim.AdamW](https://pytorch.org/docs/1.5.0/optim.html#torch.optim.AdamW)。

## mindspore.nn.AdamWeightDecay

```python
class mindspore.nn.AdamWeightDecay(
    params,
    learning_rate=1e-3,
    beta1=0.9,
    beta2=0.999,
    eps=1e-6,
    weight_decay=0.0
)
```

更多内容详见[mindspore.nn.AdamWeightDecay](https://mindspore.cn/docs/zh-CN/r1.9/api_python/nn/mindspore.nn.AdamWeightDecay.html#mindspore.nn.AdamWeightDecay)。

## 使用方式

`mindspore.nn.AdamWeightDecay` 优化器实现方式与参数更新逻辑与 `torch.optim.AdamW` 不同，详情请参考官网注释公式。

## 代码示例

```python
# The following implements AdamWeightDecay with MindSpore.
import numpy as np
import torch
import mindspore.nn as nn
import mindspore as ms

net = Net()
#1) All parameters use the same learning rate and weight decay
optim = nn.AdamWeightDecay(params=net.trainable_params())

#2) Use parameter groups and set different values
conv_params = list(filter(lambda x: 'conv' in x.name, net.trainable_params()))
no_conv_params = list(filter(lambda x: 'conv' not in x.name, net.trainable_params()))
group_params = [{'params': conv_params, 'weight_decay': 0.01, 'grad_centralization':True},
                {'params': no_conv_params, 'lr': 0.01},
                {'order_params': net.trainable_params()}]
optim = nn.AdamWeightDecay(group_params, learning_rate=0.1, weight_decay=0.0)


loss = nn.SoftmaxCrossEntropyWithLogits()
model = ms.Model(net, loss_fn=loss, optimizer=optim)

# The following implements AdamWeightDecay with torch.
input_x = torch.tensor(np.random.rand(1, 20).astype(np.float32))
input_y = torch.tensor([1.])
net = torch.nn.Sequential(torch.nn.Linear(input_x.shape[-1], 1))
loss = torch.nn.MSELoss()
optimizer = torch.optim.AdamW(net.parameters())
l = loss(net(input_x).view(-1), input_y) / 2
optimizer.zero_grad()
l.backward()
optimizer.step()
```
