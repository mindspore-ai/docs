# # Function Differences between torch.optim.AdamW and mindspore.nn.AdamWeightDecay

<a href="https://gitee.com/mindspore/docs/blob/master/docs/mindspore/source_en/note/api_mapping/pytorch_diff/AdamWeightDecay.md" target="_blank"><img src="https://mindspore-website.obs.cn-north-4.myhuaweicloud.com/website-images/master/resource/_static/logo_source_en.png"></a>

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

For more information, see [torch.optim.AdamW](https://pytorch.org/docs/1.5.0/optim.html#torch.optim.AdamW).

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

For more information, see [mindspore.nn.AdamWeightDecay](https://mindspore.cn/docs/en/master/api_python/nn/mindspore.nn.AdamWeightDecay.html#mindspore.nn.AdamWeightDecay).

## Differences

The code implementation and parameter update logic of `mindspore.nn.AdamWeightDecay` optimizer is different from `torch.optim.AdamW`，for more information, please refer to the docs of official website.

## Code Example

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
