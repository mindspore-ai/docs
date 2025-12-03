# mint API 介绍

[![查看源文件](https://mindspore-website.obs.cn-north-4.myhuaweicloud.com/website-images/master/resource/_static/logo_source.svg)](https://atomgit.com/mindspore/docs/blob/master/docs/mindspore/source_zh_cn/features/mint.md)

## 介绍

随着cann推出aclnn类算子，mindspore存量的ops nn等API要进行适配优化，为了不影响原有API的行为同时保证兼容性，我们为此创建一个新的API目录来做这件事，新目录的名字mint的想法来自于linux is not unix。在mindspore.mint下，提供张量创建、计算、神经网络、通信等常用pytorch-like API。本文主要介绍mint类API支持范围和入参区别等。这部分API主要包含张量创建，随机采样，数学计算，神经网络，集群通信类。

### 张量创建

以empty这个API看下主要差异点：

| torch.empty | mindspore.mint.empty | 说明 |
|:---: |   :---:  | :---:|
| `*size` (int...) | `*size` (int...) | 必选 |
| `dtype` | `dtype` | 可选 |
| `device` | `device` | 可选 |
| `layout` | 无 | 可选 |
|`requires_grad` | 无 | 可选 |
| `pin_memory` | 无 | 可选 |
|`memory_format` | 无 | 可选 |
| `out` | 无 | 可选 |

#### 当前不支持的参数说明

- `layout`：创建torch tensor时，一般默认layout是stride，即dense tensor。mindspore创建tensor时，默认是dense tensor，与torch 无差异。开发者无需设置。
- `memory_format`：tensor的内存排布，默认都是NCHW格式。torch 提供channel_last格式即NHWC，在一些场景中，这样会有性能提升，但是泛化性和兼容性需要开发者实际测试和验证。使用mindspore开发，可不设置此参数。
- `requires_grad`：由于框架自动微分求导机制不同，mindspore在tensor的属性中没有设置此参数。对于是否需要计算梯度，常用的parameter类提供了此参数。如果无需计算梯度，可参考[mindspore.ops.stop_gradient](https://www.mindspore.cn/docs/zh-CN/master/api_python/ops/mindspore.ops.stop_gradient.html)。
- `pin_memory`：返回的tensor被分配到pinned memory，我们已经规划支持此功能。计划在2.7.1版本推出。
- `out`：指定输出张量，用于原地操作和内存优化。当提供 `out` 参数时，操作结果会直接写入到指定的张量中，而不是创建新的张量。当前未规划支持此参数。

**代码示例**：

```diff
- import torch
+ import mindspore

- x = torch.empty(2, 3, dtype=torch.float32)
+ x = mindspore.mint.empty(2, 3, dtype=mindspore.float32)
```

总结：tensor相关可选参数涉及框架实现机制不同，我们也会根据开发者反馈不断完善，如tensor storage能力已规划。

### 随机采样

以bernoulli举例：

| torch.bernoulli | mindspore.mint.bernoulli | 说明 |
|:---: |   :---:  | :---:|
| `input` (Tensor...) | `input` (Tensor...) | 必选 |
| `generator` | `generator`| 可选 |
| `out` | 无 | 可选 |

out参数差异参考张量创建。

**代码示例**：

```diff
- import torch
+ import mindspore.mint

- a = torch.ones(3, 3)
+ a = mindspore.mint.ones(3, 3)

- torch.bernoulli(a)
+ mindspore.mint.bernoulli(a)
```

### 数学计算

基础计算类当前均已支持，以mul举例：

| torch.mul | mindspore.mint.mul | 说明 |
|:---: |   :---:  | :---:|
| `*size` (Tensor...) | `*size` (Tensor...) | 必选 |
| `other` | `other`| 可选 |
| `out` | 无 | 可选 |

计算类ops当前不支持的参数与tensor creation是类似的，这与tensor实现机制相关。例如out：

**代码示例**：

```diff
- import torch
+ import mindspore.mint

- a = torch.randn(3)
+ a = mindspore.mint.randn(3)

- torch.mul(a,50)
+ mindspore.mint.mul(a,50)
```

### 神经网络

常用nn类，例如conv2d，参数均一致。

| torch.conv2d | mindspore.mint.conv2d | 说明 |
|:---: |   :---:  | :---:|
| `in_channels` (int) | `in_channels` (int) | 必选 |
| `out_channels`(int) | `out_channels`(int) | 必选 |
| `kernel_size` (int or tuple) | `kernel_size` (int or tuple) | 必选 |
| `stride` (int or tuple) | `stride` (int or tuple) | 可选 |
| `padding`(int, tuple or str) | `padding`(int, tuple or str) | 可选 |
| `padding_mode` (str) | `padding_mode` (str)  | 可选 |
| `dilation`(int or tuple) | `dilation`(int or tuple) | 可选 |
| `groups` (int) | `groups` (int) | 可选 |
| `bias`(bool) | `bias`(bool) | 可选 |

**代码示例**：

```diff
- import torch
+ import mindspore

in_channels = 16
out_channels = 33
kernel_size = (3, 5)
stride = (2, 1)
padding = (4, 2)
dilation = (3, 1)

- input = torch.rand(20,16,50,100)
+ input = mindspore.mint.rand(20,16,50,100)

- model = torch.conv2d(16,33,(3,5),stride=(2, 1), padding=(4, 2), dilation=(3, 1))
+ model = mindspore.mint.conv2d(16,33,(3,5),stride=(2, 1), padding=(4, 2), dilation=(3, 1))

output = model(input)
```

包含inplace参数的，当前未全部支持，例如：

| API  | Args  |
| :--- | :--- |
| torch.nn.functional_dropout2d | input, p=0.5, training=True, inplace=False |
| mindspore.mint.nn.functional_dropout2d | input, p=0.5, training=True|

torch废弃的参数，不支持，例如：

| torch.nn.MSELoss | 是否废弃 | mindspore.nn.MSELoss | 说明 |
|:---: |  :--- |  :---:  | :---:|
| `size_average`  | yes  | N.A | 不支持 |
| `reduce`  | yes  | N.A | 不支持 |
| `reduction`  | no  | `reduction` | 支持 |

### 集群通信类

常用all_gather/all_reduce/all_to_all等均已支持，参数也保持一致，例如：

| torch.distributed.all_gather | mindspore.mint.distributed.all_gather | 说明 |
|:---: |   :---:  | :---:|
| `tensor_list` (list[Tensor]) | `tensor_list` (list(Tensor)) | 必选 |
| `tensor`(Tensor) | `tensor`(Tensor) | 可选 |
| `group`(ProcessGroup) | `group` (ProcessGroup) | 可选 |
| `async_op` (bool) | `async_op` (bool) | 可选 |

| torch.distributed.all_reduce | mindspore.mint.distributed.all_reduce | 说明 |
|:---: |   :---:  | :---:|
| `tensor` (Tensor) | `Tensor` (Tensor) | 必选 |
| `op` | `op` | 可选 |
| `group`(ProcessGroup) | `group` (ProcessGroup) | 可选 |
| `async_op` (bool) | `async_op` (bool) | 可选 |

更多API支持情况请查阅[mint支持列表](https://www.mindspore.cn/docs/zh-CN/master/api_python/mindspore.mint.html)。
