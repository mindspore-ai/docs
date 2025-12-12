# Introduction to mint API

[![View Source On Gitee](https://mindspore-website.obs.cn-north-4.myhuaweicloud.com/website-images/master/resource/_static/logo_source_en.svg)](https://gitee.com/mindspore/docs/blob/master/docs/mindspore/source_en/features/mint.md)

## Overview

With the introduction of aclnn-class operators in cann, existing MindSpore APIs like ops and nn require adaptation and optimization. To preserve the behavior of existing APIs while ensuring compatibility, we have created a new API directory for this purpose. The name "mint" for this directory draws inspiration from the Linux motto "Linux is not unix". Under mindspore.mint, common pytorch-like APIs are provided for tensor creation, computation, neural networks, communication, and more. This article primarily introduces the scope of support for mint-class APIs and differences in input parameters. This set of APIs mainly includes tensor creation, random sampling, mathematical computation, neural networks, and cluster communication classes.

### Tensor Creation

Let's examine the key differences using the API empty:

| torch.empty | mindspore.mint.empty | Explanation |
|:---: |   :---:  | :---:|
| `*size` (int...) | `*size` (int...) | Required |
| `dtype` | `dtype` | Optional |
| `device` | `device` | Optional |
| `layout` | - | Optional |
|`requires_grad` | - | Optional |
|`memory_format` | - | Optional |
| `out` | - | Optional |

#### Description of Currently Unsupported Parameters

- `layout`: When torch creates a tensor, the default layout is typically stride, i.e, a dense tensor. When MindSpore creates a tensor, the default is also a dense tensor, identical to torch. Developers do not need to set this.
- `memory_format`: The default memory layout for tensors is NCHW format. Torch provides the channel_last format (NHWC), which may offer performance improvements in certain scenarios. However, developers should conduct actual testing and verification to ensure its generalizability and compatibility. When developing with MindSpore, this parameter does not need to be set.
- `requires_grad`: Due to differences in the framework's automatic differentiation mechanism, MindSpore does not include this parameter in its Tensor attributes. For determining whether gradient computation is required, the commonly used parameter class provides this parameter. If gradient computation is unnecessary, refer to [mindspore.ops.stop_gradient](https://www.mindspore.cn/docs/en/master/api_python/ops/mindspore.ops.stop_gradient.html).
- `out`: Specify the output tensor for in-place operations and memory optimization. When the `out` parameter is provided, the operation result is written directly to the specified tensor instead of creating a new one. Support for this parameter is currently not planned.

**Code Example**:

```diff
- import torch
+ import mindspore

- x = torch.empty(2, 3, dtype=torch.float32)
+ x = mindspore.mint.empty(2, 3, dtype=mindspore.float32)
```

Summary: Tensor-related optional parameters vary depending on framework implementation mechanisms. We will continue to refine them based on developer feedback, such as the planned enhancement of tensor storage capabilities.

### Random Sampling

Take Bernoulli as an example:

| torch.bernoulli | mindspore.mint.bernoulli | Explanation |
|:---: |   :---:  | :---:|
| `input` (Tensor...) | `input` (Tensor...) | Required |
| `generator` | `generator`| Optional |
| `out` | - | Optional |

For differences in the `out` parameter, refer to Tensor creation.

**Code Example**:

```diff
- import torch
+ import mindspore.mint

- a = torch.ones(3, 3)
+ a = mindspore.mint.ones(3, 3)

- torch.bernoulli(a)
+ mindspore.mint.bernoulli(a)
```

### Mathematical Calculation

All basic arithmetic operations are now supported. For example, the multiplication operation:

| torch.mul | mindspore.mint.mul | Explanation |
|:---: |   :---:  | :---:|
| `*size` (Tensor...) | `*size` (Tensor...) | Required |
| `other` | `other`| Optional |
| `out` | - | Optional |

The parameters currently unsupported by computational ops are similar to those for tensor creation, which relates to the underlying implementation mechanism of tensors. For example, `out`:

**Code Example**:

```diff
- import torch
+ import mindspore.mint

- a = torch.randn(3)
+ a = mindspore.mint.randn(3)

- torch.mul(a,50)
+ mindspore.mint.mul(a,50)
```

### Neural Network

Common nn classes, such as conv2d, share identical parameters.

| torch.conv2d | mindspore.mint.conv2d | Explanation |
|:---: |   :---:  | :---:|
| `in_channels` (int) | `in_channels` (int) | Required |
| `out_channels`(int) | `out_channels`(int) | Required |
| `kernel_size` (int or tuple) | `kernel_size` (int or tuple) | Required |
| `stride` (int or tuple) | `stride` (int or tuple) | Optional |
| `padding`(int, tuple or str) | `padding`(int, tuple or str) | Optional |
| `padding_mode` (str) | `padding_mode` (str)  | Optional |
| `dilation`(int or tuple) | `dilation`(int or tuple) | Optional |
| `groups` (int) | `groups` (int) | Optional |
| `bias`(bool) | `bias`(bool) | Optional |

**Code Example**:

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

Functions containing the `inplace` parameter are not yet fully supported. For example:

| API  | Args  |
| :--- | :--- |
| torch.nn.functional_dropout2d | input, p=0.5, training=True, inplace=False |
| mindspore.mint.nn.functional_dropout2d | input, p=0.5, training=True|

Deprecated parameters in Torch are not supported, for example:

| torch.nn.MSELoss | Discontinued or not | mindspore.nn.MSELoss | Explanation |
|:---: |  :--- |  :---:  | :---:|
| `size_average`  | yes  | N.A | Not supported |
| `reduce`  | yes  | N.A | Not supported |
| `reduction`  | no  | `reduction` | Supported |

### Cluster Communication Class

Common operations such as `all_gather`, `all_reduce`, and `all_to_all` are now supported, with consistent parameters. For example:

| torch.distributed.all_gather | mindspore.mint.distributed.all_gather | Explanation |
|:---: |   :---:  | :---:|
| `tensor_list` (list[Tensor]) | `tensor_list` (list(Tensor)) | Required |
| `tensor`(Tensor) | `tensor`(Tensor) | Optional |
| `group`(ProcessGroup) | `group` (ProcessGroup) | Optional |
| `async_op` (bool) | `async_op` (bool) | Optional |

| torch.distributed.all_reduce | mindspore.mint.distributed.all_reduce | Explanation |
|:---: |   :---:  | :---:|
| `tensor` (Tensor) | `Tensor` (Tensor) | Required |
| `op` | `op` | Optional |
| `group`(ProcessGroup) | `group` (ProcessGroup) | Optional |
| `async_op` (bool) | `async_op` (bool) | Optional |

For more API support details, please refer to the [mint support list](https://www.mindspore.cn/docs/en/master/api_python/mindspore.mint.html).
