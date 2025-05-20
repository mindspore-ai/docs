# MSAdapter概述

MSAdapter是一款MindSpore生态适配工具，在不改变用户原有使用习惯下，将PyTorch/JAX等三方框架代码快速迁移到MindSpore生态上，帮助用户高效使用昇腾算力。当前MSAdapter适配PyTorch版本2.1.0。

## 相关说明

文档中对API的支持程度分为三类，Stable、Beta、Not Support：  
**Stable**：经过测试验证的API，与PyTorch原生API行为一致。  
**Beta**：核心功能已经完成，还未进行测试的API，且可能存在入参与PyTorch原生API不完全一致的情况。  
**Not Support**：当前还未实现的API

## 使用限制

### Out参数使用限制

为更好地兼容PyTorch，我们的部分API设计中支持了out形式的输出参数。关于out参数的使用，有以下限制：

**反向传播限制**：
在PyTorch中，指定out参数后，相关操作通常不支持反向传播（即不会自动进行梯度计算），这是由于底层实现无法追踪到out张量的梯度信息。我们的实现也遵循这一行为，即当使用out参数时，不支持自动的反向传播计算。另外，在PyTorch中同时传入out参数和requires_grad=True的张量时会报错，MSAdapter当前不报错，请注意。

**Shape 要求**：
与PyTorch能够自动调整out参数的张量shape不同，MSAdapter的out参数不支持自动resize。在使用out参数时，用户必须显式地传入与运算结果shape完全一致的张量。例如，如果函数的输出shape 为 [2, 3, 4]，则传入的out张量也必须为[2, 3, 4]，否则会报错。请务必确保所传入的out张量shape正确匹配，MSAdapter不会对传入的out参数自动调用resize。

### 暂不支持Complex64/Complex128

示例代码：

```python
  from torch.utils.data import DataLoader
  from torchvision import datasets
  from torchvision.transforms import ToTensor

  training_data = datasets.FashionMNIST(root="data", train=True, download=True, transform=ToTensor())
  train_dataloader = DataLoader(training_data, batch_size=64, pin_memory=True)
  for batch, (X, y) in enumerate(train_dataloader):
      X, y = X.cuda(), y.cuda()
```

报错信息如下：

```python
 Traceback (most recent call last):
     File "/path/to/your/torch/utils/data/_utils/pin_memory.py", line 98, in pin_memory
         clone[i] = pin_memory(item, device)
     File "/path/to/your/torch/utils/data/_utils/pin_memory.py", line 64, in pin_memory
         return data.pin_memory(device)
 TypeError: pin_memory() takes 1 positional argument but 2 were given
```

### 暂不支持动态Profiling

示例代码：

```python
 import torch
 from torch.profiler import profile, record_function, ProfilerActivity

 with profile(activities=[ProfilerActivity.CPU], record_shapes=True) as prof:
     # 训练代码
     for i in range(10):
     # 模拟训练步骤
         pass

```

报错信息如下：

```python
 Traceback (most recent call last):
     File "/path/to/your/demo.py", line 102, in <module>
         with profile(activities=[ProfilerActivity.CPU], record_shapes=True) as prof:
     File "/path/to/your/torch/profiler/profiler.py", line 54, in __init__
         profiler_level = experimental_config._profiler_level,
 AttributeError: 'NoneType' object has no attribute '_profiler_level'

```

### Dataloader中的pin_memory参数仅支持设置为False

示例代码：

```python
  from torch.utils.data import DataLoader
  from torchvision import datasets
  from torchvision.transforms import ToTensor

  training_data = datasets.FashionMNIST(root="data", train=True, download=True, transform=ToTensor())
  train_dataloader = DataLoader(training_data, batch_size=64, pin_memory=True)
  for batch, (X, y) in enumerate(train_dataloader):
      X, y = X.cuda(), y.cuda()
```

报错信息如下：

```python
 Traceback (most recent call last):
     File "/path/to/your/torch/utils/data/_utils/pin_memory.py", line 98, in pin_memory
         clone[i] = pin_memory(item, device)
     File "/path/to/your/torch/utils/data/_utils/pin_memory.py", line 64, in pin_memory
         return data.pin_memory(device)
 TypeError: pin_memory() takes 1 positional argument but 2 were given
```

### 不支持tensor.backward()操作

示例代码：

```python
  import torch
  x = torch.randn(2,)
  x.backward()
```

报错信息如下：

```python
 Traceback (most recent call last):
     File "/path/to/your/demo.py", line XX, in <module>
         x.backward()
     File "/path/to/your/torch/_tensor.py", line 325, in backward
         raise ValueError('not support Tensor.backward yet.')
 ValueError: not support Tensor.backward yet.
```

### 不支持to(device)操作

示例代码：

```python
  import torch
  x = torch.randn(2,)
  device = "cuda"
  x.to(device)
```

报错信息如下：

```python
 Traceback (most recent call last):
     File "/path/to/your/demo.py", line XX, in <module>
         x.to(device)
     File "/path/to/your/mindspore/common/tensor.py", line 3018, in to
         return self if self.dtype == dtype else self._to(dtype)
 TypeError: _to(): argument 'dtype' (position 1) must be mstype, not str.

 ----------------------------------------------------
 - C++ Call Stack: (For framework developers)
 ----------------------------------------------------
 mindspore/ccsrc/pynative/op_function/converter.cc:657 Parse
```

### MindSpore导出的ckpt文件无法被直接加载到PyTorch模型中

示例代码：

```python
 import torch
 from torch import nn
 import mindspore as ms

 class NeuralNetwork(nn.Module):
     def __init__(self):
         super().__init__()
         self.linear = nn.Linear(28*28, 512)

     def forward(self, x):
         logits = self.linear(x)
         return logits

 class myNN(ms.nn.Cell):
     def __init__(self):
         super().__init__()
         self.linear = nn.Linear(28*28, 512)

     def construct(self, x):
         logits = self.linear(x)
         return logits

 model = myNN()
 ms.save_checkpoint(model, "./net.ckpt")
 model2 = NeuralNetwork()
 model.load_state_dict(torch.load("./net.ckpt"))
```

报错信息如下：

```python
 Traceback (most recent call last):
     File "/path/to/your/demo.py", line 99, in <module>
         model.load_state_dict(torch.load("./mynn.ckpt"))
     File "/path/to/your/torch/serialization.py", line 1020, in load
         return _legacy_load(opened_file, pickle_module, **pickle_load_args)
     File "/path/to/your/torch/serialization.py", line 1118, in _legacy_load
         magic_number = pickle_module.load(f, **pickle_load_args)
 EOFError: Ran out of input
```

### 不支持MindSpore与MS-Adapter混合运行

import torch后，mindspore的部分行为会变更为torch的行为，从而产生不可预期的错误。
示例代码：

```python
 from mindspore import Tensor

 a = Tensor([2, 2])
 print(f'before import torch: a.shape={a.shape}')

 import torch
 print(f'after import torch: a.shape={a.shape}')

```

执行结果如下，可以看到，import torch后，原本的mindspore.Tensor.shape行为发生了改变。

```python
 before import torch: a.shape=(2,)
 after import torch: a.shape=torch.Size([2])
```

不支持混合运行的MindSpore接口详见下表：
| 模块 | 受影响接口 |
|------|-----------|
| mindspore.Tensor/mindspore.StubTensor | is_shared, softmax, type_, retain_grad, shape, to_dense, \_base, data, numel, nelement, repeat, cuda, npu, cpu, size, dim, clone, log_softmax, narrow, view, \_\_or\_\_, device, \_\_and\_\_, \_\_xor\_\_, \_\_iter\_\_, \_\_reduce_ex\_\_, expand, detach, T, transpose, mean, clamp, is_cuda, is_cpu, repeat_interleave, is_sparse, requires_grad, requires_grad_, unsqueeze, \_\_pow\_\_, float, backward, split, norm, record_stream, data_ptr, pin_memory, grad, \_\_imul\_\_, reshape, squeeze, element_size, exponential\_ |

## PyTorch模块支持情况

用户可以参考下列表格查看各模块API支持情况。

|模块名|支持情况|
|------|-------|
|torch|部分支持|
|torch.nn|部分支持|
|torch.nn.functional|部分支持|
|torch.Tensor|部分支持|
|Tensor Attributes|不支持|
|torch.amp|不支持|
|torch.autograd|不支持|
|torch.library|不支持|
|torch.accelerator|不支持|
|torch.cpu|不支持|
|torch.cuda|不支持|
|torch.mps|不支持|
|torch.xpu|不支持|
|torch.mtia|不支持|
|torch.mtia.memory|不支持|
|torch.backends|不支持|
|torch.export|不支持|
|torch.distributed|不支持|
|torch.distributed.tensor|不支持|
|torch.distributed.algorithms.join|不支持|
|torch.distributed.elastic|不支持|
|torch.distributed.fsdp|不支持|
|torch.distributed.fsdp.fully_shard|不支持|
|torch.distributed.tensor.parallel|不支持|
|torch.distributed.optim|不支持|
|torch.distributed.pipelining|不支持|
|torch.distributed.checkpoint|不支持|
|torch.distributions|不支持|
|torch.compiler|不支持|
|torch.fft|不支持|
|torch.func|不支持|
|torch.futures|不支持|
|torch.fx|不支持|
|torch.fx.experimental|不支持|
|torch.hub|不支持|
|torch.jit|不支持|
|torch.linalg|不支持|
|torch.monitor|不支持|
|torch.signal|不支持|
|torch.special|不支持|
|torch.overrides|不支持|
|torch.package|不支持|
|torch.profiler|不支持|
|torch.nn.init|不支持|
|torch.nn.attention|不支持|
|torch.onnx|不支持|
|torch.optim|不支持|
|torch.random|不支持|
|torch.masked|不支持|
|torch.nested|不支持|
|torch.Size|不支持|
|torch.sparse|不支持|
|torch.Storage|不支持|
|torch.testing|不支持|
|torch.utils|不支持|
|torch.utils.benchmark|不支持|
|torch.utils.bottleneck|不支持|
|torch.utils.checkpoint|不支持|
|torch.utils.cpp_extension|不支持|
|torch.utils.data|不支持|
|torch.utils.deterministic|不支持|
|torch.utils.jit|不支持|
|torch.utils.dlpack|不支持|
|torch.utils.mobile_optimizer|不支持|
|torch.utils.model_zoo|不支持|
|torch.utils.tensorboard|不支持|
|torch.utils.module_tracker|不支持|
|torch.**config**|不支持|
|torch.**future**|不支持|
|torch._logging|不支持|
