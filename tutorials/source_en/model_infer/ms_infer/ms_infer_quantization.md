# Model Quantization

[![View Source on AtomGit](https://mindspore-website.obs.cn-north-4.myhuaweicloud.com/website-images/feature/atomgit/resource/_static/logo_source_en.svg)](https://atomgit.com/mindspore/docs/blob/master/tutorials/source_en/model_infer/ms_infer/ms_infer_quantization.md)

## Overview

MindSpore is an all-scenario AI framework. When a model is deployed on the device or other lightweight devices, it may be subject to memory, power consumption, and latency. Therefore, the model needs to be compressed before deployment.

[MindSpore Golden Stick](https://www.mindspore.cn/golden_stick/docs/en/master/index.html) provides the model compression capability of MindSpore. MindSpore Golden Stick is a set of model compression algorithms jointly designed and developed by Huawei Noah's Ark team and MindSpore team. It provides a series of model compression algorithms for MindSpore, supporting quantization modes such as A16W8, A16W4, A8W8, and KVCache. For details, see [MindSpore Golden Stick](https://www.mindspore.cn/golden_stick/docs/en/master/index.html).

## Basic Model Quantization Process

To help you understand the basic model quantization process of MindSpore Golden Stick, this section uses the quantization algorithm as an example to describe the basic usage.

### Procedure

The MindSpore Golden Stick quantization algorithm can be divided into two phases: quantization phase and deployment phase. The quantization phase is completed before deployment. The main tasks are as follows: collecting weight distribution, computing quantization parameters, quantizing weight data, and inserting dequantization nodes. The deployment phase refers to the process of using the MindSpore framework to perform inference on the quantized model in the production environment.

MindSpore Golden Stick mainly uses `PTQConfig` to customize quantization and deployment, and uses the `apply` and `convert` APIs to implement quantization and deployment. You can configure whether to quantize the weight, activation, and KVCache, and configure the quantization bit in `PTQConfig`. In addition, you can configure the data calibration policy. For details, see [PTQConfig Description](#ptqconfig-description).

The quantization procedure of MindSpore Golden Stick is as follows:

```python
import numpy as np
import mindspore as ms
import mindspore.nn as nn
from mindspore import Tensor, dtype
from mindformers.modules import Linear
from mindspore_gs.common import BackendTarget
from mindspore_gs.ptq import PTQMode, PTQConfig
from mindspore_gs.ptq.ptq import PTQ
from mindspore.dataset import GeneratorDataset

class SimpleNet(nn.Cell):
    class DecoderCell(nn.Cell):
        def __init__(self, linear):
            super().__init__()
            self.linear = linear

        def construct(self, *args, **kwargs):
            return self.linear(*args, **kwargs)

    def __init__(self, foo_seq_length=1024):
        super().__init__()

        self.foo_seq_length = foo_seq_length
        linear =  Linear(in_channels=foo_seq_length, out_channels=foo_seq_length, weight_init="ones")
        self.decoder = SimpleNet.DecoderCell(linear)

    def construct(self, x):
        return self.decoder(x)

    def generate(self, input_ids, do_sample=False, max_new_tokens=1):
        input_ids = np.pad(input_ids, ((0, 0), (0, self.foo_seq_length - input_ids.shape[1])), 'constant',
                            constant_values=0)
        return self.construct(Tensor(input_ids, dtype=dtype.float16))

def create_foo_ds(repeat=1):
    class SimpleIterable:
        def __init__(self, repeat=1):
            self._index = 0
            self.data = []
            for _ in range(repeat):
                self.data.append(np.array([[1, 1, 1]], dtype=np.int32))

        def __next__(self):
            if self._index >= len(self.data):
                raise StopIteration
            item = (self.data[self._index],)
            self._index += 1
            return item

        def __iter__(self):
            self._index = 0
            return self

        def __len__(self):
            return len(self.data)

    return GeneratorDataset(source=SimpleIterable(repeat), column_names=["input_ids"])


net = SimpleNet() # The float model that needs to be quantized
ds = create_foo_ds(1)
cfg = PTQConfig(mode=PTQMode.QUANTIZE, backend=BackendTarget.ASCEND, weight_quant_dtype=dtype.int8)
ptq = PTQ(cfg)
ptq.apply(net, datasets=ds)
ptq.convert(net)

ms.save_checkpoint(net.parameters_dict(), './simplenet_ptq.ckpt')
```

1. Use [nn.Cell](https://www.mindspore.cn/docs/en/master/api_python/nn/mindspore.nn.Cell.html) to define the network. After the model is trained, the floating-point weights of the model are obtained. During inference, the floating-point weights of the model are loaded. The preceding example simplifies the process by directly creating a network and quantizing the network using the initial floating-point weights.
2. Use PTQConfig to set the mode to quantization and backend to Ascend for 8-bit quantization of the weights. For details, see [PTQConfig Description](#ptqconfig-description).
3. Use the apply API to convert the network into a fake-quantized network and collect statistics on the quantization objects according to `PTQConfig`.
4. Use the convert API to perform real quantization on the fake-quantized network obtained in the previous step to obtain the quantized network.

After the quantization is complete, you can use the quantized model for inference. The procedure is as follows:

```python
import numpy as np
import mindspore as ms
import mindspore.nn as nn
from mindspore import Tensor, dtype
from mindformers.modules import Linear
from mindspore_gs.common import BackendTarget
from mindspore_gs.ptq import PTQMode, PTQConfig
from mindspore_gs.ptq.ptq import PTQ
from mindspore.dataset import GeneratorDataset

class SimpleNet(nn.Cell):
    class DecoderCell(nn.Cell):
        def __init__(self, linear):
            super().__init__()
            self.linear = linear

        def construct(self, *args, **kwargs):
            return self.linear(*args, **kwargs)

    def __init__(self, foo_seq_length=1024):
        super().__init__()

        self.foo_seq_length = foo_seq_length
        linear =  Linear(in_channels=foo_seq_length, out_channels=foo_seq_length, weight_init="ones")
        self.decoder = SimpleNet.DecoderCell(linear)

    def construct(self, x):
        return self.decoder(x)

    def generate(self, input_ids, do_sample=False, max_new_tokens=1):
        input_ids = np.pad(input_ids, ((0, 0), (0, self.foo_seq_length - input_ids.shape[1])), 'constant',
                            constant_values=0)
        return self.construct(Tensor(input_ids, dtype=dtype.float16))

net = SimpleNet()
cfg = PTQConfig(mode=PTQMode.DEPLOY, backend=BackendTarget.ASCEND, weight_quant_dtype=dtype.int8)
ptq = PTQ(cfg)
ptq.apply(net)
ptq.convert(net)
ms.load_checkpoint('./simplenet_ptq.ckpt', net)

input = Tensor(np.ones((5, 1024), dtype=np.float32), dtype=dtype.float32)
output = net(input)
print(output)
```

1. Use PTQConfig to set the mode to deployment and backend to Ascend for 8-bit quantization of the weights. For details, see [PTQConfig Description](#ptqconfig-description).
2. Use the apply and convert APIs to convert the network into a quantized network. In the deployment phase, no information statistics are collected or quantization computing is performed. Only the network structure is converted into a quantized network.
3. Load the quantized weights to the quantized network for inference.

### PTQConfig Description

You can customize the PTQConfig to enable different quantization capabilities. For details about PTQConfig, see the [API document](https://www.mindspore.cn/golden_stick/docs/en/master/ptq/mindspore_gs.ptq.PTQConfig.html#mindspore_gs.ptq.PTQConfig). The following lists the configuration examples of these algorithms:

> **A** indicates activation, **W** indicates weight, **C** indicates KVCache, and the number indicates the bit. For example, A16W8 indicates that the activation is quantized to float16 and the weight is quantized to int8.

- A16W8 weight quantization

    ```python
    from mindspore import dtype as msdtype
    from mindspore_gs.ptq import PTQConfig, OutliersSuppressionType

    ptq_config = PTQConfig(weight_quant_dtype=msdtype.int8,  act_quant_dtype=None,  kvcache_quant_dtype=None,
                        outliers_suppression=OutliersSuppressionType.NONE)
    ```

- A8W8 quantization

    > A8W8 quantization is based on the [SmoothQuant](https://gitcode.com/gh_mirrors/smo/smoothquant/overview) algorithm. PTQConfig provides the **outliers_suppression** field to specify whether to perform the smooth operation.

    ```python
    from mindspore import dtype as msdtype
    from mindspore_gs.ptq import PTQConfig, OutliersSuppressionType

    ptq_config = PTQConfig(weight_quant_dtype=msdtype.int8, act_quant_dtype=msdtype.int8, kvcache_quant_dtype=None,
                        outliers_suppression=OutliersSuppressionType.SMOOTH)
    ```

- KVCache int8 quantization

    ```python
    from mindspore import dtype as msdtype
    from mindspore_gs.ptq import PTQConfig, OutliersSuppressionType

    ptq_config = PTQConfig(weight_quant_dtype=None, act_quant_dtype=None, kvcache_quant_dtype=msdtype.int8,
                        outliers_suppression=OutliersSuppressionType.NONE)
    ```

## Examples

### PTQ Examples

The following provides the complete process of quantizing and deploying the post-training quantization (PTQ) algorithm on the Llama2 network:

- [PTQ algorithm](https://www.mindspore.cn/golden_stick/docs/en/master/ptq/ptq.html): supports 8-bit weight quantization, 8-bit full quantization, and KVCacheInt8 quantization. SmoothQuant can be used to improve the quantization precision. Combined quantization algorithms of different algorithms are supported to improve the quantization inference performance.

### Perceptual Quantization Training Examples

- [SimQAT algorithm](https://www.mindspore.cn/golden_stick/docs/en/master/quantization/simulated_quantization.html): A basic quantization aware algorithm based on the fake quantization technology.
- [SLB quantization algorithm](https://www.mindspore.cn/golden_stick/docs/en/master/quantization/slb.html): A non-linear low-bit quantization aware algorithm.

### Pruning Examples

- [SCOP pruning algorithm](https://www.mindspore.cn/golden_stick/docs/en/master/pruner/scop.html): A structured weight pruning algorithm.
