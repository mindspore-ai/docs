# Model Quantization

[![View Source On Gitee](https://mindspore-website.obs.cn-north-4.myhuaweicloud.com/website-images/r2.6.0/resource/_static/logo_source_en.svg)](https://gitee.com/mindspore/docs/blob/r2.6.0/tutorials/source_en/model_infer/ms_infer/quantization.md)

## Overview

MindSpore is an all-scenario AI framework. When a model is deployed on the device or other lightweight devices, the memory, power consumption, and latency are limited. Therefore, the model needs to be compressed before deployment.

[MindSpore Golden Stick](https://www.mindspore.cn/golden_stick/docs/en/r1.1.0/index.html) provides the model compression capability. MindSpore Golden Stick is a model compression algorithm set jointly designed and developed by Huawei Noah team and Huawei MindSpore team. It provides a series of model compression algorithms for MindSpore and supports quantization modes such as A16W8, A16W4, A8W8, and KVCache. For details, see [MindSpore Golden Stick](https://www.mindspore.cn/golden_stick/docs/en/r1.1.0/index.html).

## Basic Model Quantization Process

To help you understand the basic quantization process of the MindSpore Golden Stick model, the following provides examples of quantization algorithms along with the basic usage methods.

### Procedure

MindSpore Golden Stick quantization algorithms can be divided into two phases: quantization and deployment. The quantization phase is completed before deployment. The main tasks are as follows: collecting weight distribution, calculating quantization parameters, quantizing weight data, and inserting dequantization nodes. The deployment phase refers to the process of using the MindSpore framework to perform inference on the quantized model in the production environment.

MindSpore Golden Stick uses `PTQConfig` to define quantization and deployment, and uses the `apply` and `convert` APIs to implement quantization and deployment. In `PTQConfig`, you can configure the data calibration policy, whether to quantize the weight, activation, and KVCache, and the quantization bits. For details, see [PTQConfig Configuration Description](#ptqconfig-configuration-description).

The quantization procedure of MindSpore Golden Stick is as follows:

```python
import numpy as np
import mindspore as ms
import mindspore.nn as nn
from mindspore import Tensor
from mindformers.modules import Linear
from mindspore_gs.common import BackendTarget
from mindspore_gs.ptq import PTQMode, PTQConfig
from mindspore_gs.ptq import RoundToNearest as RTN
from mindspore_gs.ptq.network_helpers import NetworkHelper

class SimpleNet(nn.Cell):
    def __init__(self):
        super().__init__()
        self.linear = Linear(in_channels=5, out_channels=6, transpose_b=True, bias_init="normal", weight_init="normal")

    def construct(self, x):
        return self.linear(x)

class SimpleNetworkHelper(NetworkHelper):
    def __init__(self, **kwargs):
        self.attrs = kwargs

    def get_spec(self, name: str):
        return self.attrs.get(name, None)

    def generate(self, network: nn.Cell, input_ids: np.ndarray, max_new_tokens=1, **kwargs):
        input_ids = np.pad(input_ids, ((0, 0), (0, self.get_spec("seq_length") - inputs_ids.shape[1])), 'constant', constant_values=0)
        network(Tensor(input_ids, dtype=ms.dtype.float16))

net = SimpleNet() # The float model that needs to be quantized
cfg = PTQConfig(mode=PTQMode.QUANTIZE, backend=BackendTarget.ASCEND, weight_quant_dtype=ms.dtype.int8)
net_helper = SimpleNetworkHelper(batch_size=1, seq_length=5)
rtn = RTN(cfg)
rtn.apply(net, net_helper)
rtn.convert(net)

ms.save_checkpoint(net.parameters_dict(), './simplenet_rtn.ckpt')
```

1. Use [nn.Cell](https://www.mindspore.cn/docs/en/r2.0/api_python/nn/mindspore.nn.Cell.html) to define the network. After training the model, obtain the floating-point weight of the model, and then load the floating-point weight during inference. The above example simplifies the process by directly creating a network and using the initial floating-point weight for quantization.
2. Use PTQConfig to set **mode** to quantization mode, set the backend to Ascend, and perform 8-bit quantization on the weight. For details, see [PTQConfig Configuration Description](#ptqconfig-configuration-description).
3. Use the apply API to convert the network into a pseudo-quantized network and collect statistics on the quantized object based on the configuration in `PTQConfig`.
4. Use the convert API to quantize the pseudo-quantized network in the previous step to obtain the quantized network.

After the quantization is complete, you can use the quantized model for inference as follows:

```python
import numpy as np
import mindspore as ms
import mindspore.nn as nn
from mindspore import Tensor
from mindformers.modules import Linear
from mindspore_gs.common import BackendTarget
from mindspore_gs.ptq import PTQMode, PTQConfig
from mindspore_gs.ptq import RoundToNearest as RTN

class SimpleNet(nn.Cell):
    def __init__(self):
        super().__init__()
        self.linear = Linear(in_channels=5, out_channels=6, transpose_b=True, bias_init="normal", weight_init="normal")

    def construct(self, x):
        return self.linear(x)

net = SimpleNet()
cfg = PTQConfig(mode=PTQMode.DEPLOY, backend=BackendTarget.ASCEND, weight_quant_dtype=ms.dtype.int8)
rtn = RTN(cfg)
rtn.apply(net)
rtn.convert(net)
ms.load_checkpoint('./simplenet_rtn.ckpt', net)

input = Tensor(np.ones((5, 5), dtype=np.float32), dtype=ms.dtype.float32)
output = net(input)
print(output)
```

1. Use PTQConfig to set **mode** to deployment mode, set the backend to Ascend, and perform 8-bit quantization on the weight. For details, see [PTQConfig Configuration Description](#ptqconfig-configuration-description).
2. Use the apply and convert APIs to convert a network into a quantized network. In the deployment phase, information statistics and quantitative calculation are not performed, and only the network structure is converted into a quantized network.
3. Load the quantized weights to the quantized network for inference.

### PTQConfig Configuration Description

You can customize PTQConfig to enable different quantization capabilities. For details about PTQConfig, see [the API document](https://www.mindspore.cn/golden_stick/docs/en/r1.1.0/ptq/mindspore_gs.ptq.PTQConfig.html#mindspore_gs.ptq.PTQConfig). The following shows the configuration examples of some algorithms.

> **A** indicates activation, **W** indicates weight, **C** indicates KVCache, and the number indicates bits. For example, **A16W8** indicates a quantization where activations are represented as float16 and weights as int8.

- A16W8 quantization

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

## Case Analysis

### Post-training Quantization

The following provides a complete process of quantization and deployment of the PTQ algorithm and the RoundToNearest algorithm on the Llama2 network.

- [PTQ algorithm](https://www.mindspore.cn/golden_stick/docs/en/r1.1.0/ptq/ptq.html): supports 8-bit weight quantization, 8-bit full quantization, and KVCacheInt8 quantization. SmoothQuant can be used to improve the quantization accuracy. Combining different quantization algorithms can improve the quantization inference performance.
- [RoundToNearest algorithm](https://www.mindspore.cn/golden_stick/docs/en/r1.1.0/ptq/round_to_nearest.html): the simplest 8-bit PTQ algorithm, which supports linear weight quantization and KVCacheInt8 quantization. This algorithm will be discarded in the future. You are advised to use the PTQ algorithm.

### Perceptual Quantization Training

- [SimQAT algorithm](https://www.mindspore.cn/golden_stick/docs/en/r1.1.0/quantization/simulated_quantization.html): a basic quantization aware algorithm based on the pseudo-quantization technology.
- [SLB quantization algorithm](https://www.mindspore.cn/golden_stick/docs/en/r1.1.0/quantization/slb.html): a non-linear low-bit quantization aware algorithm.

### Pruning

- [SCOP pruning algorithm example](https://www.mindspore.cn/golden_stick/docs/en/r1.1.0/pruner/scop.html): a structured weight pruning algorithm.
