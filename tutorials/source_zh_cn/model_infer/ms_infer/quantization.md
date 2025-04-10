# 模型量化

[![查看源文件](https://mindspore-website.obs.cn-north-4.myhuaweicloud.com/website-images/r2.6.0/resource/_static/logo_source.svg)](https://gitee.com/mindspore/docs/blob/r2.6.0/tutorials/source_zh_cn/model_infer/ms_infer/quantization.md)

## 概述

MindSpore是一个全场景的AI框架。当模型部署到端侧或者其他轻量化设备上时，对于部署的内存、功耗、时延等有各种限制，因此在部署前需要对模型进行压缩。

MindSpore的模型压缩能力由 [MindSpore Golden Stick](https://www.mindspore.cn/golden_stick/docs/zh-CN/r1.1.0/index.html) 提供，MindSpore Golden Stick是华为诺亚团队和华为MindSpore团队联合设计开发的一个模型压缩算法集，为MindSpore提供了一系列模型压缩算法，支持A16W8、A16W4、A8W8和KVCache等量化方式。详细资料可前往 [MindSpore Golden Stick官方资料](https://www.mindspore.cn/golden_stick/docs/zh-CN/r1.1.0/index.html) 查看。

## 模型量化基本流程

为了便于用户对MindSpore Golden Stick模型量化基本流程的了解，这里以量化算法为例，给出基本的使用方法。

### 基本流程

MindSpore Golden Stick量化算法主要可以分为两个阶段：量化阶段和部署阶段。量化阶段是部署前提前完成的，主要的工作是：收集权重的分布、计算量化参数、量化权重数据、插入反量化节点。部署阶段通常是指在生产环境，使用MindSpore框架对量化后的模型进行推理的过程。

MindSpore Golden Stick主要通过`PTQConfig`来自定义如何量化和部署，通过`apply`和`convert`接口实现量化和部署过程。`PTQConfig`中可配置是否对权重、激活和KVCache进行量化及量化到的bit位，同时也可配置数据校准策略。详细说明可参考[PTQConfig的配置说明](#ptqconfig的配置说明)。

MindSpore Golden Stick的量化步骤如下:

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

1. 使用[nn.Cell定义网络](https://www.mindspore.cn/docs/zh-CN/r2.0/api_python/nn/mindspore.nn.Cell.html)，训练模型后得到模型的浮点权重，在推理过程中，加载该模型的浮点权重。上述例子对该过程进行了简化，直接创建网络，使用初始浮点权重进行量化。
2. 使用PTQConfig配置mode为量化模式，后端为Ascend，对权重进行8bit量化。详细说明可参考[PTQConfig的配置说明](#ptqconfig的配置说明)。
3. 使用apply接口将网络转换为伪量化网络，根据`PTQConfig`中的配置统计量化对象的信息。
4. 使用convert接口对上一步的伪量化网络进行真实量化，得到量化后的网络。

量化结束后，可使用量化后的模型进行推理，步骤如下:

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

1. 使用PTQConfig配置mode为部署模式，后端为Ascend，对权重进行8bit量化。详细说明可参考[PTQConfig的配置说明](#ptqconfig的配置说明)。
2. 使用apply和convert接口将网络转化为量化网络。部署阶段不进行实际的信息统计和量化计算，只将网络结构转换为量化网络。
3. 加载量化后的权重至量化网络中，进行推理。

### PTQConfig的配置说明

可通过自定义PTQConfig的配置来启用不同的量化能力，PTQConfig的含义可以参考其[API文档](https://www.mindspore.cn/golden_stick/docs/zh-CN/r1.1.0/ptq/mindspore_gs.ptq.PTQConfig.html#mindspore_gs.ptq.PTQConfig)，这里我们展示这几种算法的配置样例：

> A表示激活，W表示权重，C表示KVCache，数字代表bit位。例如：A16W8表示激活为float16，权重为int8的量化。

- A16W8权重量化

    ```python
    from mindspore import dtype as msdtype
    from mindspore_gs.ptq import PTQConfig, OutliersSuppressionType

    ptq_config = PTQConfig(weight_quant_dtype=msdtype.int8,  act_quant_dtype=None,  kvcache_quant_dtype=None,
                        outliers_suppression=OutliersSuppressionType.NONE)
    ```

- A8W8量化

    > A8W8量化基于[SmoothQuant](https://gitcode.com/gh_mirrors/smo/smoothquant/overview)算法，PTQConfig提供outliers_suppression字段控制是否进行smooth操作。

    ```python
    from mindspore import dtype as msdtype
    from mindspore_gs.ptq import PTQConfig, OutliersSuppressionType

    ptq_config = PTQConfig(weight_quant_dtype=msdtype.int8, act_quant_dtype=msdtype.int8, kvcache_quant_dtype=None,
                        outliers_suppression=OutliersSuppressionType.SMOOTH)
    ```

- KVCache int8量化

    ```python
    from mindspore import dtype as msdtype
    from mindspore_gs.ptq import PTQConfig, OutliersSuppressionType

    ptq_config = PTQConfig(weight_quant_dtype=None, act_quant_dtype=None, kvcache_quant_dtype=msdtype.int8,
                        outliers_suppression=OutliersSuppressionType.NONE)
    ```

## 实例讲解

### 训练后量化实例讲解

下面给出了PTQ算法和RoundToNearest算法在Llama2网络上量化与部署的完整流程：

- [PTQ算法示例](https://www.mindspore.cn/golden_stick/docs/zh-CN/r1.1.0/ptq/ptq.html)：训练后量化算法，支持8bit权重量化、8bit全量化、KVCacheInt8量化；支持使用SmoothQuant提升量化精度；支持不同算法间的组合量化算法提升量化推理性能。
- [RoundToNearest算法示例](https://www.mindspore.cn/golden_stick/docs/zh-CN/r1.1.0/ptq/round_to_nearest.html)：最简单的8bit训练后量化算法，支持Linear的权重量化和KVCacheInt8量化。该算法后续会被废弃，推荐直接使用PTQ算法。

### 感知量化训练实例讲解

- [SimQAT算法示例](https://www.mindspore.cn/golden_stick/docs/zh-CN/r1.1.0/quantization/simulated_quantization.html)：一种基础的基于伪量化技术的感知量化算法。
- [SLB量化算法示例](https://www.mindspore.cn/golden_stick/docs/zh-CN/r1.1.0/quantization/slb.html)：一种非线性的低比特感知量化算法。

### 剪枝方法实例讲解

- [SCOP剪枝算法示例](https://www.mindspore.cn/golden_stick/docs/zh-CN/r1.1.0/pruner/scop.html)：一个结构化权重剪枝算法。
