# 模型性能Profiler

[![查看源文件](https://mindspore-website.obs.cn-north-4.myhuaweicloud.com/website-images/r2.6.0/resource/_static/logo_source.svg)](https://gitee.com/mindspore/docs/blob/r2.6.0/tutorials/source_zh_cn/model_infer/ms_infer/profiling.md)

MindSpore中提供了profiler接口，可以对神经网络的性能进行采集。目前支持AICORE算子、AICPU算子、HostCPU算子、内存、设备通信、集群等数据的分析。

样例：

```python
import numpy as np
import mindspore
from mindspore import Tensor
from mindspore.train import Model

input_data = Tensor(np.random.randint(0, 255, [1, 1, 32, 32]), mindspore.float32)
# Define the network structure of LeNet5. Refer to
# https://gitee.com/mindspore/docs/blob/r2.6.0/docs/mindspore/code/lenet.py
# Init Profiler
# Note that the Profiler should be initialized before model.predict
with mindspore.profiler.profile() as prof:
    model = Model(LeNet5())
    result = model.predict(input_data)
    # Profiler end
    prof.step()

```

推理方面性能调试方式与训练基本一致，收集到性能数据后，可参考：[性能调试](https://www.mindspore.cn/tutorials/zh-CN/r2.6.0/debug/profiler.html)进行性能分析。推理上重点关注算子性能分析、计算量性能分析、Timeline分析等。

详细接口参考：[mindspore.profiler.profile](https://www.mindspore.cn/docs/zh-CN/r2.6.0/api_python/mindspore/mindspore.profiler.profile.html)。
