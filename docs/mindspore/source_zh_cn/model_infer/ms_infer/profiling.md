# 模型性能Profiler

[![查看源文件](https://mindspore-website.obs.cn-north-4.myhuaweicloud.com/website-images/master/resource/_static/logo_source.svg)](https://gitee.com/mindspore/docs/blob/master/docs/mindspore/source_zh_cn/model_infer/ms_infer/profiling.md)

MindSpore中提供了profiler接口，可以对神经网络的性能进行采集。目前支持AICORE算子、AICPU算子、HostCPU算子、内存、设备通信、集群等数据的分析。

样例：

```python
import numpy as np
import mindspore
from mindspore import Tensor
from mindspore.train import Model
from mindspore import Profiler

input_data = Tensor(np.random.randint(0, 255, [1, 1, 32, 32]), mindspore.float32)
# Define the network structure of LeNet5. Refer to
# https://gitee.com/mindspore/docs/blob/master/docs/mindspore/code/lenet.py
# Init Profiler
# Note that the Profiler should be initialized before model.predict
profiler = Profiler()
model = Model(LeNet5())
result = model.predict(input_data)

# Profiler end
profiler.analyse()

```

推理方面性能调试方式与训练基本一致，收集到性能数据后，可参考：[性能调试](https://www.mindspore.cn/docs/zh-CN/master/model_train/optimize/profiler.html)进行性能分析。推理上重点关注算子性能分析、计算量性能分析、Timeline分析等。

详细接口参考：[mindspore.Profiler](https://www.mindspore.cn/docs/zh-CN/master/api_python/mindspore/mindspore.Profiler.html?highlight=profiler#mindspore.Profiler)。
