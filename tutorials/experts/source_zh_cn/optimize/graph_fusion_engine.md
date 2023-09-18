# 使能图算融合

[![View Source On Gitee](https://mindspore-website.obs.cn-north-4.myhuaweicloud.com/website-images/r2.1/resource/_static/logo_source.svg)](https://gitee.com/mindspore/docs/blob/r2.1/tutorials/experts/source_zh_cn/optimize/graph_fusion_engine.md)

## 概述

图算融合是MindSpore特有的网络性能优化技术。它可以通过自动分析和优化现有网络计算图逻辑，并结合目标硬件能力，对计算图进行计算化简和替代、算子拆分和融合、算子特例化编译等优化，以提升设备计算资源利用率，实现对网络性能的整体优化。相比传统优化技术，图算融合具有多算子跨边界联合优化、与MindSpore AKG（基于Polyhedral的算子编译器）跨层协同、即时编译等独特优势。另外，图算融合只需要用户打开对应配置后，整个优化过程即可自动完成，不需要网络开发人员进行其它额外感知，使得用户可以聚焦网络算法实现。

> MindSpore默认自动安装MindSpore AKG。对于CPU后端并且通过源码安装的MindSpore，需确保已正确安装[llvm 12.0.1版本](https://github.com/llvm/llvm-project/archive/refs/tags/llvmorg-12.0.1.tar.gz)。

图算融合的适用场景包括：

- 对网络执行时间具有较高性能要求的场景；
- 通过拼接基本算子实现自定义组合算子，并希望对这些基本算子进行自动融合，以提升自定义组合算子性能的场景。

## 使用方法

当前图算融合优化默认关闭状态，我们只需在训练脚本中为`context`指定参数`enable_graph_kernel=True`即可启用图算融合：

```python
import mindspore as ms
ms.set_context(enable_graph_kernel=True)
```

> - 图算融合优化可以选择在Graph和PyNative两种模式下使能。在使能之后，会根据计算图特征以及图算优化能力，选择性进行融合优化。并且图算优化能力也可能在不同版本之间会进行变化和演进。例如：PyNative模式当前会选择性对jit子图或者反向子图进行融合优化，另外如某些动态shape算子可能会跳过融合。
> - 对于大部分场景下，图算融合优化通常可以获得正向性能收益以及相同或相近的计算精度。但在极少数场景下，也有可能会出现性能劣化。另外由于算子实现差别，也可能会有一些精度上变化。建议用户结合自身场景选择性使用。
> - 对于CPU平台，图算融合采用了[OpenMP](https://www.openmp.org/)并行计算技术进行算子性能加速。为了获取更好的执行性能，建议配置OMP_NUM_THREADS环境变量以指定OpenMP并行线程数。推荐配置为小于等于当前CPU核数的正整数，如：`export OMP_NUM_THREADS=10`

### 样例脚本

为了说明图算融合优化场景，我们构造了一个简单网络`MyNet`, 包含一个乘法和加法计算。在打开图算融合进行优化之后，这两个计算便会自动合成一个融合算子:

```python
import numpy as np
import mindspore as ms
from mindspore.nn import Cell
import mindspore.ops as ops

ms.set_context(mode=ms.GRAPH_MODE, device_target="GPU")
# save graph ir to view fusion detail.
ms.set_context(save_graphs=2)
# enable graph kernel optimization.
ms.set_context(enable_graph_kernel=True)

class MyNet(Cell):
    def construct(self, x):
        a = ops.mul(x, 2.0)
        res = ops.add(a, 1.0)
        return res

x = np.ones((4, 4)).astype(np.float32) * 0.5
net = MyNet()
result = net(ms.Tensor(x))
print("result: {}".format(result))
```

输出结果：

```text
result: [[2. 2. 2. 2.]
 [2. 2. 2. 2.]
 [2. 2. 2. 2.]
 [2. 2. 2. 2.]]
```

该计算图的融合结果如图1所示，其中左图为未使能图算融合时的对应计算图，右图为使能图算融合后的对应计算图。可以看到该网络中的加法和乘法被融合成一个算子。该融合过程可以通过查看中间IR，或者通过Profiling等工具跟踪算子执行过程进行验证。

![基本算子融合示例](images/graph_kernel_example_fuse_basic.png)

*图1：图算融合优化计算图*

## 自定义组合算子

基于图算融合技术，用户可以很方便地实现高性能的自定义组合算子。其主要流程为：  

1. 在脚本中用基本算子组合的方式实现自定义算子定义和使用；
2. 打开图算融合配置；
3. 图算融合对自定义组合算子中的基本算子自动进行算子融合，并生成高性能融合算子。

相比其它自定义算子方式，这种方式具有对框架无侵入、简单易用等优点。

### 样例脚本

我们构造一个简单网络`MyNet`，并在其中使用了自定义算子`MyOp`。代码样例如下:

```python
import numpy as np
import mindspore as ms
from mindspore.nn import Cell
import mindspore.ops as ops

ms.set_context(mode=ms.GRAPH_MODE, device_target="GPU")
# enable graph kernel optimization.
ms.set_context(enable_graph_kernel=True)

class MyOp(Cell):
    """ my first custom OP composited by basic OPs """
    def construct(self, x, y):
        a = ops.sub(x, y)
        return ops.mul(a, x)

class MyNet(Cell):
    def __init__(self):
        super(MyNet, self).__init__()
        self.my_op = MyOp()

    def construct(self, x, y):
        a = ops.mul(x, 2.0)
        b = ops.pow(a, 3.0)
        res = self.my_op(b, y)
        return res

x = np.ones((4, 4)).astype(np.float32) * 0.2
y = np.ones((4, 4)).astype(np.float32) * 0.3
net = MyNet()
result = net(ms.Tensor(x), ms.Tensor(y))
print("result: {}".format(result))
```

输出结果：

```text
result: [[-0.015104 -0.015104 -0.015104 -0.015104]
 [-0.015104 -0.015104 -0.015104 -0.015104]
 [-0.015104 -0.015104 -0.015104 -0.015104]
 [-0.015104 -0.015104 -0.015104 -0.015104]]
```

该计算图的融合结果如图2所示，其中左图为未使能图算融合时的对应计算图，右图为使能图算融合后的对应计算图。可以看到不仅自定义算子`MyOp`中的基本算子进行了融合，并且与主图中的其他算子也进行了更大范围融合。该融合过程可以通过查看中间IR，或者通过Profiling等工具跟踪算子执行过程进行验证。

![自定义组合算子融合示例](images/graph_kernel_example_custom_op.png)

*图2：自定义组合算子优化计算图*

## FAQs

### Cuda头文件缺失

Akg依赖cuda相关头文件用于生成cuda kernel，若自动搜索头文件失败（提示 **error: cuda_runtime.h: No such file or directory**），请尝试设置相关环境变量：

```bash
# Linux-X86_64系统示例
export CPATH=/usr/local/cuda/targets/x86_64-linux/include:$CPATH
export LD_LIBRARY_PATH=/usr/local/cuda/targets/x86_64-linux/lib:$LD_LIBRARY_PATH
export PATH=/usr/local/cuda/bin:$PATH
```
