# pip方式安装MindSpore GPU Nightly版本

<!-- TOC -->

- [pip方式安装MindSpore GPU Nightly版本](#pip方式安装mindspore-gpu-nightly版本)
    - [确认系统环境信息](#确认系统环境信息)
    - [下载安装MindSpore](#下载安装mindspore)
    - [验证是否成功安装](#验证是否成功安装)
    - [升级MindSpore版本](#升级mindspore版本)

<!-- /TOC -->

<a href="https://gitee.com/mindspore/docs/blob/master/install/mindspore_gpu_install_pip.md" target="_blank"><img src="https://gitee.com/mindspore/docs/raw/master/resource/_static/logo_source.png"></a>

MindSpore Nightly是包含当前最新功能与bugfix的预览版本，但是可能未经完整的测试与验证，希望体验最新功能或者问题修复的用户可以使用该版本。

本文档介绍如何在GPU环境的Linux系统上，使用pip方式快速安装MindSpore Nightly。

在确认系统环境信息的过程中，如需了解如何安装第三方依赖软件，可以参考社区提供的实践——[在Linux上体验源码编译安装MindSpore GPU版本](https://www.mindspore.cn/news/newschildren?id=401)中的第三方依赖软件安装相关部分，在此感谢社区成员[飞翔的企鹅](https://gitee.com/zhang_yi2020)的分享。

## 确认系统环境信息

- 确认安装64位操作系统，[glibc](https://www.gnu.org/software/libc/)>=2.17，其中Ubuntu 18.04是经过验证的。
- 确认安装[GCC 7.3.0版本](https://ftp.gnu.org/gnu/gcc/gcc-7.3.0/gcc-7.3.0.tar.gz)。
- 确认安装[CUDA 11.1](https://developer.nvidia.com/cuda-11.1.0-download-archive)配套[cuDNN 8.0.X版本](https://developer.nvidia.com/rdp/cudnn-archive)。
    - CUDA安装后，若CUDA没有安装在默认位置，需要设置环境变量PATH（如：`export PATH=/usr/local/cuda-${version}/bin:$PATH`）和`LD_LIBRARY_PATH`（如：`export LD_LIBRARY_PATH=/usr/local/cuda-${version}/lib64:$LD_LIBRARY_PATH`），详细安装后的设置可参考[CUDA安装手册](https://docs.nvidia.com/cuda/cuda-installation-guide-linux/index.html#post-installation-actions)。
- 确认安装[OpenMPI 4.0.3版本](https://www.open-mpi.org/faq/?category=building#easy-build)（可选，单机多卡/多机多卡训练需要）。
- 确认安装[OpenSSL 1.1.1及以上版本](https://github.com/openssl/openssl.git)。
    - 安装完成后设置环境变量`export OPENSSL_ROOT_DIR=“OpenSSL安装目录”`。
- 确认安装[TensorRT-7.2.2](https://developer.nvidia.com/nvidia-tensorrt-download)（可选，Serving推理需要）。
- 确认安装[gmp 6.1.2版本](https://gmplib.org/download/gmp/gmp-6.1.2.tar.xz)。
- 确认安装Python 3.7.5或3.9.0版本。如果未安装或者已安装其他版本的Python，可以选择下载并安装：

    - Python 3.7.5版本 64位，下载地址：[官网](https://www.python.org/ftp/python/3.7.5/Python-3.7.5.tgz)或[华为云](https://mirrors.huaweicloud.com/python/3.7.5/Python-3.7.5.tgz)。
    - Python 3.9.0版本 64位，下载地址：[官网](https://www.python.org/ftp/python/3.9.0/Python-3.9.0.tgz)或[华为云](https://mirrors.huaweicloud.com/python/3.9.0/Python-3.9.0.tgz)。

## 下载安装MindSpore

CUDA 11.1 版本：

```bash
pip install mindspore-cuda11-dev -i https://pypi.tuna.tsinghua.edu.cn/simple
```

其中：

- 当前MindSpore GPU Nightly仅提供CUDA11版本。
- 在联网状态下，安装whl包时会自动下载mindspore安装包的依赖项（依赖项详情参见[setup.py](https://gitee.com/mindspore/mindspore/blob/master/setup.py)中的required_package），其余情况需自行安装。运行模型时，需要根据[ModelZoo](https://gitee.com/mindspore/models/tree/master/)中不同模型指定的requirements.txt安装额外依赖，常见依赖可以参考[requirements.txt](https://gitee.com/mindspore/mindspore/blob/master/requirements.txt)。
- pip会自动安装当前最新版本的Nightly版本MindSpore，如果需要安装指定版本，请参照下方升级MindSpore版本相关指导，在下载时手动指定版本。

## 验证是否成功安装

方法一：

```bash
python -c "import mindspore;mindspore.run_check()"
```

如果输出：

```text
MindSpore version: 版本号
The result of multiplication calculation is correct, MindSpore has been installed successfully!
```

说明MindSpore安装成功了。

方法二：

```python
import numpy as np
from mindspore import Tensor
import mindspore.ops as ops
import mindspore.context as context

context.set_context(device_target="GPU")
x = Tensor(np.ones([1,3,3,4]).astype(np.float32))
y = Tensor(np.ones([1,3,3,4]).astype(np.float32))
print(ops.add(x, y))
```

如果输出：

```text
[[[[2. 2. 2. 2.]
   [2. 2. 2. 2.]
   [2. 2. 2. 2.]]

  [[2. 2. 2. 2.]
   [2. 2. 2. 2.]
   [2. 2. 2. 2.]]

  [[2. 2. 2. 2.]
   [2. 2. 2. 2.]
   [2. 2. 2. 2.]]]]
```

说明MindSpore安装成功了。
