# Conda方式安装MindSpore GPU版本

<!-- TOC -->

- [Conda方式安装MindSpore GPU版本](#conda方式安装mindspore-gpu版本)
    - [确认系统环境信息](#确认系统环境信息)
    - [创建并进入Conda虚拟环境](#创建并进入conda虚拟环境)
    - [安装MindSpore](#安装mindspore)
    - [验证是否成功安装](#验证是否成功安装)

<!-- /TOC -->

[![查看源文件](https://gitee.com/mindspore/docs/raw/r1.6/resource/_static/logo_source.png)](https://gitee.com/mindspore/docs/blob/r1.6/install/mindspore_gpu_install_conda.md)

[Conda](https://docs.conda.io/en/latest/)是一个开源跨平台语言无关的包管理与环境管理系统，允许用户方便地安装不同版本的二进制软件包与该计算平台需要的所有库。

本文档介绍如何在GPU环境的Linux系统上，使用Conda方式快速安装MindSpore。

在确认系统环境信息的过程中，如需了解如何安装第三方依赖软件，可以参考社区提供的实践——[在Linux上体验源码编译安装MindSpore GPU版本](https://www.mindspore.cn/news/newschildren?id=401)中的第三方依赖软件安装相关部分，在此感谢社区成员[飞翔的企鹅](https://gitee.com/zhang_yi2020)的分享。

## 确认系统环境信息

- 确认安装64位操作系统，[glibc](https://www.gnu.org/software/libc/)>=2.17，其中Ubuntu 18.04是经过验证的。
- 确认安装[GCC 7.3.0版本](https://ftp.gnu.org/gnu/gcc/gcc-7.3.0/gcc-7.3.0.tar.gz)。
- 确认安装[CUDA 10.1](https://developer.nvidia.com/cuda-10.1-download-archive-base)配套[cuDNN 7.6.X版本](https://developer.nvidia.com/rdp/cudnn-archive) 或者 [CUDA 11.1](https://developer.nvidia.com/cuda-11.1.0-download-archive)配套[cuDNN 8.0.X版本](https://developer.nvidia.com/rdp/cudnn-archive)。
    - CUDA安装后，若CUDA没有安装在默认位置，需要设置环境变量PATH（如：`export PATH=/usr/local/cuda-${version}/bin:$PATH`）和`LD_LIBRARY_PATH`（如：`export LD_LIBRARY_PATH=/usr/local/cuda-${version}/lib64:$LD_LIBRARY_PATH`），详细安装后的设置可参考[CUDA安装手册](https://docs.nvidia.com/cuda/cuda-installation-guide-linux/index.html#post-installation-actions)。
- 确认安装[OpenMPI 4.0.3版本](https://www.open-mpi.org/faq/?category=building#easy-build)（可选，单机多卡/多机多卡训练需要）。
- 确认安装[OpenSSL 1.1.1及以上版本](https://github.com/openssl/openssl.git)。
    - 安装完成后设置环境变量`export OPENSSL_ROOT_DIR=“OpenSSL安装目录”`。
- 确认安装配套CUDA 10.1的[NCCL 2.7.6版本](https://docs.nvidia.com/deeplearning/sdk/nccl-install-guide/index.html#debian)或者配套CUDA 11.1[NCCL 2.7.8版本](https://docs.nvidia.com/deeplearning/sdk/nccl-install-guide/index.html#debian)（可选，单机多卡/多机多卡训练需要）。
- 确认安装[TensorRT-7.2.2](https://developer.nvidia.com/nvidia-tensorrt-download)（可选，Serving推理需要）。
- 确认安装[gmp 6.1.2版本](https://gmplib.org/download/gmp/gmp-6.1.2.tar.xz)。
- 确认安装与当前系统兼容的Conda版本。
    - 如果您喜欢Conda提供的完整能力，可以选择下载[Anaconda3](https://repo.anaconda.com/archive/)。
    - 如果您需要节省磁盘空间，或者喜欢自定义安装Conda软件包，可以选择下载[Miniconda3](https://repo.anaconda.com/miniconda/)。

## 创建并进入Conda虚拟环境

根据您希望使用的Python版本创建对应的Conda虚拟环境并进入虚拟环境。
如果您希望使用Python3.7.5版本：

```bash
conda create -n mindspore_py37 -c conda-forge python=3.7.5
conda activate mindspore_py37
```

如果您希望使用Python3.9.0版本：

```bash
conda create -n mindspore_py39 -c conda-forge python=3.9.0
conda activate mindspore_py39
```

## 安装MindSpore

执行如下命令安装MindSpore。

CUDA 10.1 版本：

```bash
conda install mindspore-gpu={version} cudatoolkit=10.1 -c mindspore -c conda-forge
```

CUDA 11.1 版本：

```bash
conda install mindspore-gpu={version} cudatoolkit=11.1 -c mindspore -c conda-forge
```

其中：

- 在联网状态下，安装Conda安装包时会自动下载MindSpore安装包的依赖项（依赖项详情参见[setup.py](https://gitee.com/mindspore/mindspore/blob/r1.6/setup.py)中的required_package），其余情况需自行安装。运行模型时，需要根据[ModelZoo](https://gitee.com/mindspore/models/tree/r1.6/)中不同模型指定的requirements.txt安装额外依赖，常见依赖可以参考[requirements.txt](https://gitee.com/mindspore/mindspore/blob/r1.6/requirements.txt)。
- `{version}`表示MindSpore版本号，例如安装1.5.0-rc1版本MindSpore时，`{version}`应写为1.5.0rc1。

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
