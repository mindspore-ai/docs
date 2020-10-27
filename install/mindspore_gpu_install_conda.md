# Conda方式安装MindSpore

<!-- TOC -->

- [Conda方式安装MindSpore](#conda方式安装mindspore)
    - [确认系统环境信息](#确认系统环境信息)
    - [安装Conda](#安装conda)
    - [添加Conda清华镜像源](#添加conda清华镜像源)
    - [创建并激活Conda环境](#创建并激活conda环境)
    - [安装MindSpore](#安装mindspore)
    - [验证是否成功安装](#验证是否成功安装)

<!-- /TOC -->

<a href="https://gitee.com/mindspore/docs/blob/master/install/mindspore_gpu_install_conda.md" target="_blank"><img src="https://gitee.com/mindspore/docs/raw/master/resource/_static/logo_source.png"></a>

本文档介绍如何在GPU环境的Linux系统上，使用conda方式快速安装MindSpore。

## 确认系统环境信息

- 确认安装Ubuntu18.04是64位操作系统。
- 确认安装[GCC](http://ftp.gnu.org/gnu/gcc/gcc-7.3.0/gcc-7.3.0.tar.gz)为7.3.0版本。
- 确认安装[CUDA10.1](https://developer.nvidia.com/cuda-10.1-download-archive-base)  
    CUDA安装后，若CUDA没有安装在默认位置，需要设置环境变量PATH（如：`export PATH=/usr/local/cuda-${version}/bin:$PATH`）和`LD_LIBRARY_PATH`（如：`export LD_LIBRARY_PATH=/usr/local/cuda-${version}/lib64:$LD_LIBRARY_PATH`），详细安装后的设置可参考[CUDA安装手册](https://docs.nvidia.com/cuda/cuda-installation-guide-linux/index.html#post-installation-actions)。
- 确认安装[CuDNN](https://developer.nvidia.com/cuda-10.1-download-archive-base)，选择7.6.X版本。
- 确认安装[OpenMPI](https://www.open-mpi.org/faq/?category=building#easy-build)，选择3.1.5版本（可选，单机多卡/多机多卡训练需要）
- 确认安装[NCCL](https://docs.nvidia.com/deeplearning/sdk/nccl-install-guide/index.html#debian)，选择2.7.6-1 （可选，单机多卡/多机多卡训练需要）
- 确认安装[gmp](https://gmplib.org/download/gmp/gmp-6.1.2.tar.xz)，版本选择 6.1.2

## 安装Conda

- 官网下载地址：[X86 Anaconda](https://www.anaconda.com/distribution/) 或 [X86 Miniconda](https://docs.conda.io/en/latest/miniconda.html)
- 清华镜像源下载地址：[X86 Anaconda](https://mirrors.tuna.tsinghua.edu.cn/anaconda/archive/Anaconda3-2020.02-Linux-x86_64.sh)

## 添加Conda清华镜像源

从清华源镜像源下载Conda安装包的可跳过此步操作。

```bash
conda config --add channels https://mirrors.tuna.tsinghua.edu.cn/anaconda/pkgs/free/
conda config --add channels https://mirrors.tuna.tsinghua.edu.cn/anaconda/pkgs/main/
conda config --set show_channel_urls yes
```

## 创建并激活Conda环境

```bash
conda create -n mindspore python=3.7.5
conda activate mindspore
```

## 安装MindSpore

```bash
pip install https://ms-release.obs.cn-north-4.myhuaweicloud.com/{version}/MindSpore/gpu/ubuntu_x86/cuda-10.1/mindspore_gpu-{version}-cp37-cp37m-linux_{arch}.whl --trusted-host ms-release.obs.cn-north-4.myhuaweicloud.com -i https://mirrors.huaweicloud.com/repository/pypi/simple
```

> `{version}`表示MindSpore版本号，例如下载1.0.1版本MindSpore时，`{version}`应写为1.0.1。  
> `{arch}`表示系统架构，例如使用的Linux系统是x86架构64位时，`{arch}`应写为`x86_64`。如果系统是ARMv8架构64位，则写为`aarch64`。

## 验证是否成功安装

```python
import numpy as np
from mindspore import Tensor
import mindspore.ops as ops
import mindspore.context as context

context.set_context(device_target="GPU")
x = Tensor(np.ones([1,3,3,4]).astype(np.float32))
y = Tensor(np.ones([1,3,3,4]).astype(np.float32))
print(ops.tensor_add(x, y))
```

如果输出：

```text
[[[ 2.  2.  2.  2.],
    [ 2.  2.  2.  2.],
    [ 2.  2.  2.  2.]],

    [[ 2.  2.  2.  2.],
    [ 2.  2.  2.  2.],
    [ 2.  2.  2.  2.]],

    [[ 2.  2.  2.  2.],
    [ 2.  2.  2.  2.],
    [ 2.  2.  2.  2.]]]
```

说明MindSpore安装成功了。
