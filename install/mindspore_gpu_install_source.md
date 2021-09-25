# 源码编译方式安装MindSpore GPU版本

<!-- TOC -->

- [源码编译方式安装MindSpore GPU版本](#源码编译方式安装mindspore-gpu版本)
    - [确认系统环境信息](#确认系统环境信息)
    - [从代码仓下载源码](#从代码仓下载源码)
    - [编译MindSpore](#编译mindspore)
    - [安装MindSpore](#安装mindspore)
    - [验证是否成功安装](#验证是否成功安装)
    - [升级MindSpore版本](#升级mindspore版本)

<!-- /TOC -->

<a href="https://gitee.com/mindspore/docs/blob/master/install/mindspore_gpu_install_source.md" target="_blank"><img src="https://gitee.com/mindspore/docs/raw/master/resource/_static/logo_source.png"></a>

本文档介绍如何在GPU环境的Linux系统上，使用源码编译方式快速安装MindSpore。

详细步骤可以参考社区提供的实践——[在Linux上体验源码编译安装MindSpore GPU版本](https://www.mindspore.cn/news/newschildren?id=401)，在此感谢社区成员[飞翔的企鹅](https://gitee.com/zhang_yi2020)的分享。

## 确认系统环境信息

- 确认安装64位操作系统，其中Ubuntu 18.04是经过验证的。

- 确认安装[GCC 7.3.0版本](http://ftp.gnu.org/gnu/gcc/gcc-7.3.0/gcc-7.3.0.tar.gz)。

- 确认安装[gmp 6.1.2版本](https://gmplib.org/download/gmp/gmp-6.1.2.tar.xz)。

- 确认安装Python 3.7.5或3.9.0版本。如果未安装或者已安装其他版本的Python，可以选择下载并安装：

    - Python 3.7.5版本 64位，下载地址：[官网](https://www.python.org/ftp/python/3.7.5/Python-3.7.5.tgz)或[华为云](https://mirrors.huaweicloud.com/python/3.7.5/Python-3.7.5.tgz)。
    - Python 3.9.0版本 64位，下载地址：[官网](https://www.python.org/ftp/python/3.9.0/Python-3.9.0.tgz)或[华为云](https://mirrors.huaweicloud.com/python/3.9.0/Python-3.9.0.tgz)。

- 确认安装[CMake 3.18.3及以上版本](https://cmake.org/download/)。
    - 安装完成后将CMake添加到系统环境变量。

- 确认安装[patch 2.5及以上版本](http://ftp.gnu.org/gnu/patch/)。
    - 安装完成后将patch添加到系统环境变量中。

- 确认安装[Autoconf 2.69及以上版本](https://www.gnu.org/software/autoconf)（可使用系统自带版本）。

- 确认安装[Libtool 2.4.6-29.fc30及以上版本](https://www.gnu.org/software/libtool)（可使用系统自带版本）。

- 确认安装[Automake 1.15.1及以上版本](https://www.gnu.org/software/automake)（可使用系统自带版本）。

- 确认安装[Flex 2.5.35及以上版本](https://github.com/westes/flex/)。

- 确认安装[wheel 0.32.0及以上版本](https://pypi.org/project/wheel/)。

- 确认安装[CUDA 10.1](https://developer.nvidia.com/cuda-10.1-download-archive-base)配套[cuDNN 7.6.X版本](https://developer.nvidia.com/rdp/cudnn-archive) 或者 [CUDA 11.1](https://developer.nvidia.com/cuda-11.1.0-download-archive)配套[cuDNN 8.0.X版本](https://developer.nvidia.com/rdp/cudnn-archive#a-collapse805-111)。  
    - CUDA安装后，若CUDA没有安装在默认位置，需要设置环境变量PATH（如：`export PATH=/usr/local/cuda-${version}/bin:$PATH`）和`LD_LIBRARY_PATH`（如：`export LD_LIBRARY_PATH=/usr/local/cuda-${version}/lib64:$LD_LIBRARY_PATH`），详细安装后的设置可参考[CUDA安装手册](https://docs.nvidia.com/cuda/cuda-installation-guide-linux/index.html#post-installation-actions)。

- 确认安装[OpenMPI 4.0.3版本](https://www.open-mpi.org/faq/?category=building#easy-build)（可选，单机多卡/多机多卡训练需要）。

- 确认安装[OpenSSL 1.1.1及以上版本](https://github.com/openssl/openssl.git)。
    - 安装完成后设置环境变量`export OPENSSL_ROOT_DIR=“OpenSSL安装目录”`。

- 确认安装配套CUDA 10.1的[NCCL 2.7.6版本](https://docs.nvidia.com/deeplearning/sdk/nccl-install-guide/index.html#debian)或者配套CUDA 11.1[NCCL 2.7.8版本](https://docs.nvidia.com/deeplearning/sdk/nccl-install-guide/index.html#debian)（可选，单机多卡/多机多卡训练需要）。

- 确认安装[TensorRT-7.2.2.3](https://developer.nvidia.com/nvidia-tensorrt-download)（可选，Serving推理需要）。

- 确认安装[NUMA 2.0.11及以上版本](https://github.com/numactl/numactl)。
    如果未安装，使用如下命令下载安装：

    ```bash
    apt-get install libnuma-dev
    ```

- 确认安装git工具。  
    如果未安装，使用如下命令下载安装：

    ```bash
    apt-get install git
    ```

## 从代码仓下载源码

```bash
git clone https://gitee.com/mindspore/mindspore.git
```

## 编译MindSpore

在源码根目录下执行如下命令。

```bash
bash build.sh -e gpu
```

其中：

`build.sh`中默认的编译线程数为8，如果编译机性能较差可能会出现编译错误，可在执行中增加-j{线程数}来减少线程数量。如`bash build.sh -e gpu -j4`。

## 安装MindSpore

```bash
pip install output/mindspore_gpu-{version}-{python_version}-linux_x86_64.whl -i https://pypi.tuna.tsinghua.edu.cn/simple
```

其中：

- 在联网状态下，安装whl包时会自动下载mindspore安装包的依赖项（依赖项详情参见[setup.py](https://gitee.com/mindspore/mindspore/blob/master/setup.py)中的required_package），其余情况需自行安装。运行模型时，需要根据[ModelZoo](https://gitee.com/mindspore/models/tree/master)中不同模型指定的requirements.txt安装额外依赖，常见依赖可以参考[requirements.txt](https://gitee.com/mindspore/mindspore/blob/master/requirements.txt)。
- `{version}`表示MindSpore版本号，例如安装1.1.0版本MindSpore时，`{version}`应写为1.1.0。  
- `{python_version}`表示用户的Python版本，Python版本为3.7.5时，`{python_version}`应写为`cp37-cp37m`。Python版本为3.9.0时，则写为`cp39-cp39`。

## 验证是否成功安装

方法一：

```bash
python -c "import mindspore;mindspore.run_check()"
```

如果输出：

```text
mindspore version: 版本号
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

## 升级MindSpore版本

当需要升级MindSpore版本时，可执行如下命令：

- 直接在线升级

    ```bash
    pip install --upgrade mindspore-gpu
    ```

    注意：1.3.0及以上版本升级时，默认选择CUDA11版本，若仍希望使用CUDA10版本，请选择相应的完整wheel安装包。

- 本地源码编译升级

    在源码根目录下执行编译脚本`build.sh`成功后，在`output`目录下找到编译生成的whl安装包，然后执行命令进行升级。

    ```bash
    pip install --upgrade mindspore_gpu-{version}-{python_version}-linux_{arch}.whl
    ```
