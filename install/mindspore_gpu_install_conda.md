# Conda方式安装MindSpore GPU版本

<!-- TOC -->

- [Conda方式安装MindSpore GPU版本](#conda方式安装mindspore-gpu版本)
    - [自动安装](#自动安装)
    - [手动安装](#手动安装)
        - [安装CUDA](#安装cuda)
        - [安装cuDNN](#安装cudnn)
        - [安装Conda](#安装conda)
        - [安装GCC和gmp](#安装gcc和gmp)
        - [安装Open MPI-可选](#安装open-mpi-可选)
        - [安装TensorRT-可选](#安装tensorrt-可选)
        - [创建并进入Conda虚拟环境](#创建并进入conda虚拟环境)
        - [安装MindSpore](#安装mindspore)
    - [验证是否成功安装](#验证是否成功安装)

<!-- /TOC -->

<a href="https://gitee.com/mindspore/docs/blob/master/install/mindspore_gpu_install_conda.md" target="_blank"><img src="https://mindspore-website.obs.cn-north-4.myhuaweicloud.com/website-images/master/resource/_static/logo_source.png"></a>

[Conda](https://docs.conda.io/en/latest/)是一个开源跨平台语言无关的包管理与环境管理系统，允许用户方便地安装不同版本的二进制软件包与该计算平台需要的所有库。

本文档介绍如何在GPU环境的Linux系统上，使用Conda方式快速安装MindSpore。下面以Ubuntu 18.04为例说明MindSpore安装步骤。

- 如果您想在一个全新的带有GPU的Ubuntu 18.04上通过Conda安装MindSpore，可以使用[自动安装脚本](https://gitee.com/mindspore/mindspore/raw/master/scripts/install/ubuntu-gpu-conda.sh)进行一键式安装，参见[自动安装](#自动安装)小节。自动安装脚本会安装MindSpore及其所需的依赖。

- 如果您的系统已经安装了部分依赖，如CUDA，Conda，GCC等，则推荐参照[手动安装](#手动安装)小节的安装步骤手动安装。

## 自动安装

在使用自动安装脚本之前，需要确保系统正确安装了NVIDIA GPU驱动。CUDA 10.1要求最低显卡驱动版本为418.39；CUDA 11.1要求最低显卡驱动版本为450.80.02。执行以下指令检查驱动版本。

```bash
nvidia-smi
```

如果未安装GPU驱动，执行如下命令安装。

```bash
sudo apt-get update
sudo apt-get install ubuntu-drivers-common
sudo ubuntu-drivers autoinstall
```

安装完成后请重启系统。

自动安装脚本需要更改软件源配置以及通过APT安装依赖，所以在执行中需要申请root权限。使用以下命令获取自动安装脚本并执行。自动安装脚本仅支持安装MindSpore>=1.6.0。

```bash
wget https://gitee.com/mindspore/mindspore/raw/master/scripts/install/ubuntu-gpu-conda.sh
# 默认安装Python 3.7，CUDA 11.1以及最新版本的MindSpore
bash -i ./ubuntu-gpu-conda.sh
# 如需指定Python，CUDA和MindSpore版本，以Python 3.9，CUDA 10.1和MindSpore 1.6.0为例，使用以下方式
# PYTHON_VERSION=3.9 CUDA_VERSION=10.1 MINDSPORE_VERSION=1.6.0 bash -i ./ubuntu-gpu-conda.sh
```

该脚本会执行以下操作：

- 更改软件源配置为华为云源。
- 安装MindSpore所需的依赖，如GCC，gmp。
- 下载CUDA和cuDNN并安装。
- 安装Conda并为MindSpore创建虚拟环境。
- 通过Conda安装MindSpore GPU版本。
- 如果OPENMPI设置为`on`，则安装Open MPI。

自动安装脚本执行完成后，需要重新打开终端窗口以使环境变量生效。自动安装脚本会为MindSpore创建名为`mindspore_pyXX`的虚拟环境。其中`XX`为Python版本，如Python 3.7则虚拟环境名为`mindspore_py37`。执行以下命令查看所有虚拟环境。

```bash
conda env list
```

要激活虚拟环境，以Python 3.7为例，执行以下命令。

```bash
conda activate mindspore_py37
```

更多的用法请参看脚本头部的说明。

## 手动安装

下表列出了安装MindSpore所需的系统环境和第三方依赖。

|软件名称|版本|作用|
|-|-|-|
|Ubuntu|18.04|运行MindSpore的操作系统|
|[CUDA](#安装cuda)|10.1或11.1|MindSpore GPU使用的并行计算架构|
|[cuDNN](#安装cudnn)|7.6.x或8.0.x|MindSpore GPU使用的深度神经网络加速库|
|[Conda](#安装conda)|Anaconda3或Miniconda3|Python环境管理工具|
|[GCC](#安装gcc和gmp)|7.3.0到9.4.0之间|用于编译MindSpore的C++编译器|
|[gmp](#安装gcc和gmp)|6.1.2|MindSpore使用的多精度算术库|
|[Open MPI](#安装open-mpi-可选)|4.0.3|MindSpore使用的高性能消息传递库（可选，单机多卡/多机多卡训练需要）|
|[TensorRT](#安装tensorrt-可选)|7.2.2|MindSpore使用的高性能深度学习推理SDK（可选，Serving推理需要）|

下面给出第三方依赖的安装方法。

### 安装CUDA

MindSpore GPU支持CUDA 10.1和CUDA 11.1。NVIDIA官方给出了多种安装方式和安装指导，详情可查看[CUDA下载页面](https://developer.nvidia.com/cuda-toolkit-archive)和[CUDA安装指南](https://docs.nvidia.com/cuda/cuda-installation-guide-linux/index.html)。
下面仅给出Linux系统使用runfile方式安装的指导。

在安装CUDA前需要先安装相关依赖，执行以下命令。

```bash
sudo apt-get install linux-headers-$(uname -r) gcc-7
```

CUDA 10.1要求最低显卡驱动版本为418.39；CUDA 11.1要求最低显卡驱动版本为450.80.02。可以执行`nvidia-smi`指令确认显卡驱动版本。如果驱动版本不满足要求，CUDA安装过程中可以选择同时安装驱动，安装驱动后需要重启系统。

使用以下命令安装CUDA 11.1（推荐）。

```bash
wget https://developer.download.nvidia.com/compute/cuda/11.1.1/local_installers/cuda_11.1.1_455.32.00_linux.run
sudo sh cuda_11.1.1_455.32.00_linux.run
echo -e "export PATH=/usr/local/cuda-11.1/bin:\$PATH" >> ~/.bashrc
echo -e "export LD_LIBRARY_PATH=/usr/local/cuda-11.1/lib64:\$LD_LIBRARY_PATH" >> ~/.bashrc
source ~/.bashrc
```

或者使用以下命令安装CUDA 10.1。

```bash
wget https://developer.download.nvidia.com/compute/cuda/10.1/Prod/local_installers/cuda_10.1.243_418.87.00_linux.run
sudo sh cuda_10.1.243_418.87.00_linux.run
echo -e "export PATH=/usr/local/cuda-10.1/bin:\$PATH" >> ~/.bashrc
echo -e "export LD_LIBRARY_PATH=/usr/local/cuda-10.1/lib64:\$LD_LIBRARY_PATH" >> ~/.bashrc
source ~/.bashrc
```

### 安装cuDNN

完成CUDA的安装后，在[cuDNN页面](https://developer.nvidia.com/zh-cn/cudnn)登录并下载对应的cuDNN安装包。如果之前安装了CUDA 10.1，下载配套CUDA 10.1的cuDNN v7.6.x；如果之前安装了CUDA 11.1，下载配套CUDA 11.1的cuDNN v8.0.x。注意下载后缀名为tgz的压缩包。假设下载的cuDNN包名为`cudnn.tgz`，安装的CUDA版本为11.1，执行以下命令安装cuDNN。

```bash
tar -zxvf cudnn.tgz
sudo cp cuda/include/cudnn.h /usr/local/cuda-11.1/include
sudo cp cuda/lib64/libcudnn* /usr/local/cuda-11.1/lib64
sudo chmod a+r /usr/local/cuda-11.1/include/cudnn.h /usr/local/cuda-11.1/lib64/libcudnn*
```

如果之前安装了其他CUDA版本或者CUDA安装路径不同，只需替换上述命令中的`/usr/local/cuda-11.1`为当前安装的CUDA路径。

### 安装Conda

执行以下指令安装Miniconda。

```bash
cd /tmp
curl -O https://mirrors.tuna.tsinghua.edu.cn/anaconda/miniconda/Miniconda3-py37_4.10.3-Linux-x86_64.sh
bash Miniconda3-py37_4.10.3-Linux-x86_64.sh -b
cd -
. ~/miniconda3/etc/profile.d/conda.sh
conda init bash
```

安装完成后，可以为Conda设置清华源加速下载，参考[此处](https://mirrors.tuna.tsinghua.edu.cn/help/anaconda/)。

### 安装GCC和gmp

可以通过以下命令安装GCC和gmp。

```bash
sudo apt-get install gcc-7 libgmp-dev -y
```

如果要安装更高版本的GCC，使用以下命令安装GCC 8。

```bash
sudo apt-get install gcc-8 -y
```

或者安装GCC 9（注意，GCC 9不兼容CUDA 10.1）。

```bash
sudo apt-get install software-properties-common -y
sudo add-apt-repository ppa:ubuntu-toolchain-r/test
sudo apt-get update
sudo apt-get install gcc-9 -y
```

### 安装Open MPI-可选

可以通过以下命令编译安装Open MPI。

```bash
curl -O https://download.open-mpi.org/release/open-mpi/v4.0/openmpi-4.0.3.tar.gz
tar xzf openmpi-4.0.3.tar.gz
cd openmpi-4.0.3
./configure --prefix=/usr/local/openmpi-4.0.3
make
sudo make install
echo -e "export PATH=/usr/local/openmpi-4.0.3/bin:\$PATH" >> ~/.bashrc
echo -e "export LD_LIBRARY_PATH=/usr/local/openmpi-4.0.3/lib:\$LD_LIBRARY_PATH" >> ~/.bashrc
source ~/.bashrc
cd -
```

### 安装TensorRT-可选

完成CUDA和cuDNN的安装后，在[TensorRT下载页面](https://developer.nvidia.com/nvidia-tensorrt-7x-download)下载配套CUDA 11.1的TensorRT 7.2.2，注意选择下载TAR格式的安装包。假设下载的文件名为`TensorRT-7.2.2.3.Ubuntu-18.04.x86_64-gnu.cuda-11.1.cudnn8.0.tar.gz`。使用以下命令安装TensorRT。

```bash
tar xzf TensorRT-7.2.2.3.Ubuntu-18.04.x86_64-gnu.cuda-11.1.cudnn8.0.tar.gz
cd TensorRT-7.2.2.3
echo -e "export TENSORRT_HOME=$PWD" >> ~/.bashrc
echo -e "export LD_LIBRARY_PATH=\$TENSORRT_HOME/lib:\$LD_LIBRARY_PATH" >> ~/.bashrc
source ~/.bashrc
cd -
```

### 创建并进入Conda虚拟环境

根据您希望使用的Python版本创建对应的Conda虚拟环境并进入虚拟环境。

如果您希望使用Python3.7.5版本：

```bash
conda create -n mindspore_py37 python=3.7.5 -y
conda activate mindspore_py37
```

如果希望使用其他版本Python，只需更改以上命令中的Python版本。当前支持Python 3.7，Python 3.8和Python 3.9。

### 安装MindSpore

确认您处于Conda虚拟环境中，并执行如下命令安装最新版本的MindSpore。如需安装其他版本，可参考[版本列表](https://www.mindspore.cn/versions)在`mindspore-gpu=`后指定版本号。

CUDA 10.1版本：

```bash
conda install mindspore-gpu cudatoolkit=10.1 -c mindspore -c conda-forge
```

CUDA 11.1版本：

```bash
conda install mindspore-gpu cudatoolkit=11.1 -c mindspore -c conda-forge
```

在联网状态下，安装MindSpore时会自动下载MindSpore安装包的依赖项（依赖项详情参见[setup.py](https://gitee.com/mindspore/mindspore/blob/master/setup.py)中的required_package），其余情况需自行安装。运行模型时，需要根据[ModelZoo](https://gitee.com/mindspore/models/tree/master/)中不同模型指定的requirements.txt安装额外依赖，常见依赖可以参考[requirements.txt](https://gitee.com/mindspore/mindspore/blob/master/requirements.txt)。

## 验证是否成功安装

运行MindSpore GPU版本前，请确保nvcc的安装路径已经添加到`PATH`与`LD_LIBRARY_PATH`环境变量中，如果没有添加，以安装在默认路径的CUDA11为例，可以执行如下操作：

```bash
export PATH=/usr/local/cuda-11.1/bin:$PATH
export LD_LIBRARY_PATH=/usr/local/cuda-11.1/lib64:$LD_LIBRARY_PATH
```

如果之前安装了其他CUDA版本或者CUDA安装路径不同，只需替换上述命令中的`/usr/local/cuda-11.1`为当前安装的CUDA路径。

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
