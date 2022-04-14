# 源码编译方式安装MindSpore GPU版本

<!-- TOC -->

- [源码编译方式安装MindSpore GPU版本](#源码编译方式安装mindspore-gpu版本)
    - [环境准备（自动，推荐）](#环境准备自动推荐)
    - [环境准备（手动）](#环境准备手动)
        - [安装CUDA](#安装cuda)
        - [安装cuDNN](#安装cudnn)
        - [安装Python](#安装python)
        - [安装wheel和setuptools](#安装wheel和setuptools)
        - [安装GCC、git等依赖](#安装gccgit等依赖)
        - [安装CMake](#安装cmake)
        - [安装Open MPI（可选）](#安装open-mpi可选)
        - [安装LLVM（可选）](#安装llvm可选)
        - [安装TensorRT（可选）](#安装tensorrt可选)
    - [从代码仓下载源码](#从代码仓下载源码)
    - [编译MindSpore](#编译mindspore)
    - [安装MindSpore](#安装mindspore)
    - [验证是否成功安装](#验证是否成功安装)
    - [升级MindSpore版本](#升级mindspore版本)

<!-- /TOC -->

<a href="https://gitee.com/mindspore/docs/blob/r1.7/install/mindspore_gpu_install_source.md" target="_blank"><img src="https://gitee.com/mindspore/docs/raw/r1.7/resource/_static/logo_source.png"></a>

本文档介绍如何在GPU环境的Linux系统上，使用源码编译方式快速安装MindSpore。下面以Ubuntu 18.04为例说明MindSpore编译安装步骤。

- 如果您想在一个全新的带有GPU的Ubuntu 18.04上配置一个可以编译MindSpore的环境，可以使用[自动安装脚本](https://gitee.com/mindspore/mindspore/raw/r1.7/scripts/install/ubuntu-gpu-source.sh)进行一键式配置，参见[环境准备（自动，推荐）](#环境准备自动推荐)小节。自动安装脚本会安装编译MindSpore所需的依赖。

- 如果您的系统已经安装了部分依赖，如CUDA，Python，GCC等，则推荐参照[环境准备（手动）](#环境准备手动)小节的安装步骤手动安装。

## 环境准备（自动，推荐）

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

自动安装脚本需要更改软件源配置以及通过APT安装依赖，所以需要申请root权限。使用以下命令获取自动安装脚本并执行。通过自动安装脚本配置的环境，仅支持编译MindSpore>=1.6.0。

```bash
wget https://gitee.com/mindspore/mindspore/raw/r1.7/scripts/install/ubuntu-gpu-source.sh
# 默认安装Python 3.7，CUDA 11.1
bash -i ./ubuntu-gpu-source.sh
# 如需指定安装Python 3.9和CUDA 10.1，并且安装可选依赖Open MPI，使用以下方式
# PYTHON_VERSION=3.9 CUDA_VERSION=10.1 OPENMPI=on bash -i ./ubuntu-gpu-source.sh
```

该脚本会执行以下操作：

- 更改软件源配置为华为云源。
- 安装MindSpore所需的编译依赖，如GCC，CMake等。
- 通过APT安装Python3和pip3，并设为默认。
- 下载CUDA和cuDNN并安装。
- 如果OPENMPI设置为`on`，则安装Open MPI。
- 如果LLVM设置为`on`，则安装LLVM。

自动安装脚本执行完成后，需要重新打开终端窗口以使环境变量生效，然后跳转到[从代码仓下载源码](#从代码仓下载源码)小节下载编译MindSpore。

更多的用法请参看脚本头部的说明。

## 环境准备（手动）

下表列出了编译安装MindSpore GPU所需的系统环境和第三方依赖。

|软件名称|版本|作用|
|-|-|-|
|Ubuntu|18.04|编译和运行MindSpore的操作系统|
|[CUDA](#安装cuda)|10.1或11.1|MindSpore GPU使用的并行计算架构|
|[cuDNN](#安装cudnn)|7.6.x或8.0.x|MindSpore GPU使用的深度神经网络加速库|
|[Python](#安装python)|3.7-3.9|MindSpore的使用依赖Python环境|
|[wheel](#安装wheel和setuptools)|0.32.0及以上|MindSpore使用的Python打包工具|
|[setuptools](#安装wheel和setuptools)|44.0及以上|MindSpore使用的Python包管理工具|
|[GCC](#安装gccgit等依赖)|7.3.0到9.4.0之间|用于编译MindSpore的C++编译器|
|[git](#安装gccgit等依赖)|-|MindSpore使用的源代码管理工具|
|[CMake](#安装cmake)|3.18.3及以上|编译构建MindSpore的工具|
|[Autoconf](#安装gccgit等依赖)|2.69及以上版本|编译构建MindSpore的工具|
|[Libtool](#安装gccgit等依赖)|2.4.6-29.fc30及以上版本|编译构建MindSpore的工具|
|[Automake](#安装gccgit等依赖)|1.15.1及以上版本|编译构建MindSpore的工具|
|[gmp](#安装gccgit等依赖)|6.1.2|MindSpore使用的多精度算术库|
|[Flex](#安装gccgit等依赖)|2.5.35及以上版本|MindSpore使用的词法分析器|
|[tclsh](#安装gccgit等依赖)|-|MindSpore sqlite编译依赖|
|[patch](#安装gccgit等依赖)|2.5及以上|MindSpore使用的源代码补丁工具|
|[NUMA](#安装gccgit等依赖)|2.0.11及以上|MindSpore使用的非一致性内存访问库|
|[Open MPI](#安装open-mpi可选)|4.0.3|MindSpore使用的高性能消息传递库（可选，单机多卡/多机多卡训练需要）|
|[LLVM](#安装llvm可选)|12.0.1|MindSpore使用的编译器框架（可选，图算融合以及稀疏计算需要）|
|[TensorRT](#安装tensorrt可选)|7.2.2|MindSpore使用的高性能深度学习推理SDK（可选，Serving推理需要）|

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

### 安装Python

[Python](https://www.python.org/)可通过多种方式进行安装。

- 通过Conda安装Python。

    安装Miniconda：

    ```bash
    cd /tmp
    curl -O https://mirrors.tuna.tsinghua.edu.cn/anaconda/miniconda/Miniconda3-py37_4.10.3-Linux-x86_64.sh
    bash Miniconda3-py37_4.10.3-Linux-x86_64.sh -b
    cd -
    . ~/miniconda3/etc/profile.d/conda.sh
    conda init bash
    ```

    安装完成后，可以为Conda设置清华源加速下载，参考[此处](https://mirrors.tuna.tsinghua.edu.cn/help/anaconda/)。

    创建虚拟环境，以Python 3.7.5为例：

    ```bash
    conda create -n mindspore_py37 python=3.7.5 -y
    conda activate mindspore_py37
    ```

- 通过APT安装Python，命令如下。

    ```bash
    sudo apt-get update
    sudo apt-get install software-properties-common -y
    sudo add-apt-repository ppa:deadsnakes/ppa -y
    sudo apt-get install python3.7 python3.7-dev python3.7-distutils python3-pip -y
    # 将新安装的Python设为默认
    sudo update-alternatives --install /usr/bin/python python /usr/bin/python3.7 100
    # 安装pip
    python -m pip install pip -i https://repo.huaweicloud.com/repository/pypi/simple
    sudo update-alternatives --install /usr/bin/pip pip ~/.local/bin/pip3.7 100
    pip config set global.index-url https://repo.huaweicloud.com/repository/pypi/simple
    ```

    若要安装其他Python版本，只需更改命令中的`3.7`。

可以通过以下命令查看Python版本。

```bash
python --version
```

### 安装wheel和setuptools

在安装完成Python后，使用以下命令安装。

```bash
pip install wheel
pip install -U setuptools
```

### 安装GCC、git等依赖

可以通过以下命令安装GCC，git，Autoconf，Libtool，Automake，gmp，Flex，tclsh，patch，NUMA。

```bash
sudo apt-get install gcc-7 git automake autoconf libtool libgmp-dev tcl patch libnuma-dev flex -y
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

### 安装CMake

可以通过以下命令安装[CMake](https://cmake.org/)。

```bash
wget -O - https://apt.kitware.com/keys/kitware-archive-latest.asc 2>/dev/null | sudo apt-key add -
sudo apt-add-repository "deb https://apt.kitware.com/ubuntu/ $(lsb_release -cs) main"
sudo apt-get install cmake -y
```

### 安装Open MPI（可选）

可以通过以下命令编译安装[Open MPI](https://www.open-mpi.org/)。

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

### 安装LLVM（可选）

可以通过以下命令安装[LLVM](https://llvm.org/)。

```bash
wget -O - https://apt.llvm.org/llvm-snapshot.gpg.key | sudo apt-key add -
sudo add-apt-repository "deb http://apt.llvm.org/bionic/ llvm-toolchain-bionic-12 main"
sudo apt-get update
sudo apt-get install llvm-12-dev -y
```

### 安装TensorRT（可选）

完成CUDA和cuDNN的安装后，在[TensorRT下载页面](https://developer.nvidia.com/nvidia-tensorrt-7x-download)下载配套CUDA 11.1的TensorRT 7.2.2，注意选择下载TAR格式的安装包。假设下载的文件名为`TensorRT-7.2.2.3.Ubuntu-18.04.x86_64-gnu.cuda-11.1.cudnn8.0.tar.gz`。使用以下命令安装TensorRT。

```bash
tar xzf TensorRT-7.2.2.3.Ubuntu-18.04.x86_64-gnu.cuda-11.1.cudnn8.0.tar.gz
cd TensorRT-7.2.2.3
echo -e "export TENSORRT_HOME=$PWD" >> ~/.bashrc
echo -e "export LD_LIBRARY_PATH=\$TENSORRT_HOME/lib:\$LD_LIBRARY_PATH" >> ~/.bashrc
source ~/.bashrc
cd -
```

## 从代码仓下载源码

```bash
git clone https://gitee.com/mindspore/mindspore.git
```

## 编译MindSpore

执行编译前，请确保nvcc的安装路径已经添加到`PATH`与`LD_LIBRARY_PATH`环境变量中，如果没有添加，以安装在默认路径的CUDA11为例，可以执行如下操作：

```bash
export PATH=/usr/local/cuda-11.1/bin:$PATH
export LD_LIBRARY_PATH=/usr/local/cuda-11.1/lib64:$LD_LIBRARY_PATH
```

如果之前安装了其他CUDA版本或者CUDA安装路径不同，只需替换上述命令中的`/usr/local/cuda-11.1`为当前安装的CUDA路径。

进入MindSpore根目录，然后执行编译脚本。

```bash
cd mindspore
bash build.sh -e gpu -S on
```

其中：

- `build.sh`中默认的编译线程数为8，如果编译机性能较差可能会出现编译错误，可在执行中增加-j{线程数}来减少线程数量。如`bash build.sh -e gpu -j4`。
- 默认从github下载依赖源码，当-S选项设置为`on`时，从对应的gitee镜像下载。
- 关于`build.sh`更多用法请参看脚本头部的说明。

## 安装MindSpore

```bash
pip install output/mindspore_gpu-*.whl -i https://pypi.tuna.tsinghua.edu.cn/simple
```

在联网状态下，安装MindSpore时会自动下载MindSpore安装包的依赖项（依赖项详情参见[setup.py](https://gitee.com/mindspore/mindspore/blob/r1.7/setup.py)中的required_package），其余情况需自行安装。运行模型时，需要根据[ModelZoo](https://gitee.com/mindspore/models/tree/r1.7/)中不同模型指定的requirements.txt安装额外依赖，常见依赖可以参考[requirements.txt](https://gitee.com/mindspore/mindspore/blob/r1.7/requirements.txt)。

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
    pip install --upgrade mindspore_gpu-*.whl
    ```
