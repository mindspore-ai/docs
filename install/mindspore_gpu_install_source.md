# 源码编译方式安装MindSpore GPU版本

<!-- TOC -->

- [源码编译方式安装MindSpore GPU版本](#源码编译方式安装mindspore-gpu版本)
    - [安装依赖软件](#安装依赖软件)
        - [安装CUDA](#安装cuda)
        - [安装cuDNN](#安装cudnn)
        - [安装Python](#安装python)
        - [安装wheel setuptools PyYAML和Numpy](#安装wheel-setuptools-pyyaml和numpy)
        - [安装GCC git等依赖](#安装gcc-git等依赖)
        - [安装CMake](#安装cmake)
        - [安装LLVM-可选](#安装llvm-可选)
        - [安装TensorRT-可选](#安装tensorrt-可选)
    - [从代码仓下载源码](#从代码仓下载源码)
    - [编译MindSpore](#编译mindspore)
    - [安装MindSpore](#安装mindspore)
    - [验证是否成功安装](#验证是否成功安装)
    - [升级MindSpore版本](#升级mindspore版本)

<!-- /TOC -->

[![查看源文件](https://mindspore-website.obs.cn-north-4.myhuaweicloud.com/website-images/r2.6.0rc1/resource/_static/logo_source.svg)](https://gitee.com/mindspore/docs/blob/r2.6.0rc1/install/mindspore_gpu_install_source.md)

本文档介绍如何在GPU环境的Linux系统上，使用源码编译方式快速安装MindSpore。下面以Ubuntu 18.04为例说明MindSpore编译安装步骤。

## 安装依赖软件

下表列出了编译安装MindSpore GPU所需的系统环境和第三方依赖。

|软件名称|版本|作用|
|-|-|-|
|Ubuntu|18.04|编译和运行MindSpore的操作系统|
|[CUDA](#安装cuda)|10.1或11.1或11.6|MindSpore GPU使用的并行计算架构|
|[cuDNN](#安装cudnn)|7.6.x或8.0.x或8.5.x|MindSpore GPU使用的深度神经网络加速库|
|[Python](#安装python)|3.9-3.11|MindSpore的使用依赖Python环境|
|[wheel](#安装wheel-setuptools-pyyaml和numpy)|0.32.0及以上|MindSpore使用的Python打包工具|
|[setuptools](#安装wheel-setuptools-pyyaml和numpy)|44.0及以上|MindSpore使用的Python包管理工具|
|[PyYAML](#安装wheel-setuptools-pyyaml和numpy)|6.0-6.0.2|MindSpore里的算子编译功能依赖PyYAML模块|
|[Numpy](#安装wheel-setuptools-pyyaml和numpy)|1.19.3-1.26.4|MindSpore里的Numpy相关功能依赖Numpy模块|
|[GCC](#安装gcc-git等依赖)|7.3.0-9.4.0|用于编译MindSpore的C++编译器|
|[git](#安装gcc-git等依赖)|-|MindSpore使用的源代码管理工具|
|[CMake](#安装cmake)|3.22.2及以上|编译构建MindSpore的工具|
|[Autoconf](#安装gcc-git等依赖)|2.69及以上|编译构建MindSpore的工具|
|[Libtool](#安装gcc-git等依赖)|2.4.6-29.fc30及以上|编译构建MindSpore的工具|
|[Automake](#安装gcc-git等依赖)|1.15.1及以上|编译构建MindSpore的工具|
|[Flex](#安装gcc-git等依赖)|2.5.35及以上|MindSpore使用的词法分析器|
|[tclsh](#安装gcc-git等依赖)|-|MindSpore sqlite编译依赖|
|[patch](#安装gcc-git等依赖)|2.5及以上|MindSpore使用的源代码补丁工具|
|[NUMA](#安装gcc-git等依赖)|2.0.11及以上|MindSpore使用的非一致性内存访问库|
|[LLVM](#安装llvm-可选)|12.0.1|MindSpore使用的编译器框架（可选，图算融合以及稀疏计算需要）|
|[TensorRT](#安装tensorrt-可选)|7.2.2或8.4|MindSpore使用的高性能深度学习推理SDK（可选，Serving推理需要）|

下面给出第三方依赖的安装方法。

### 安装CUDA

MindSpore GPU支持CUDA 10.1、CUDA 11.1和CUDA 11.6。NVIDIA官方给出了多种安装方式和安装指导，详情可查看[CUDA下载页面](https://developer.nvidia.com/cuda-toolkit-archive)和[CUDA安装指南](https://docs.nvidia.com/cuda/cuda-installation-guide-linux/index.html)。
下面仅给出Linux系统使用runfile方式安装的指导。

在安装CUDA前需要先安装相关依赖，执行以下命令。

```bash
sudo apt-get install linux-headers-$(uname -r) gcc-7
```

CUDA 10.1要求最低显卡驱动版本为418.39；CUDA 11.1要求最低显卡驱动版本为450.80.02；CUDA 11.6要求最低显卡驱动为510.39.01。可以执行`nvidia-smi`命令确认显卡驱动版本。如果驱动版本不满足要求，CUDA安装过程中可以选择同时安装驱动，安装驱动后需要重启系统。

使用以下命令安装CUDA 11.6（推荐）。

```bash
wget https://developer.download.nvidia.com/compute/cuda/11.6.0/local_installers/cuda_11.6.0_510.39.01_linux.run
sudo sh cuda_11.6.0_510.39.01_linux.run
echo -e "export PATH=/usr/local/cuda-11.6/bin:\$PATH" >> ~/.bashrc
echo -e "export LD_LIBRARY_PATH=/usr/local/cuda-11.6/lib64:\$LD_LIBRARY_PATH" >> ~/.bashrc
source ~/.bashrc
```

或者使用以下命令安装CUDA 11.1。

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

当默认路径`/usr/local/cuda`存在安装包的时候，LD_LIBRARY_PATH环境变量不起作用；原因是MindSpore采用DT_RPATH方式支持无环境变量启动，减少用户设置；DT_RPATH优先级比LD_LIBRARY_PATH环境变量高。

### 安装cuDNN

完成CUDA的安装后，在[cuDNN页面](https://developer.nvidia.com/cudnn)登录并下载对应的cuDNN安装包。如果之前安装了CUDA 10.1，下载配套CUDA 10.1的cuDNN v7.6.x；如果之前安装了CUDA 11.1，下载配套CUDA 11.1的cuDNN v8.0.x；如果之前安装了CUDA 11.6，下载配套CUDA 11.6的cuDNN v8.5.x。注意下载后缀名为tgz的压缩包。假设下载的cuDNN包名为`cudnn.tgz`，安装的CUDA版本为11.6，执行以下命令安装cuDNN。

```bash
tar -zxvf cudnn.tgz
sudo cp cuda/include/cudnn*.h /usr/local/cuda-11.6/include
sudo cp cuda/lib64/libcudnn* /usr/local/cuda-11.6/lib64
sudo chmod a+r /usr/local/cuda-11.6/include/cudnn*.h /usr/local/cuda-11.6/lib64/libcudnn*
```

如果之前安装了其他CUDA版本或者CUDA安装路径不同，只需替换以上命令中的`/usr/local/cuda-11.6`为当前安装的CUDA路径。

### 安装Python

[Python](https://www.python.org/)可通过多种方式进行安装。

- 通过Conda安装Python。

    安装Miniconda：

    ```bash
    cd /tmp
    curl -O https://mirrors.tuna.tsinghua.edu.cn/anaconda/miniconda/Miniconda3-py37_4.10.3-Linux-$(arch).sh
    bash Miniconda3-py37_4.10.3-Linux-$(arch).sh -b
    cd -
    . ~/miniconda3/etc/profile.d/conda.sh
    conda init bash
    ```

    安装完成后，可以为Conda设置清华源加速下载，参考[此处](https://mirrors.tuna.tsinghua.edu.cn/help/anaconda/)。

    创建虚拟环境，以Python 3.9.11为例：

    ```bash
    conda create -n mindspore_py39 python=3.9.11 -y
    conda activate mindspore_py39
    ```

- 通过APT安装Python，命令如下。

    ```bash
    sudo apt-get update
    sudo apt-get install software-properties-common -y
    sudo add-apt-repository ppa:deadsnakes/ppa -y
    sudo apt-get install python3.9 python3.9-dev python3.9-distutils python3-pip -y
    # 将新安装的Python设为默认
    sudo update-alternatives --install /usr/bin/python python /usr/bin/python3.9 100
    # 安装pip
    python -m pip install pip -i https://repo.huaweicloud.com/repository/pypi/simple
    sudo update-alternatives --install /usr/bin/pip pip ~/.local/bin/pip3.9 100
    pip config set global.index-url https://repo.huaweicloud.com/repository/pypi/simple
    ```

    若要安装其他Python版本，只需更改命令中的`3.9`。

可以通过以下命令查看Python版本。

```bash
python --version
```

### 安装wheel setuptools PyYAML和Numpy

在安装完成Python后，使用以下命令安装。

```bash
pip install wheel
pip install -U setuptools
pip install pyyaml
pip install "numpy>=1.19.3,<=1.26.4"
```

运行环境使用的Numpy版本需不小于编译环境的Numpy版本，以保证框架内Numpy相关能力的正常使用。

### 安装GCC git等依赖

可以通过以下命令安装GCC、git、Autoconf、Libtool、Automake、Flex、tclsh、patch和NUMA。

```bash
sudo apt-get install gcc-7 git automake autoconf libtool tcl patch libnuma-dev flex -y
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

### 安装LLVM-可选

可以通过以下命令安装[LLVM](https://llvm.org/)。

```bash
wget -O - https://apt.llvm.org/llvm-snapshot.gpg.key | sudo apt-key add -
sudo add-apt-repository "deb http://apt.llvm.org/bionic/ llvm-toolchain-bionic-12 main"
sudo apt-get update
sudo apt-get install llvm-12-dev -y
```

### 安装TensorRT-可选

完成CUDA和cuDNN的安装后，在[TensorRT下载页面](https://developer.nvidia.com/nvidia-tensorrt-8x-download)下载配套CUDA 11.6的TensorRT 8.4，注意选择下载TAR格式的安装包。假设下载的文件名为`TensorRT-8.4.1.5.Ubuntu-18.04.x86_64-gnu.cuda-11.6.cudnn8.4.tar.gz`。使用以下命令安装TensorRT。

```bash
tar xzf TensorRT-8.4.1.5.Ubuntu-18.04.x86_64-gnu.cuda-11.6.cudnn8.4.tar.gz
cd TensorRT-8.4.1.5
echo -e "export TENSORRT_HOME=$PWD" >> ~/.bashrc
echo -e "export LD_LIBRARY_PATH=\$TENSORRT_HOME/lib:\$LD_LIBRARY_PATH" >> ~/.bashrc
source ~/.bashrc
cd -
```

## 从代码仓下载源码

```bash
git clone -b v2.6.0rc1 https://gitee.com/mindspore/mindspore.git
```

## 编译MindSpore

执行编译前，请确保nvcc的安装路径已经添加到`PATH`与`LD_LIBRARY_PATH`环境变量中，如果没有添加，以安装在默认路径的CUDA11为例，可以执行如下操作：

```bash
export PATH=/usr/local/cuda-11.6/bin:$PATH
export LD_LIBRARY_PATH=/usr/local/cuda-11.6/lib64:$LD_LIBRARY_PATH
export CUDA_HOME=/usr/local/cuda-11.6
```

如果之前安装了其他CUDA版本或者CUDA安装路径不同，只需替换以上命令中的`/usr/local/cuda-11.6`为当前安装的CUDA路径。

进入MindSpore根目录，然后执行编译脚本。

```bash
cd mindspore
bash build.sh -e gpu -S on
```

其中：

- `build.sh`中默认的编译线程数为8，如果编译机性能较差可能会出现编译错误，可在执行中增加`-j{线程数}`来减少线程数量。如`bash build.sh -e gpu -j4`。
- 默认从github下载依赖源码，当-S选项设置为`on`时，从对应的Gitee镜像下载。
- 关于`build.sh`更多用法，请参看脚本头部的说明。

## 安装MindSpore

```bash
pip install output/mindspore-*.whl -i https://pypi.tuna.tsinghua.edu.cn/simple
```

在联网状态下，安装MindSpore时会自动下载MindSpore安装包的依赖项（依赖项详情参见[setup.py](https://gitee.com/mindspore/mindspore/blob/v2.6.0-rc1/setup.py)中的required_package），其余情况需自行安装依赖。

## 验证是否成功安装

**方法一：**

执行以下命令：

```bash
python -c "import mindspore;mindspore.set_device(device_target='GPU');mindspore.run_check()"
```

如果输出：

```text
MindSpore version: 版本号
The result of multiplication calculation is correct, MindSpore has been installed on platform [GPU] successfully!
```

说明MindSpore安装成功了。

**方法二：**

执行以下代码：

```python
import numpy as np
import mindspore as ms
import mindspore.ops as ops

ms.set_device(device_target="GPU")
x = ms.Tensor(np.ones([1,3,3,4]).astype(np.float32))
y = ms.Tensor(np.ones([1,3,3,4]).astype(np.float32))
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

在源码根目录下执行编译脚本`build.sh`成功后，在`output`目录下找到编译生成的whl安装包，然后执行以下命令进行升级。

从MindSpore 1.x升级到MindSpore 2.x版本时，需要先手动卸载旧版本：

```bash
pip uninstall mindspore-gpu
```

然后安装新版本：

```bash
pip install mindspore-*.whl
```

从MindSpore 2.x版本升级到最新版本时，执行以下命令：

 ```bash
pip install --upgrade mindspore-*.whl
```
