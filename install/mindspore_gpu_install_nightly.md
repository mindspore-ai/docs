# pip方式安装MindSpore GPU Nightly版本

<!-- TOC -->

- [pip方式安装MindSpore GPU Nightly版本](#pip方式安装mindspore-gpu-nightly版本)
    - [安装MindSpore与依赖软件](#安装mindspore与依赖软件)
        - [安装CUDA](#安装cuda)
        - [安装cuDNN](#安装cudnn)
        - [安装Python](#安装python)
        - [安装GCC和gmp](#安装gcc和gmp)
        - [安装Open MPI（可选）](#安装open-mpi可选)
        - [安装TensorRT（可选）](#安装tensorrt可选)
    - [下载安装MindSpore](#下载安装mindspore)
    - [验证是否成功安装](#验证是否成功安装)
    - [升级MindSpore版本](#升级mindspore版本)

<!-- /TOC -->

<a href="https://gitee.com/mindspore/docs/blob/r1.7/install/mindspore_gpu_install_pip.md" target="_blank"><img src="https://mindspore-website.obs.cn-north-4.myhuaweicloud.com/website-images/r1.7/resource/_static/logo_source.png"></a>

MindSpore Nightly是包含当前最新功能与bugfix的预览版本，但是可能未经完整的测试与验证，希望体验最新功能或者问题修复的用户可以使用该版本。

本文档介绍如何在GPU环境的Linux系统上，使用pip方式快速安装MindSpore Nightly。

在确认系统环境信息的过程中，如需了解如何安装第三方依赖软件，可以参考社区提供的实践——[在Linux上体验源码编译安装MindSpore GPU版本](https://www.mindspore.cn/news/newschildren?id=401)中的第三方依赖软件安装相关部分，在此感谢社区成员[飞翔的企鹅](https://gitee.com/zhang_yi2020)的分享。

## 安装MindSpore与依赖软件

下表列出了编译安装MindSpore GPU所需的系统环境和第三方依赖。

| 软件名称                      | 版本             | 作用                                                         |
| ----------------------------- | ---------------- | ------------------------------------------------------------ |
| Ubuntu                        | 18.04            | 编译和运行MindSpore的操作系统                                |
| [CUDA](#安装cuda)             | 10.1或11.1       | MindSpore GPU使用的并行计算架构                              |
| [cuDNN](#安装cudnn)           | 7.6.x或8.0.x     | MindSpore GPU使用的深度神经网络加速库                        |
| [Python](#安装python)         | 3.7-3.9          | MindSpore的使用依赖Python环境                                |
| [GCC](#安装gcc和gmp)          | 7.3.0到9.4.0之间 | 用于编译MindSpore的C++编译器                                 |
| [gmp](#安装gcc和gmp)          | 6.1.2            | MindSpore使用的多精度算术库                                  |
| [Open MPI](#安装open-mpi可选) | 4.0.3            | MindSpore使用的高性能消息传递库（可选，单机多卡/多机多卡训练需要） |
| [TensorRT](#安装tensorrt可选) | 7.2.2            | MindSpore使用的高性能深度学习推理SDK（可选，Serving推理需要） |

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

### 安装Open MPI（可选）

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

## 下载安装MindSpore

CUDA 11.1 版本：

```bash
pip install mindspore-cuda11-dev -i https://pypi.tuna.tsinghua.edu.cn/simple
```

其中：

- 当前MindSpore GPU Nightly仅提供CUDA11版本。
- 在联网状态下，安装whl包时会自动下载mindspore安装包的依赖项（依赖项详情参见[setup.py](https://gitee.com/mindspore/mindspore/blob/r1.7/setup.py)中的required_package），其余情况需自行安装。运行模型时，需要根据[ModelZoo](https://gitee.com/mindspore/models/tree/r1.7/)中不同模型指定的requirements.txt安装额外依赖，常见依赖可以参考[requirements.txt](https://gitee.com/mindspore/mindspore/blob/r1.7/requirements.txt)。
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

## 升级MindSpore版本

当需要升级MindSpore版本时，可执行如下命令：

```bash
pip install --upgrade mindspore-cuda11-dev=={version}
```

其中：

- 升级到rc版本时，需要手动指定`{version}`为rc版本号，例如1.6.0rc1.dev20211125；如果希望自动升级到最新版本，`=={version}`字段可以缺省。

注意：当前MindSpore GPU nightly仅提供CUDA11版本，若仍希望使用CUDA10版本，请参考源码编译指导在安装CUDA10的环境上自行编译。
