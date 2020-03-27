# 安装MindSpore

本文档介绍如何在Nvidia GPU的环境上快速安装MindSpore。

<!-- TOC -->

- [安装MindSpore](#安装mindspore)
    - [环境要求](#环境要求)
        - [硬件要求](#硬件要求)
        - [系统要求和软件依赖](#系统要求和软件依赖)
        - [Conda安装（可选）](#conda安装可选)
    - [安装指南](#安装指南)
        - [从源码编译安装](#从源码编译安装)
    - [安装验证](#安装验证)
- [安装MindArmour](#安装mindarmour)

<!-- /TOC -->

## 环境要求

### 硬件要求

- Nvidia GPU

### 系统要求和软件依赖

| 版本号 | 操作系统 | 可执行文件安装依赖 | 源码编译安装依赖 |
| ---- | :--- | :--- | :--- |
| MindSpore master | Ubuntu 16.04（及以上） x86_64 | - [Python](https://www.python.org/downloads/) 3.7.5 <br> - [CUDA 9.2](https://developer.nvidia.com/cuda-92-download-archive) / [CUDA 10.1](https://developer.nvidia.com/cuda-10.1-download-archive-base) <br> - [CuDNN](https://developer.nvidia.com/rdp/cudnn-archive) >= 7.6 <br> - [OpenMPI](https://www.open-mpi.org/faq/?category=building#easy-build) 3.1.5 （可选，单机多卡/多机多卡训练需要） <br> - [NCCL](https://docs.nvidia.com/deeplearning/sdk/nccl-install-guide/index.html#debian) 2.4.8-1 （可选，单机多卡/多机多卡训练需要） <br> - 其他依赖项参见[requirements.txt](https://gitee.com/mindspore/mindspore/blob/master/requirements.txt) | **编译依赖：**<br> - [Python](https://www.python.org/downloads/) 3.7.5 <br> - [wheel](https://pypi.org/project/wheel/) >= 0.32.0 <br> - [CMake](https://cmake.org/download/) >= 3.14.1 <br> - [GCC](https://gcc.gnu.org/releases.html) 7.3.0 <br> - [patch](http://ftp.gnu.org/gnu/patch/) >= 2.5 <br> - [Autoconf](https://www.gnu.org/software/autoconf) >= 2.64 <br> - [Libtool](https://www.gnu.org/software/libtool) >= 2.4.6 <br> - [Automake](https://www.gnu.org/software/automake) >= 1.15.1 <br> - [CUDA 9.2](https://developer.nvidia.com/cuda-92-download-archive) / [CUDA 10.1](https://developer.nvidia.com/cuda-10.1-download-archive-base) <br> - [CuDNN](https://developer.nvidia.com/rdp/cudnn-archive) >= 7.6 <br> **安装依赖：**<br> 与可执行文件安装依赖相同 |

- Ubuntu版本为18.04时，GCC 7.3.0可以直接通过apt命令安装。
- 在联网状态下，安装whl包时会自动下载requirements.txt中的依赖项，其余情况需自行安装。

### Conda安装（可选）

1. Conda安装包下载路径如下。

   - [X86 Anaconda](https://www.anaconda.com/distribution/) 或 [X86 Miniconda](https://docs.conda.io/en/latest/miniconda.html)

2. 创建并激活Python环境。

    ```bash
    conda create -n {your_env_name} python=3.7.5
    conda activate {your_env_name}
    ```

> Conda是强大的Python环境管理工具，建议初学者上网查阅更多资料。

## 安装指南

### 从源码编译安装

1. 从代码仓下载源码。

    ```bash
    git clone https://gitee.com/mindspore/mindspore.git
    ```

2. 在源码根目录下执行如下命令编译MindSpore。
    ```bash
	bash build.sh -e gpu -M on -z
    ```
    > - 在执行上述命令前，需保证可执行文件cmake和patch所在路径已加入环境变量PATH中。
    > - build.sh中会执行git clone获取第三方依赖库的代码，请提前确保git的网络设置正确可用。
    > - build.sh中默认的编译线程数为8，如果编译机性能较差可能会出现编译错误，可在执行中增加-j{线程数}来减少线程数量。如`bash build.sh -e gpu -M on -z -j4`。

3. 执行如下命令安装MindSpore。

    ```bash
    chmod +x build/package/mindspore-{version}-cp37-cp37m-linux_{arch}.whl
    pip install build/package/mindspore-{version}-cp37-cp37m-linux_{arch}.whl
    ```

## 安装验证

- 安装好MindSpore之后，执行如下python脚本：

    ```bash
    import numpy as np
    from mindspore import Tensor
    from mindspore.ops import functional as F
    import mindspore.context as context

    context.set_context(device_target="GPU")
    x = Tensor(np.ones([1,3,3,4]).astype(np.float32))
    y = Tensor(np.ones([1,3,3,4]).astype(np.float32))
    print(F.tensor_add(x, y))
    ```

- 若出现如下结果，即安装验证通过。

    ```bash
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

# 安装MindArmour

当您进行AI模型安全研究或想要增强AI应用模型的防护能力时，可以选装MindArmour。

## 环境要求

### 系统要求和软件依赖

| 版本号                 | 操作系统            | 可执行文件安装依赖                                           | 源码编译安装依赖         |
| ---------------------- | :------------------ | :----------------------------------------------------------- | :----------------------- |
| MindArmour master | Ubuntu 16.04（及以上） x86_64 | - [Python](https://www.python.org/downloads/) 3.7.5 <br> - MindSpore master <br> - 其他依赖项参见[setup.py](https://gitee.com/mindspore/mindarmour/blob/master/setup.py) | 与可执行文件安装依赖相同 |

- 在联网状态下，安装whl包时会自动下载setup.py中的依赖项，其余情况需自行安装。

## 安装指南

### 从源码编译安装

1. 从代码仓下载源码。

   ```bash
   git clone https://gitee.com/mindspore/mindarmour.git
   ```

2. 在源码根目录下，执行如下命令编译并安装MindArmour。

   ```bash
   cd mindarmour
   python setup.py install
   ```

3. 执行如下命令，如果没有提示`No module named 'mindarmour'`等加载错误的信息，则说明安装成功。

   ```bash
   python -c 'import mindarmour'
   ```

