# 源码编译方式安装MindSpore Ascend 910版本

<!-- TOC -->

- [源码编译方式安装MindSpore Ascend 910版本](#源码编译方式安装mindspore-ascend-910版本)
    - [环境准备-自动，推荐](#环境准备-自动推荐)
    - [环境准备-手动](#环境准备-手动)
        - [安装Python](#安装python)
        - [安装昇腾AI处理器配套软件包](#安装昇腾ai处理器配套软件包)
        - [安装wheel和setuptools](#安装wheel和setuptools)
        - [安装GCC](#安装gcc)
        - [安装git，gmp，tclsh，patch，NUMA，Flex](#安装gitgmptclshpatchnumaflex)
        - [安装git-lfs](#安装git-lfs)
        - [安装CMake](#安装cmake)
        - [安装Open MPI（可选）](#安装open-mpi可选)
    - [从代码仓下载源码](#从代码仓下载源码)
    - [编译MindSpore](#编译mindspore)
    - [安装MindSpore](#安装mindspore)
    - [配置环境变量](#配置环境变量)
    - [验证是否成功安装](#验证是否成功安装)
    - [升级MindSpore版本](#升级mindspore版本)

<!-- /TOC -->

<a href="https://gitee.com/mindspore/docs/blob/r1.7/install/mindspore_ascend_install_source.md" target="_blank"><img src="https://mindspore-website.obs.cn-north-4.myhuaweicloud.com/website-images/r1.7/resource/_static/logo_source.png"></a>

本文档介绍如何在Ascend 910环境的Linux系统上，使用源码编译方式快速安装MindSpore。

- 如果您想在一个已经配置好昇腾AI处理器配套软件包的EulerOS 2.8上配置一个可以编译MindSpore的环境，可以使用[自动安装脚本](https://gitee.com/mindspore/mindspore/raw/r1.7/scripts/install/euleros-ascend-source.sh)进行一键式配置，参见[环境准备-自动，推荐](#环境准备-自动推荐)小节。自动安装脚本会安装编译MindSpore所需的依赖。

- 如果您的系统已经安装了部分依赖，如Python，GCC等，则推荐参照[环境准备-手动](#环境准备-手动)小节的安装步骤手动安装。

## 环境准备-自动，推荐

在使用自动安装脚本之前，需要确保系统正确安装了昇腾AI处理器配套软件包。如果没有安装，请先参考[安装昇腾AI处理器配套软件包](#安装昇腾ai处理器配套软件包)小节进行安装。

使用以下命令获取自动安装脚本并执行。通过自动安装脚本配置的环境，仅支持编译MindSpore>=1.6.0。

```bash
wget https://gitee.com/mindspore/mindspore/raw/r1.7/scripts/install/euleros-ascend-source.sh
# 默认安装Python 3.7
# 默认LOCAL_ASCEND路径为/usr/local/Ascend
bash -i ./euleros-ascend-source.sh
# 如需指定安装Python 3.9，并且安装可选依赖Open MPI
# 且指定LOCAL_ASCEND路径为/home/xxx/Ascend，使用以下方式
# LOCAL_ASCEND=/home/xxx/Ascend PYTHON_VERSION=3.9 OPENMPI=on bash -i ./euleros-ascend-source.sh
```

该脚本会执行以下操作：

- 安装MindSpore所需的编译依赖，如GCC，CMake等。
- 安装Python3和pip3，并设为默认。
- 如果OPENMPI设置为`on`，则安装Open MPI。

自动安装脚本执行完成后，需要重新打开终端窗口以使环境变量生效。

自动安装脚本会为MindSpore创建名为`mindspore_pyXX`的虚拟环境。其中`XX`为Python版本，如Python 3.7则虚拟环境名为`mindspore_py37`。执行以下命令查看所有虚拟环境。

```bash
conda env list
```

以Python 3.7为例，执行以下命令激活虚拟环境。

```bash
conda activate mindspore_py37
```

现在您可以跳转到[从代码仓下载源码](#从代码仓下载源码)小节开始下载编译MindSpore。

更多的用法请参看脚本头部的说明。

## 环境准备-手动

下表列出了编译安装MindSpore所需的系统环境和第三方依赖。

|软件名称|版本|作用|
|-|-|-|
|Ubuntu 18.04/CentOS 7.6/EulerOS 2.8/OpenEuler 20.03/KylinV10 SP1|-|编译和运行MindSpore的操作系统|
|[Python](#安装python)|3.7-3.9|MindSpore的使用依赖Python环境|
|[昇腾AI处理器配套软件包](#安装昇腾ai处理器配套软件包)|-|MindSpore使用的Ascend平台AI计算库|
|[wheel](#安装wheel和setuptools)|0.32.0及以上|MindSpore使用的Python打包工具|
|[setuptools](#安装wheel和setuptools)|44.0及以上|MindSpore使用的Python包管理工具|
|[GCC](#安装gcc)|7.3.0|用于编译MindSpore的C++编译器|
|[git](#安装gitgmptclshpatchnumaflex)|-|MindSpore使用的源代码管理工具|
|[git-lfs](#安装git-lfs)|-|MindSpore使用的源代码管理工具|
|[CMake](#安装cmake)|3.18.3及以上|编译构建MindSpore的工具|
|[gmp](#安装gitgmptclshpatchnumaflex)|6.1.2|MindSpore使用的多精度算术库|
|[Flex](#安装gitgmptclshpatchnumaflex)|2.5.35及以上版本|MindSpore使用的词法分析器|
|[tclsh](#安装gitgmptclshpatchnumaflex)|-|MindSpore sqlite编译依赖|
|[patch](#安装gitgmptclshpatchnumaflex)|2.5及以上|MindSpore使用的源代码补丁工具|
|[NUMA](#安装gitgmptclshpatchnumaflex)|2.0.11及以上|MindSpore使用的非一致性内存访问库|
|[Open MPI](#安装open-mpi可选)|4.0.3|MindSpore使用的高性能消息传递库（可选，单机多卡/多机多卡训练需要）|

下面给出第三方依赖的安装方法。

### 安装Python

[Python](https://www.python.org/)可通过Conda进行安装。

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

可以通过以下命令查看Python版本。

```bash
python --version
```

### 安装昇腾AI处理器配套软件包

详细安装方法请参考[Ascend Data Center Solution 21.1.0安装指引文档]。

安装包默认安装路径为`/usr/local/Ascend`。安装后确认当前用户有权限访问昇腾AI处理器配套软件包的安装路径，若无权限，需要root用户将当前用户添加到`/usr/local/Ascend`所在的用户组。

安装昇腾AI处理器配套软件包提供的whl包，whl包随配套软件包发布。如果之前安装过昇腾AI处理器配套软件包，需要先使用如下命令卸载相应的包。

```bash
pip uninstall te topi hccl -y
```

默认安装路径使用以下指令安装。如果安装路径不是默认路径，需要将命令中的路径替换为安装路径。

```bash
pip install /usr/local/Ascend/ascend-toolkit/latest/fwkacllib/lib64/topi-*-py3-none-any.whl
pip install /usr/local/Ascend/ascend-toolkit/latest/fwkacllib/lib64/te-*-py3-none-any.whl
pip install /usr/local/Ascend/ascend-toolkit/latest/fwkacllib/lib64/hccl-*-py3-none-any.whl
```

### 安装wheel和setuptools

在安装完成Python后，使用以下命令安装。

```bash
pip install wheel
pip install -U setuptools
```

### 安装GCC

- Ubuntu 18.04可以使用以下命令安装。

    ```bash
    sudo apt-get install gcc-7 -y
    ```

- CentOS 7可以使用以下命令安装。

    ```bash
    sudo yum install centos-release-scl
    sudo yum install devtoolset-7
    ```

    安装完成后，需要使用如下命令切换到GCC 7。

    ```bash
    scl enable devtoolset-7 bash
    ```

- EulerOS和OpenEuler可以使用以下命令安装。

    ```bash
    sudo yum install gcc -y
    ```

### 安装git，gmp，tclsh，patch，NUMA，Flex

- Ubuntu 18.04可以使用以下命令安装。

    ```bash
    sudo apt-get install git libgmp-dev tcl patch libnuma-dev flex -y
    ```

- CentOS 7，EulerOS和OpenEuler可以使用以下命令安装。

    ```bash
    sudo yum install git gmp-devel tcl patch numactl-devel flex -y
    ```

### 安装git-lfs

- Ubuntu使用如下命令安装。

    ```bash
    curl -s https://packagecloud.io/install/repositories/github/git-lfs/script.deb.sh | sudo bash
    sudo apt-get install git-lfs -y
    git lfs install
    ```

- CentOS 7使用如下命令安装。

    ```bash
    curl -s https://packagecloud.io/install/repositories/github/git-lfs/script.rpm.sh | sudo bash
    sudo yum install git-lfs -y
    git lfs install
    ```

- EulerOS和OpenEuler使用以下命令安装。

    根据系统架构选择相应的版本下载。

    ```bash
    # x64
    curl -OL https://github.com/git-lfs/git-lfs/releases/download/v3.1.2/git-lfs-linux-amd64-v3.1.2.tar.gz
    # arm64
    curl -OL https://github.com/git-lfs/git-lfs/releases/download/v3.1.2/git-lfs-linux-arm64-v3.1.2.tar.gz
    ```

    解压并安装。

    ```bash
    mkdir git-lfs
    tar xf git-lfs-linux-*-v3.1.2.tar.gz -C git-lfs
    cd git-lfs
    sudo bash install.sh
    ```

### 安装CMake

- Ubuntu 18.04可以通过以下命令安装[CMake](https://cmake.org/)。

    ```bash
    wget -O - https://apt.kitware.com/keys/kitware-archive-latest.asc 2>/dev/null | sudo apt-key add -
    sudo apt-add-repository "deb https://apt.kitware.com/ubuntu/ $(lsb_release -cs) main"
    sudo apt-get install cmake -y
    ```

- 其他Linux系统可以使用以下命令安装。

    根据系统架构选择不同的下载链接。

    ```bash
    # x86使用
    curl -O https://cmake.org/files/v3.19/cmake-3.19.8-Linux-x86_64.sh
    # aarch64使用
    curl -O https://cmake.org/files/v3.19/cmake-3.19.8-Linux-aarch64.sh
    ```

    执行安装脚本安装CMake，默认安装到`/usr/local`目录下。

    ```bash
    sudo mkdir /usr/local/cmake-3.19.8
    sudo bash cmake-3.19.8-Linux-*.sh --prefix=/usr/local/cmake-3.19.8 --exclude-subdir
    ```

    最后需要将CMake添加到`PATH`环境变量中。如果使用默认安装目录执行以下命令，其他安装目录需要做相应修改。

    ```bash
    echo -e "export PATH=/usr/local/cmake-3.19.8/bin:\$PATH" >> ~/.bashrc
    source ~/.bashrc
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

## 从代码仓下载源码

```bash
git clone https://gitee.com/mindspore/mindspore.git -b r1.7
```

## 编译MindSpore

进入MindSpore根目录，然后执行编译脚本。

```bash
cd mindspore
bash build.sh -e ascend -S on
```

其中：

- `build.sh`中默认的编译线程数为8，如果编译机性能较差可能会出现编译错误，可在执行中增加-j{线程数}来减少线程数量。如`bash build.sh -e ascend -j4`。
- 默认从github下载依赖源码，当-S选项设置为`on`时，从对应的gitee镜像下载。
- 关于`build.sh`更多用法请参看脚本头部的说明。

## 安装MindSpore

```bash
pip install output/mindspore_ascend-*.whl -i https://pypi.tuna.tsinghua.edu.cn/simple
```

在联网状态下，安装whl包时会自动下载MindSpore安装包的依赖项（依赖项详情参见[setup.py](https://gitee.com/mindspore/mindspore/blob/r1.7/setup.py)中的required_package），其余情况需自行安装。运行模型时，需要根据[ModelZoo](https://gitee.com/mindspore/models/tree/r1.7/)中不同模型指定的requirements.txt安装额外依赖，常见依赖可以参考[requirements.txt](https://gitee.com/mindspore/mindspore/blob/r1.7/requirements.txt)。

## 配置环境变量

**如果昇腾AI处理器配套软件包没有安装在默认路径**，安装好MindSpore之后，需要导出Runtime相关环境变量，下述命令中`LOCAL_ASCEND=/usr/local/Ascend`的`/usr/local/Ascend`表示配套软件包的安装路径，需注意将其改为配套软件包的实际安装路径。

```bash
# control log level. 0-DEBUG, 1-INFO, 2-WARNING, 3-ERROR, 4-CRITICAL, default level is WARNING.
export GLOG_v=2

# Conda environmental options
LOCAL_ASCEND=/usr/local/Ascend # the root directory of run package

# lib libraries that the run package depends on
export LD_LIBRARY_PATH=${LOCAL_ASCEND}/ascend-toolkit/latest/fwkacllib/lib64:${LOCAL_ASCEND}/driver/lib64:${LOCAL_ASCEND}/ascend-toolkit/latest/opp/op_impl/built-in/ai_core/tbe/op_tiling:${LD_LIBRARY_PATH}

# Environment variables that must be configured
## TBE operator implementation tool path
export TBE_IMPL_PATH=${LOCAL_ASCEND}/ascend-toolkit/latest/opp/op_impl/built-in/ai_core/tbe
## OPP path
export ASCEND_OPP_PATH=${LOCAL_ASCEND}/ascend-toolkit/latest/opp
## AICPU path
export ASCEND_AICPU_PATH=${ASCEND_OPP_PATH}/..
## TBE operator compilation tool path
export PATH=${LOCAL_ASCEND}/ascend-toolkit/latest/fwkacllib/ccec_compiler/bin/:${PATH}
## Python library that TBE implementation depends on
export PYTHONPATH=${TBE_IMPL_PATH}:${PYTHONPATH}
```

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

context.set_context(device_target="Ascend")
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
    pip install --upgrade mindspore-ascend
    ```

- 本地源码编译升级

    在源码根目录下执行编译脚本`build.sh`成功后，在`output`目录下找到编译生成的whl安装包，然后执行命令进行升级。

    ```bash
    pip install --upgrade mindspore_ascend-*.whl
    ```
