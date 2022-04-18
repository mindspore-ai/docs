# 源码编译方式安装MindSpore CPU版本

<!-- TOC -->

- [源码编译方式安装MindSpore CPU版本](#源码编译方式安装mindspore-cpu版本)
    - [环境准备-自动 推荐](#环境准备-自动-推荐)
    - [环境准备-手动](#环境准备-手动)
        - [安装Python](#安装python)
        - [安装wheel和setuptools](#安装wheel和setuptools)
        - [安装GCC git gmp tclsh patch和NUMA](#安装gcc-git-gmp-tclsh-patch和numa)
        - [安装CMake](#安装cmake)
        - [安装LLVM-可选](#安装llvm-可选)
    - [从代码仓下载源码](#从代码仓下载源码)
    - [编译MindSpore](#编译mindspore)
    - [安装MindSpore](#安装mindspore)
    - [验证安装是否成功](#验证安装是否成功)
    - [升级MindSpore版本](#升级mindspore版本)

<!-- /TOC -->

<a href="https://gitee.com/mindspore/docs/blob/r1.7/install/mindspore_cpu_install_source.md" target="_blank"><img src="https://mindspore-website.obs.cn-north-4.myhuaweicloud.com/website-images/r1.7/resource/_static/logo_source.png"></a>

本文档介绍如何在CPU环境的Linux系统上，使用源码编译方式快速安装MindSpore。下面以Ubuntu 18.04为例说明MindSpore编译安装步骤。

- 如果您想在一个全新的Ubuntu 18.04上配置一个可以编译MindSpore的环境，可以使用[自动安装脚本](https://gitee.com/mindspore/mindspore/raw/r1.7/scripts/install/ubuntu-cpu-source.sh)进行一键式配置，参见[环境准备-自动，推荐](#环境准备-自动推荐)小节。自动安装脚本会安装编译MindSpore所需的依赖。

- 如果您的系统已经安装了部分依赖，如Python，GCC等，则推荐参照[环境准备-手动](#环境准备-手动)小节的安装步骤手动安装。

## 环境准备-自动 推荐

自动安装脚本需要更改软件源配置以及通过APT安装依赖，所以需要申请root权限。使用以下命令获取自动安装脚本并执行。通过自动安装脚本配置的环境，仅支持编译MindSpore>=1.6.0。

```bash
wget https://gitee.com/mindspore/mindspore/raw/r1.7/scripts/install/ubuntu-cpu-source.sh
# 默认安装Python 3.7
bash ./ubuntu-cpu-source.sh
# 如需指定Python版本，以Python 3.9为例，使用以下方式
# PYTHON_VERSION=3.9 bash ./ubuntu-cpu-source.sh
```

该脚本会执行以下操作：

- 更改软件源配置为华为云源。
- 安装MindSpore所需的编译依赖，如GCC，CMake等。
- 通过APT安装Python3和pip3，并设为默认。

自动安装脚本执行完成后，需要重新打开终端窗口以使环境变量生效，然后跳转到[从代码仓下载源码](#从代码仓下载源码)小节下载编译MindSpore。

更多的用法请参看脚本头部的说明。

## 环境准备-手动

下表列出了编译安装MindSpore所需的系统环境和第三方依赖。

|软件名称|版本|作用|
|-|-|-|
|Ubuntu|18.04|编译和运行MindSpore的操作系统|
|[Python](#安装python)|3.7-3.9|MindSpore的使用依赖Python环境|
|[wheel](#安装wheel和setuptools)|0.32.0及以上|MindSpore使用的Python打包工具|
|[setuptools](#安装wheel和setuptools)|44.0及以上|MindSpore使用的Python包管理工具|
|[GCC](#安装gcc-git-gmp-tclsh-patch和numa)|7.3.0到9.4.0之间|用于编译MindSpore的C++编译器|
|[git](#安装gcc-git-gmp-tclsh-patch和numa)|-|MindSpore使用的源代码管理工具|
|[CMake](#安装cmake)|3.18.3及以上|编译构建MindSpore的工具|
|[gmp](#安装gcc-git-gmp-tclsh-patch和numa)|6.1.2|MindSpore使用的多精度算术库|
|[tclsh](#安装gcc-git-gmp-tclsh-patch和numa)|-|MindSpore sqlite编译依赖|
|[patch](#安装gcc-git-gmp-tclsh-patch和numa)|2.5及以上|MindSpore使用的源代码补丁工具|
|[NUMA](#安装gcc-git-gmp-tclsh-patch和numa)|2.0.11及以上|MindSpore使用的非一致性内存访问库|
|[LLVM](#安装llvm-可选)|12.0.1|MindSpore使用的编译器框架（可选，图算融合以及稀疏计算需要）|

下面给出第三方依赖的安装方法。

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

### 安装GCC git gmp tclsh patch和NUMA

可以通过以下命令安装GCC，git，gmp，tclsh，patch和NUMA。

```bash
sudo apt-get install gcc-7 git libgmp-dev tcl patch libnuma-dev -y
```

如果要安装更高版本的GCC，使用以下命令安装GCC 8。

```bash
sudo apt-get install gcc-8 -y
```

或者安装GCC 9。

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

## 从代码仓下载源码

```bash
git clone https://gitee.com/mindspore/mindspore.git -b r1.7
```

## 编译MindSpore

进入mindspore根目录，然后执行编译脚本。

```bash
cd mindspore
bash build.sh -e cpu -j4 -S on
```

其中：

- 如果编译机性能较好，可在执行中增加-j{线程数}来增加线程数量。如`bash build.sh -e cpu -j12`。
- 默认从github下载依赖源码，当-S选项设置为`on`时，从对应的gitee镜像下载。
- 关于`build.sh`更多用法请参看脚本头部的说明。

## 安装MindSpore

```bash
pip install output/mindspore-*.whl -i https://pypi.tuna.tsinghua.edu.cn/simple
```

在联网状态下，安装whl包时会自动下载mindspore安装包的依赖项（依赖项详情参见[setup.py](https://gitee.com/mindspore/mindspore/blob/r1.7/setup.py)中的required_package），其余情况需自行安装。运行模型时，需要根据[ModelZoo](https://gitee.com/mindspore/models/tree/r1.7/)中不同模型指定的requirements.txt安装额外依赖，常见依赖可以参考[requirements.txt](https://gitee.com/mindspore/mindspore/blob/r1.7/requirements.txt)。

## 验证安装是否成功

```bash
python -c "import mindspore;mindspore.run_check()"
```

如果输出：

```text
MindSpore version: 版本号
The result of multiplication calculation is correct, MindSpore has been installed successfully!
```

说明MindSpore安装成功了。

## 升级MindSpore版本

当需要升级MindSpore版本时，可执行如下命令：

- 直接在线升级

    ```bash
    pip install --upgrade mindspore
    ```

- 本地源码编译升级

    在源码根目录下执行编译脚本`build.sh`成功后，在`output`目录下找到编译生成的whl安装包，然后执行命令进行升级。

    ```bash
    pip install --upgrade mindspore-*.whl
    ```
