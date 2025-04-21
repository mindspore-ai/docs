# 源码编译方式安装MindSpore CPU版本

<!-- TOC -->

- [源码编译方式安装MindSpore CPU版本](#源码编译方式安装mindspore-cpu版本)
    - [安装依赖软件](#安装依赖软件)
        - [安装Python](#安装python)
        - [安装wheel setuptools PyYAML和Numpy](#安装wheel-setuptools-pyyaml和numpy)
        - [安装GCC git tclsh patch和NUMA](#安装gcc-git-tclsh-patch和numa)
        - [安装CMake](#安装cmake)
        - [安装LLVM-可选](#安装llvm-可选)
    - [从代码仓下载源码](#从代码仓下载源码)
    - [编译MindSpore](#编译mindspore)
    - [安装MindSpore](#安装mindspore)
    - [验证安装是否成功](#验证安装是否成功)
    - [升级MindSpore版本](#升级mindspore版本)

<!-- /TOC -->

[![查看源文件](https://mindspore-website.obs.cn-north-4.myhuaweicloud.com/website-images/br_base/resource/_static/logo_source.svg)](https://gitee.com/mindspore/docs/blob/br_base/install/mindspore_cpu_install_source.md)

本文档介绍如何在CPU环境的Linux系统上，使用源码编译方式快速安装MindSpore。下面以Ubuntu 18.04为例说明MindSpore编译安装步骤。

## 安装依赖软件

下表列出了编译安装MindSpore所需的系统环境和第三方依赖。

|软件名称|版本|作用|
|-|-|-|
|Ubuntu|18.04|编译和运行MindSpore的操作系统|
|[Python](#安装python)|3.9-3.11|MindSpore的使用依赖Python环境|
|[wheel](#安装wheel-setuptools-pyyaml和numpy)|0.32.0及以上|MindSpore使用的Python打包工具|
|[setuptools](#安装wheel-setuptools-pyyaml和numpy)|44.0及以上|MindSpore使用的Python包管理工具|
|[PyYAML](#安装wheel-setuptools-pyyaml和numpy)|6.0-6.0.2|MindSpore里的算子编译功能依赖PyYAML模块|
|[Numpy](#安装wheel-setuptools-pyyaml和numpy)|1.19.3-1.26.4|MindSpore里的Numpy相关功能依赖Numpy模块|
|[GCC](#安装gcc-git-tclsh-patch和numa)|7.3.0-9.4.0|用于编译MindSpore的C++编译器|
|[git](#安装gcc-git-tclsh-patch和numa)|-|MindSpore使用的源代码管理工具|
|[CMake](#安装cmake)|3.22.2及以上|编译构建MindSpore的工具|
|[tclsh](#安装gcc-git-tclsh-patch和numa)|-|MindSpore sqlite编译依赖|
|[patch](#安装gcc-git-tclsh-patch和numa)|2.5及以上|MindSpore使用的源代码补丁工具|
|[NUMA](#安装gcc-git-tclsh-patch和numa)|2.0.11及以上|MindSpore使用的非一致性内存访问库|
|[LLVM](#安装llvm-可选)|12.0.1|MindSpore使用的编译器框架（可选，图算融合以及稀疏计算需要）|

下面给出第三方依赖的安装方法。

### 安装Python

[Python](https://www.python.org/)可通过Conda进行安装。

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

### 安装GCC git tclsh patch和NUMA

可以通过以下命令安装GCC、git、tclsh、patch和NUMA。

```bash
sudo apt-get install gcc-7 git tcl patch libnuma-dev -y
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
git clone https://gitee.com/mindspore/mindspore.git
```

## 编译MindSpore

进入mindspore根目录，然后执行编译脚本。

```bash
cd mindspore
bash build.sh -e cpu -j4 -S on
```

其中：

- 如果编译机性能较好，可在执行中增加`-j{线程数}`来增加线程数量。如`bash build.sh -e cpu -j12`。
- 默认从github下载依赖源码，当-S选项设置为`on`时，从对应的Gitee镜像下载。
- 关于`build.sh`更多用法，请参看脚本头部的说明。

## 安装MindSpore

```bash
pip install output/mindspore-*.whl -i https://pypi.tuna.tsinghua.edu.cn/simple
```

在联网状态下，安装whl包时会自动下载mindspore安装包的依赖项（依赖项详情参见[setup.py](https://gitee.com/mindspore/mindspore/blob/br_base/setup.py)中的required_package），其余情况需自行安装依赖。

## 验证安装是否成功

执行以下命令：

```bash
python -c "import mindspore;mindspore.set_device(device_target='CPU');mindspore.run_check()"
```

如果输出：

```text
MindSpore version: 版本号
The result of multiplication calculation is correct, MindSpore has been installed on platform [CPU] successfully!
```

说明MindSpore安装成功了。

## 升级MindSpore版本

在源码根目录下执行编译脚本`build.sh`成功后，在`output`目录下找到编译生成的whl安装包，然后执行以下命令进行升级。

 ```bash
pip install --upgrade mindspore-*.whl
```
