# Conda方式安装MindSpore CPU版本

<!-- TOC -->

- [Conda方式安装MindSpore CPU版本](#conda方式安装mindspore-cpu版本)
    - [自动安装](#自动安装)
    - [手动安装](#手动安装)
        - [安装Conda](#安装conda)
        - [安装GCC和gmp](#安装gcc和gmp)
        - [创建并进入Conda虚拟环境](#创建并进入conda虚拟环境)
        - [安装MindSpore](#安装mindspore)
    - [验证是否成功安装](#验证是否成功安装)
    - [升级MindSpore版本](#升级mindspore版本)

<!-- /TOC -->

<a href="https://gitee.com/mindspore/docs/blob/master/install/mindspore_cpu_install_conda.md" target="_blank"><img src="https://mindspore-website.obs.cn-north-4.myhuaweicloud.com/website-images/master/resource/_static/logo_source.png"></a>

[Conda](https://docs.conda.io/en/latest/)是一个开源跨平台语言无关的包管理与环境管理系统，允许用户方便地安装不同版本的二进制软件包与该计算平台需要的所有库。

本文档介绍如何在CPU环境的Linux系统上，使用Conda方式快速安装MindSpore。下面以Ubuntu 18.04为例说明MindSpore安装步骤。

- 如果您想在一个全新的Ubuntu 18.04上通过Conda安装MindSpore，可以使用[自动安装脚本](https://gitee.com/mindspore/mindspore/raw/master/scripts/install/ubuntu-cpu-conda.sh)进行一键式安装，参见[自动安装](#自动安装)小节。自动安装脚本会安装MindSpore及其所需的依赖。

- 如果您的系统已经安装了部分依赖，如Conda，GCC等，则推荐参照[手动安装](#手动安装)小节的安装步骤手动安装。

## 自动安装

自动安装脚本需要更改软件源配置以及通过APT安装依赖，所以在执行中需要申请root权限。使用以下命令获取自动安装脚本并执行。自动安装脚本仅支持安装MindSpore>=1.6.0。

```bash
wget https://gitee.com/mindspore/mindspore/raw/master/scripts/install/ubuntu-cpu-conda.sh
# 默认安装Python 3.7和最新版本的MindSpore
bash ./ubuntu-cpu-conda.sh
# 如需指定Python和MindSpore版本，以Python 3.9和MindSpore 1.6.0为例，使用以下方式
# PYTHON_VERSION=3.9 MINDSPORE_VERSION=1.6.0 bash ./ubuntu-cpu-conda.sh
```

该脚本会执行以下操作：

- 更改软件源配置为华为云源。
- 安装MindSpore所需的依赖，如GCC，gmp。
- 安装Conda并为MindSpore创建虚拟环境。
- 通过Conda安装MindSpore CPU版本。

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
|[Conda](#安装conda)|Anaconda3或Miniconda3|Python环境管理工具|
|[GCC](#安装gcc和gmp)|7.3.0到9.4.0之间|用于编译MindSpore的C++编译器|
|[gmp](#安装gcc和gmp)|6.1.2|MindSpore使用的多精度算术库|

下面给出第三方依赖的安装方法。

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

或者安装GCC 9。

```bash
sudo apt-get install software-properties-common -y
sudo add-apt-repository ppa:ubuntu-toolchain-r/test
sudo apt-get update
sudo apt-get install gcc-9 -y
```

### 创建并进入Conda虚拟环境

根据您希望使用的Python版本创建对应的Conda虚拟环境并进入虚拟环境。

如果您希望使用Python3.7.5版本：

```bash
conda create -c conda-forge -n mindspore_py37 python=3.7.5 -y
conda activate mindspore_py37
```

如果希望使用其他版本Python，只需更改以上命令中的Python版本。当前支持Python 3.7，Python 3.8和Python 3.9。

### 安装MindSpore

确认您处于Conda虚拟环境中，并执行如下命令安装最新版本的MindSpore。如需安装其他版本，可参考[版本列表](https://www.mindspore.cn/versions)在`mindspore-cpu=`后指定版本号。

```bash
conda install mindspore-cpu -c mindspore -c conda-forge -y
```

在联网状态下，安装Conda安装包时会自动下载MindSpore安装包的依赖项（依赖项详情参见[setup.py](https://gitee.com/mindspore/mindspore/blob/master/setup.py)中的required_package），其余情况需自行安装。运行模型时，需要根据[ModelZoo](https://gitee.com/mindspore/models/tree/master/)中不同模型指定的requirements.txt安装额外依赖，常见依赖可以参考[requirements.txt](https://gitee.com/mindspore/mindspore/blob/master/requirements.txt)。

## 验证是否成功安装

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

```bash
conda update mindspore-cpu -c mindspore -c conda-forge
```
