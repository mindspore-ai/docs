# Conda方式安装MindSpore CPU版本

<!-- TOC -->

- [Conda方式安装MindSpore CPU版本](#conda方式安装mindspore-cpu版本)
    - [安装MindSpore与依赖软件](#安装mindspore与依赖软件)
        - [安装Conda](#安装conda)
        - [安装GCC](#安装gcc)
        - [创建并进入Conda虚拟环境](#创建并进入conda虚拟环境)
        - [安装MindSpore](#安装mindspore)
    - [验证是否成功安装](#验证是否成功安装)
    - [升级MindSpore版本](#升级mindspore版本)

<!-- /TOC -->

[![查看源文件](https://mindspore-website.obs.cn-north-4.myhuaweicloud.com/website-images/master/resource/_static/logo_source.svg)](https://gitee.com/mindspore/docs/blob/master/install/mindspore_cpu_install_conda.md)

[Conda](https://docs.conda.io/en/latest/)是一个开源跨平台语言无关的包管理与环境管理系统，允许用户方便地安装不同版本的二进制软件包，以及该计算平台需要的所有库。

本文档介绍如何在CPU环境的Linux系统上，使用Conda方式快速安装MindSpore。下面以Ubuntu 18.04为例说明MindSpore安装步骤。

## 安装MindSpore与依赖软件

下表列出了安装MindSpore所需的系统环境和第三方依赖。

|软件名称|版本|作用|
|-|-|-|
|Ubuntu|18.04|运行MindSpore的操作系统|
|[Conda](#安装conda)|Anaconda3或Miniconda3|Python环境管理工具|
|[GCC](#安装gcc)|7.3.0-9.4.0|用于编译MindSpore的C++编译器|

下面给出第三方依赖的安装方法。

### 安装Conda

执行以下命令安装Miniconda。

```bash
cd /tmp
curl -O https://mirrors.tuna.tsinghua.edu.cn/anaconda/miniconda/Miniconda3-py37_4.10.3-Linux-$(arch).sh
bash Miniconda3-py37_4.10.3-Linux-$(arch).sh -b
cd -
. ~/miniconda3/etc/profile.d/conda.sh
conda init bash
```

安装完成后，可以为Conda设置清华源加速下载，参考[此处](https://mirrors.tuna.tsinghua.edu.cn/help/anaconda/)。

### 安装GCC

可以通过以下命令安装GCC。

```bash
sudo apt-get install gcc-7 -y
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

根据您希望使用的Python版本，创建对应的Conda虚拟环境，并进入虚拟环境。

如果您希望使用Python3.9.11版本，执行以下命令：

```bash
conda create -c conda-forge -n mindspore_py39 python=3.9.11 -y
conda activate mindspore_py39
```

如果希望使用其他版本Python，只需更改以上命令中的Python版本。当前支持Python 3.9、Python 3.10和Python 3.11。

### 安装MindSpore

确认您处于Conda虚拟环境中，并执行以下命令安装最新版本的MindSpore。如需安装其他版本，可参考[版本列表](https://www.mindspore.cn/versions)在`conda install mindspore=`后指定版本号。

```bash
conda install mindspore -c mindspore -c conda-forge -y
```

在联网状态下，安装Conda安装包时会自动下载MindSpore安装包的依赖项（依赖项详情参见[setup.py](https://gitee.com/mindspore/mindspore/blob/master/setup.py)中的required_package），其余情况需自行安装依赖。

## 验证是否成功安装

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

从MindSpore 1.x升级到MindSpore 2.x版本时，需要先手动卸载旧版本：

```bash
conda remove mindspore-cpu
```

然后安装新版本：

```bash
conda install mindspore -c mindspore -c conda-forge
```

从MindSpore 2.x版本升级时，执行以下命令：

```bash
conda update mindspore -c mindspore -c conda-forge
```
