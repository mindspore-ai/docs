# Conda方式安装MindSpore GPU版本-Windows

<!-- TOC -->

- [Conda方式安装MindSpore GPU版本-Windows](#conda方式安装mindspore-gpu版本-windows)
    - [确认系统环境信息](#确认系统环境信息)
    - [创建并进入Conda虚拟环境](#创建并进入conda虚拟环境)
    - [安装MindSpore](#安装mindspore)
    - [验证是否成功安装](#验证是否成功安装)
    - [升级MindSpore版本](#升级mindspore版本)

<!-- /TOC -->

[![查看源文件](https://mindspore-website.obs.cn-north-4.myhuaweicloud.com/website-images/r2.0/resource/_static/logo_source.png)](https://gitee.com/mindspore/docs/blob/r2.0/install/mindspore_gpu_win_install_conda.md)

[Conda](https://docs.conda.io/en/latest/)是一个开源跨平台语言无关的包管理与环境管理系统，允许用户方便地安装不同版本的二进制软件包与该计算平台需要的所有库。

本文档介绍如何在GPU环境的Windows系统上，使用Conda方式快速安装MindSpore。

## 确认系统环境信息

- 确认安装Windows 10是x86架构64位操作系统。
- 确认安装与当前系统兼容的Conda版本。
    - 如果您喜欢Conda提供的完整能力，可以选择下载[Anaconda3](https://repo.anaconda.com/archive/)。
    - 如果您需要节省磁盘空间，或者喜欢自定义安装Conda软件包，可以选择下载[Miniconda3](https://repo.anaconda.com/miniconda/)。
- 确认安装[CUDA 11.1/11.6](https://developer.nvidia.com/cuda-11.1.1-download-archive)以及[cuDNN](https://developer.nvidia.com/cudnn)，cuDNN的安装可以参考Nvidia[官方文档](https://docs.nvidia.com/deeplearning/cudnn/install-guide/index.html)，同时需要将cuDNN的安装目录加入到`CUDNN_HOME`环境变量。

## 创建并进入Conda虚拟环境

根据您希望使用的Python版本创建对应的Conda虚拟环境并进入虚拟环境。
如果您希望使用Python3.7.5版本：

```bash
conda create -c conda-forge -n mindspore_py37 -c conda-forge python=3.7.5
activate mindspore_py37
```

如果您希望使用Python3.8.0版本：

```bash
conda create -c conda-forge -n mindspore_py38 -c conda-forge python=3.8.0
activate mindspore_py38
```

如果您希望使用Python3.9.0版本：

```bash
conda create -c conda-forge -n mindspore_py39 -c conda-forge python=3.9.0
activate mindspore_py39
```

## 安装MindSpore

确认您处于Conda虚拟环境中，并执行如下命令安装最新版本的MindSpore。如需安装其他版本，可参考[版本列表](https://www.mindspore.cn/versions)在`mindspore=`后指定版本号。

```bash
conda install mindspore-gpu -c mindspore -c conda-forge
```

在联网状态下，安装Conda安装包时会自动下载MindSpore安装包的依赖项（依赖项详情参见[setup.py](https://gitee.com/mindspore/mindspore/blob/r2.0/setup.py)中的required_package），其余情况需自行安装。运行模型时，需要根据[ModelZoo](https://gitee.com/mindspore/models/tree/r2.0/)中不同模型指定的requirements.txt安装额外依赖，常见依赖可以参考[requirements.txt](https://gitee.com/mindspore/mindspore/blob/r2.0/requirements.txt)。

## 验证是否成功安装

```bash
python -c "import mindspore;mindspore.run_check()"
```

如果输出：

```text
MindSpore version: 版本号
The result of multiplication calculation is correct, MindSpore has been installed on platform [GPU] successfully!
```

说明MindSpore安装成功了。

## 升级MindSpore版本

当需要升级MindSpore版本时，可执行如下命令：

```bash
conda update mindspore-gpu -c mindspore -c conda-forge
```
