# Conda方式安装MindSpore CPU版本-Windows

<!-- TOC -->

- [Conda方式安装MindSpore CPU版本-Windows](#conda方式安装mindspore-cpu版本-windows)
    - [确认系统环境信息](#确认系统环境信息)
    - [创建并进入Conda虚拟环境](#创建并进入conda虚拟环境)
    - [安装MindSpore](#安装mindspore)
    - [验证是否成功安装](#验证是否成功安装)
    - [升级MindSpore版本](#升级mindspore版本)

<!-- /TOC -->

[![查看源文件](https://mindspore-website.obs.cn-north-4.myhuaweicloud.com/website-images/br_base/resource/_static/logo_source.svg)](https://gitee.com/mindspore/docs/blob/br_base/install/mindspore_cpu_win_install_conda.md)

[Conda](https://docs.conda.io/en/latest/)是一个开源跨平台语言无关的包管理与环境管理系统，允许用户方便地安装不同版本的二进制软件包，以及该计算平台需要的所有库。

本文档介绍如何在CPU环境的Windows系统上，使用Conda方式快速安装MindSpore。

## 确认系统环境信息

- 确认安装Windows 10是x86架构64位操作系统。
- 确认安装与当前系统兼容的Conda版本。

    - 如果您喜欢Conda提供的完整能力，可以选择下载[Anaconda3](https://repo.anaconda.com/archive/)。
    - 如果您需要节省磁盘空间，或者喜欢自定义安装Conda软件包，可以选择下载[Miniconda3](https://repo.anaconda.com/miniconda/)。

## 创建并进入Conda虚拟环境

在Windows上使用Anaconda，请通过`开始 | Anaconda3 | Anaconda Promt`打开Anaconda命令行。

根据您希望使用的Python版本，创建对应的Conda虚拟环境，并进入虚拟环境。
如果您希望使用Python3.9.11版本：

```bash
conda create -c conda-forge -n mindspore_py39 -c conda-forge python=3.9.11
conda activate mindspore_py39
```

## 安装MindSpore

确认您处于Conda虚拟环境中，并执行以下命令安装最新版本的MindSpore。如需安装其他版本，可参考[版本列表](https://www.mindspore.cn/versions)在`conda install mindspore=`后指定版本号。

```bash
conda install mindspore -c mindspore -c conda-forge
```

在联网状态下，安装Conda安装包时会自动下载MindSpore安装包的依赖项（依赖项详情参见[setup.py](https://gitee.com/mindspore/mindspore/blob/br_base/setup.py)中的required_package），其余情况需自行安装依赖。

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

从MindSpore 2.4.x及以下版本升级到MindSpore 2.5.x及以上版本时，需要先手动卸载旧版本：

```bash
conda remove mindspore-cpu
```

然后安装新版本：

```bash
conda install mindspore -c mindspore -c conda-forge
```

从MindSpore 2.5.x版本升级时，执行以下命令：

```bash
conda update mindspore -c mindspore -c conda-forge
```
