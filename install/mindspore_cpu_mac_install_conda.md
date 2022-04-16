# Conda方式安装MindSpore CPU版本-macOS

<!-- TOC -->

- [Conda方式安装MindSpore CPU版本-macOS](#conda方式安装mindspore-cpu版本-macos)
    - [确认系统环境信息](#确认系统环境信息)
    - [创建并进入Conda虚拟环境](#创建并进入conda虚拟环境)
    - [安装MindSpore](#安装mindspore)
    - [验证是否成功安装](#验证是否成功安装)
    - [升级MindSpore版本](#升级mindspore版本)

<!-- /TOC -->

<a href="https://gitee.com/mindspore/docs/blob/master/install/mindspore_cpu_install_conda.md" target="_blank"><img src="https://mindspore-website.obs.cn-north-4.myhuaweicloud.com/website-images/master/resource/_static/logo_source.png"></a>

[Conda](https://docs.conda.io/en/latest/)是一个开源跨平台语言无关的包管理与环境管理系统，允许用户方便地安装不同版本的二进制软件包与该计算平台需要的所有库。

本文档介绍如何在macOS系统上，使用Conda方式快速安装MindSpore。

## 确认系统环境信息

- 根据下表中的系统及芯片情况确定合适的Python与Conda版本，macOS版本及芯片信息可点击桌面左上角苹果标志->`关于本机`获悉：

    |芯片|计算架构|macOS版本|支持Python版本|支持Conda版本|
    |-|-|-|-|-|
    |M1|ARM|11.3|Python 3.8-3.9|Mambaforge 或 Miniforge|
    |Intel|x86_64|10.15/11.3|Python 3.7-3.9|Anaconda 或 MiniConda|

- 确认安装与当前系统及芯片型号兼容的Conda版本。

    - 如果您喜欢Conda提供的完整能力，可以选择下载[Anaconda3](https://repo.anaconda.com/archive/)或[Mambaforge](https://github.com/conda-forge/miniforge)。
    - 如果您需要节省磁盘空间，或者喜欢自定义安装Conda软件包，可以选择下载[Miniconda3](https://repo.anaconda.com/miniconda/)或[Miniforge](https://github.com/conda-forge/miniforge)。

## 创建并进入Conda虚拟环境

根据您希望使用的Python版本创建对应的Conda虚拟环境并进入虚拟环境。

- 如果您希望使用Python3.7.5版本(适配64-bit macOS 10.15)：

  ```bash
  conda create -n mindspore_py37 -c conda-forge python=3.7.5
  conda activate mindspore_py37
  ```

- 如果您希望使用Python3.8.0版本(适配64-bit macOS 10.15或11.3)：

  ```bash
  conda create -n mindspore_py38 -c conda-forge python=3.8.0
  conda activate mindspore_py38
  ```

- 如果您希望使用Python3.9.0版本(适配64-bit macOS 10.15或11.3)：

  ```bash
  conda create -n mindspore_py39 -c conda-forge python=3.9.0
  conda activate mindspore_py39
  ```

## 安装MindSpore

确认您处于Conda虚拟环境中，并执行如下命令安装最新版本的MindSpore。如需安装其他版本，可参考[版本列表](https://www.mindspore.cn/versions)在`mindspore-cpu=`后指定版本号。

```bash
conda install mindspore-cpu -c mindspore -c conda-forge
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
