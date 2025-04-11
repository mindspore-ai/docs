# 源码编译方式安装MindSpore CPU版本-macOS

<!-- TOC -->

- [源码编译方式安装MindSpore CPU版本-macOS](#源码编译方式安装mindspore-cpu版本-macos)
    - [确认系统环境信息](#确认系统环境信息)
    - [创建并进入Conda虚拟环境](#创建并进入conda虚拟环境)
    - [从代码仓下载源码](#从代码仓下载源码)
    - [编译MindSpore](#编译mindspore)
    - [安装MindSpore](#安装mindspore)
    - [验证安装是否成功](#验证安装是否成功)
    - [升级MindSpore版本](#升级mindspore版本)

<!-- /TOC -->

[![查看源文件](https://mindspore-website.obs.cn-north-4.myhuaweicloud.com/website-images/r2.6.0/resource/_static/logo_source.svg)](https://gitee.com/mindspore/docs/blob/r2.6.0/install/mindspore_cpu_mac_install_source.md)

[Conda](https://docs.conda.io/en/latest/)是一个开源跨平台语言无关的包管理与环境管理系统，允许用户方便地安装不同版本的二进制软件包，以及该计算平台需要的所有库。推荐在MacOS上通过Conda使用MindSpore。

本文档介绍如何在macOS系统上的Conda环境中，使用源码编译方式快速安装MindSpore。

## 确认系统环境信息

- 根据下表中的系统及芯片情况，确定合适的Python与Conda版本，其中macOS版本及芯片信息可点击桌面左上角苹果标志->`关于本机`获悉：

    |芯片|计算架构|macOS版本|支持Python版本|支持Conda版本|
    |-|-|-|-|-|
    |M1|ARM|11.3|Python 3.9-3.11|Mambaforge 或 Miniforge|
    |Intel|x86_64|10.15/11.3|Python 3.9-3.11|Anaconda 或 MiniConda|

- 确认安装与当前系统及芯片型号兼容的Conda版本。

    - 如果您喜欢Conda提供的完整能力，可以选择下载[Anaconda3](https://repo.anaconda.com/archive/)或[Mambaforge](https://github.com/conda-forge/miniforge)。
    - 如果您需要节省磁盘空间，或者喜欢自定义安装Conda软件包，可以选择下载[Miniconda3](https://repo.anaconda.com/miniconda/)或[Miniforge](https://github.com/conda-forge/miniforge)。

- 确认安装[Xcode](https://xcodereleases.com/) (>=12.4 并且 <= 13.0) ，12.4(X86)及13.0(M1) 已测试。

- 确认安装`Command Line Tools for Xcode`。如果没有安装，可以使用 `sudo xcode-select --install` 命令安装。

- 确认安装[CMake 3.22.2及以上版本](https://cmake.org/download/)。如果没有安装，可以使用 `brew install cmake` 命令安装。

- 确认安装[patch 2.5](https://ftp.gnu.org/gnu/patch/)。如果没有安装，可以使用 `brew install patch` 命令安装。

- 确认安装[wheel 0.32.0及以上版本](https://pypi.org/project/wheel/)。如果没有安装，可以使用 `conda install wheel` 命令安装。

- 确认安装[PyYAML](https://pypi.org/project/pyyaml/) (>=6.0 并且 <= 6.0.2)。如果没有安装，可以使用 `conda install pyyaml` 命令安装。

- 确认安装[autoconf](https://ftp.gnu.org/gnu/autoconf/)。如果没有安装，可以使用 `brew install autoconf` 命令安装。

- 确认安装[Numpy](https://pypi.org/project/numpy/) (>=1.19.3 并且 <= 1.26.4)。如果没有安装，可以使用 `conda install numpy` 命令安装。

## 创建并进入Conda虚拟环境

根据您希望使用的Python版本，创建对应的Conda虚拟环境，并进入虚拟环境。

- 如果您希望使用Python3.9.11版本（适配64-bit macOS 10.15或11.3）：

  ```bash
  conda create -c conda-forge -n mindspore_py39 -c conda-forge python=3.9.11
  conda activate mindspore_py39
  ```

## 从代码仓下载源码

```bash
git clone -b v2.6.0 https://gitee.com/mindspore/mindspore.git
```

## 编译MindSpore

在源码根目录下执行以下命令。

```bash
export CC=/usr/bin/clang
export CXX=/usr/bin/clang++
bash build.sh -e cpu -S on -j4  # -j 为编译时线程配置，如果CPU性能较好，使用多线程方式编译，参数通常为CPU核数的两倍
```

## 安装MindSpore

```bash
# install prerequisites
conda install scipy -c conda-forge

pip install output/mindspore-*.whl -i https://pypi.tuna.tsinghua.edu.cn/simple
```

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
