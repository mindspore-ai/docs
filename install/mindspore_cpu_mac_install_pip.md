# pip方式安装MindSpore CPU版本-macOS

<!-- TOC -->

- [pip方式安装MindSpore CPU版本-macOS](#pip方式安装mindspore-cpu版本-macos)
    - [确认系统环境信息](#确认系统环境信息)
    - [创建并进入Conda虚拟环境](#创建并进入conda虚拟环境)
    - [安装MindSpore](#安装mindspore)
    - [验证是否成功安装](#验证是否成功安装)
    - [升级MindSpore版本](#升级mindspore版本)

<!-- /TOC -->

[![查看源文件](https://mindspore-website.obs.cn-north-4.myhuaweicloud.com/website-images/r2.6.0/resource/_static/logo_source.svg)](https://gitee.com/mindspore/docs/blob/r2.6.0/install/mindspore_cpu_mac_install_pip.md)

[Conda](https://docs.conda.io/en/latest/)是一个开源跨平台语言无关的包管理与环境管理系统，允许用户方便地安装不同版本的二进制软件包，以及该计算平台需要的所有库。推荐在MacOS上通过Conda使用MindSpore。

本文档介绍如何在macOS系统上的Conda环境中，使用pip方式快速安装MindSpore。

## 确认系统环境信息

- 根据下表中的系统及芯片情况，确定合适的Python与Conda版本，其中macOS版本及芯片信息可点击桌面左上角苹果标志->`关于本机`获悉：

    |芯片|计算架构|macOS版本|支持Python版本|支持Conda版本|
    |-|-|-|-|-|
    |M1|ARM|11.3|Python 3.9-3.11|Mambaforge 或 Miniforge|
    |Intel|x86_64|10.15/11.3|Python 3.9-3.11|Anaconda 或 MiniConda|

- 确认安装与当前系统及芯片型号兼容的Conda版本。

    - 如果您喜欢Conda提供的完整能力，可以选择下载[Anaconda3](https://repo.anaconda.com/archive/)或[Mambaforge](https://github.com/conda-forge/miniforge)。
    - 如果您需要节省磁盘空间，或者喜欢自定义安装Conda软件包，可以选择下载[Miniconda3](https://repo.anaconda.com/miniconda/)或[Miniforge](https://github.com/conda-forge/miniforge)。

## 创建并进入Conda虚拟环境

根据您希望使用的Python版本，创建对应的Conda虚拟环境，并进入虚拟环境。

- 如果您希望使用Python3.9.11版本（适配64-bit macOS 10.15或11.3）：

  ```bash
  conda create -c conda-forge -n mindspore_py39 -c conda-forge python=3.9.11
  conda activate mindspore_py39
  ```

## 安装MindSpore

首先参考[版本列表](https://www.mindspore.cn/versions)选择想要安装的MindSpore版本，并进行SHA-256完整性校验。以2.6.0版本为例，执行以下命令。

```bash
export MS_VERSION=2.6.0
```

然后根据系统架构及Python版本，执行以下命令安装MindSpore。

```bash
# install prerequisites
conda install scipy -c conda-forge

# x86_64 + Python3.9
pip install https://ms-release.obs.cn-north-4.myhuaweicloud.com/${MS_VERSION}/MindSpore/cpu/x86_64/mindspore-${MS_VERSION/-/}-cp39-cp39-macosx_10_15_x86_64.whl --trusted-host ms-release.obs.cn-north-4.myhuaweicloud.com -i https://pypi.tuna.tsinghua.edu.cn/simple
# x86_64 + Python3.10
pip install https://ms-release.obs.cn-north-4.myhuaweicloud.com/${MS_VERSION}/MindSpore/cpu/x86_64/mindspore-${MS_VERSION/-/}-cp310-cp310-macosx_10_15_x86_64.whl --trusted-host ms-release.obs.cn-north-4.myhuaweicloud.com -i https://pypi.tuna.tsinghua.edu.cn/simple
# x86_64 + Python3.11
pip install https://ms-release.obs.cn-north-4.myhuaweicloud.com/${MS_VERSION}/MindSpore/cpu/x86_64/mindspore-${MS_VERSION/-/}-cp311-cp311-macosx_10_15_x86_64.whl --trusted-host ms-release.obs.cn-north-4.myhuaweicloud.com -i https://pypi.tuna.tsinghua.edu.cn/simple
# aarch64 + Python3.9
pip install https://ms-release.obs.cn-north-4.myhuaweicloud.com/${MS_VERSION}/MindSpore/cpu/aarch64/mindspore-${MS_VERSION/-/}-cp39-cp39-macosx_11_0_arm64.whl --trusted-host ms-release.obs.cn-north-4.myhuaweicloud.com -i https://pypi.tuna.tsinghua.edu.cn/simple
# aarch64 + Python3.10
pip install https://ms-release.obs.cn-north-4.myhuaweicloud.com/${MS_VERSION}/MindSpore/cpu/aarch64/mindspore-${MS_VERSION/-/}-cp310-cp310-macosx_11_0_arm64.whl --trusted-host ms-release.obs.cn-north-4.myhuaweicloud.com -i https://pypi.tuna.tsinghua.edu.cn/simple
# aarch64 + Python3.11
pip install https://ms-release.obs.cn-north-4.myhuaweicloud.com/${MS_VERSION}/MindSpore/cpu/aarch64/mindspore-${MS_VERSION/-/}-cp311-cp311-macosx_11_0_arm64.whl --trusted-host ms-release.obs.cn-north-4.myhuaweicloud.com -i https://pypi.tuna.tsinghua.edu.cn/simple
```

在联网状态下，安装whl包时会自动下载mindspore安装包的依赖项（依赖项详情参见[setup.py](https://gitee.com/mindspore/mindspore/blob/v2.6.0/setup.py)中的required_package），其余情况需自行安装依赖。

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

当需要升级MindSpore版本时，可执行以下命令：

```bash
pip install --upgrade mindspore=={version}
```

其中：

- 升级到rc版本时，需要手动指定`{version}`为rc版本号，例如1.5.0rc1；如果升级到正式版本，`=={version}`字段可以缺省。
