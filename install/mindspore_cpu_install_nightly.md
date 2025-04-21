# pip方式安装MindSpore CPU Nightly版本

<!-- TOC -->

- [pip方式安装MindSpore CPU Nightly版本](#pip方式安装mindspore-cpu-nightly版本)
    - [安装MindSpore与依赖软件](#安装mindspore与依赖软件)
        - [安装Python](#安装python)
        - [安装GCC](#安装gcc)
    - [下载安装MindSpore](#下载安装mindspore)
    - [验证是否成功安装](#验证是否成功安装)
    - [升级MindSpore版本](#升级mindspore版本)

<!-- /TOC -->

[![查看源文件](https://mindspore-website.obs.cn-north-4.myhuaweicloud.com/website-images/r2.6.0rc1/resource/_static/logo_source.svg)](https://gitee.com/mindspore/docs/blob/r2.6.0rc1/install/mindspore_cpu_install_nightly.md)

MindSpore Nightly是包含当前最新功能与bugfix的预览版本，但是可能未经完整的测试与验证，希望体验最新功能或者问题修复的用户可以使用该版本。

本文档介绍如何在CPU环境的Linux系统上，使用pip方式快速安装MindSpore Nightly。

在确认系统环境信息的过程中，如需了解如何安装第三方依赖软件，可以参考社区提供的实践——[在Ubuntu（CPU）上进行源码编译安装MindSpore](https://www.mindspore.cn/news/newschildren?id=365)中的第三方依赖软件安装相关部分，在此感谢社区成员[damon0626](https://gitee.com/damon0626)的分享。

## 安装MindSpore与依赖软件

下表列出了安装MindSpore所需的系统环境和第三方依赖。

| 软件名称              | 版本             | 作用                          |
| --------------------- | ---------------- | ----------------------------- |
| Ubuntu                | 18.04            | 运行MindSpore的操作系统       |
| [Python](#安装python) | 3.9-3.11          | MindSpore的使用依赖Python环境 |
| [GCC](#安装gcc)  | 7.3.0-9.4.0 | 用于编译MindSpore的C++编译器  |

下面给出第三方依赖的安装方法。

### 安装Python

[Python](https://www.python.org/)可通过多种方式进行安装。

- 通过Conda安装Python

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
  conda activate mindspore_py9
  ```

- 通过APT安装Python，命令如下。

  ```bash
  sudo apt-get update
  sudo apt-get install software-properties-common -y
  sudo add-apt-repository ppa:deadsnakes/ppa -y
  sudo apt-get install python3.9 python3.9-dev python3.9-distutils python3-pip -y
  # 将新安装的Python设为默认
  sudo update-alternatives --install /usr/bin/python python /usr/bin/python3.9 100
  # 安装pip
  python -m pip install pip -i https://repo.huaweicloud.com/repository/pypi/simple
  sudo update-alternatives --install /usr/bin/pip pip ~/.local/bin/pip3.9 100
  pip config set global.index-url https://repo.huaweicloud.com/repository/pypi/simple
  ```

  若要安装其他Python版本，只需更改命令中的`3.9`。

可以通过以下命令查看Python版本。

```bash
python --version
```

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

## 下载安装MindSpore

执行以下命令安装MindSpore：

```bash
pip install mindspore-dev -i https://pypi.tuna.tsinghua.edu.cn/simple
```

其中：

- 在联网状态下，安装whl包时会自动下载mindspore安装包的依赖项（依赖项详情参见[setup.py](https://gitee.com/mindspore/mindspore/blob/v2.6.0-rc1/setup.py)中的required_package），其余情况需自行安装依赖。
- pip会自动安装当前最新版本的MindSpore Nightly版本，如果需要安装指定版本，请参照下方升级MindSpore版本相关指导，在下载时手动指定版本。

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

当需要升级MindSpore版本时，可执行如下命令：

```bash
pip install --upgrade mindspore-dev=={version}
```

其中：

- 升级到rc版本时，需要手动指定`{version}`为rc版本号，例如1.6.0rc1.dev20211125；如果希望自动升级到最新版本，`=={version}`字段可以缺省。
