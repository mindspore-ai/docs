# pip方式安装MindSpore CPU Nightly版本

<!-- TOC -->

- [pip方式安装MindSpore CPU Nightly版本](#pip方式安装mindspore-cpu-nightly版本)
    - [确认系统环境信息](#确认系统环境信息)
    - [下载安装MindSpore](#下载安装mindspore)
    - [验证是否成功安装](#验证是否成功安装)
    - [升级MindSpore版本](#升级mindspore版本)

<!-- /TOC -->

<a href="https://gitee.com/mindspore/docs/blob/master/install/mindspore_cpu_install_nightly.md" target="_blank"><img src="https://gitee.com/mindspore/docs/raw/master/resource/_static/logo_source.png"></a>

MindSpore Nightly是包含当前最新功能与bugfix的预览版本，但是可能未经完整的测试与验证，希望体验最新功能或者问题修复的用户可以使用该版本。

本文档介绍如何在CPU环境的Linux系统上，使用pip方式快速安装MindSpore Nightly。

在确认系统环境信息的过程中，如需了解如何安装第三方依赖软件，可以参考社区提供的实践——[在Ubuntu（CPU）上进行源码编译安装MindSpore](https://www.mindspore.cn/news/newschildren?id=365)中的第三方依赖软件安装相关部分，在此感谢社区成员[damon0626](https://gitee.com/damon0626)的分享。

## 确认系统环境信息

- 确认安装64位操作系统，[glibc](https://www.gnu.org/software/libc/)>=2.17，其中Ubuntu 18.04是经过验证的。
- 确认安装[GCC 7.3.0版本](https://ftp.gnu.org/gnu/gcc/gcc-7.3.0/gcc-7.3.0.tar.gz)。
- 确认安装[gmp 6.1.2版本](https://gmplib.org/download/gmp/gmp-6.1.2.tar.xz)。
- 确认安装Python 3.7.5或3.9.0版本。如果未安装或者已安装其他版本的Python，可以选择下载并安装：

    - Python 3.7.5版本 64位，下载地址：[官网](https://www.python.org/ftp/python/3.7.5/Python-3.7.5.tgz)或[华为云](https://mirrors.huaweicloud.com/python/3.7.5/Python-3.7.5.tgz)。
    - Python 3.9.0版本 64位，下载地址：[官网](https://www.python.org/ftp/python/3.9.0/Python-3.9.0.tgz)或[华为云](https://mirrors.huaweicloud.com/python/3.9.0/Python-3.9.0.tgz)。

- 如果您的环境为ARM架构，请确认当前使用的Python配套的pip版本>=19.3。

## 下载安装MindSpore

执行如下命令安装MindSpore：

```bash
pip install mindspore-dev -i https://pypi.tuna.tsinghua.edu.cn/simple
```

其中：

- 在联网状态下，安装whl包时会自动下载mindspore安装包的依赖项（依赖项详情参见[setup.py](https://gitee.com/mindspore/mindspore/blob/master/setup.py)中的required_package），其余情况需自行安装。运行模型时，需要根据[ModelZoo](https://gitee.com/mindspore/models/tree/master/)中不同模型指定的requirements.txt安装额外依赖，常见依赖可以参考[requirements.txt](https://gitee.com/mindspore/mindspore/blob/master/requirements.txt)。
- pip会自动安装当前最新版本的Nightly版本MindSpore，如果需要安装指定版本，请参照下方升级MindSpore版本相关指导，在下载时手动指定版本。

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
pip install --upgrade mindspore-dev=={version}
```

其中：

- 升级到rc版本时，需要手动指定`{version}`为rc版本号，例如1.6.0rc1.dev20211125；如果希望自动升级到最新版本，`=={version}`字段可以缺省。
