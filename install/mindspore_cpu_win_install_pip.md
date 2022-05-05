# pip方式安装MindSpore CPU版本-Windows

<!-- TOC -->

- [pip方式安装MindSpore CPU版本-Windows](#pip方式安装mindspore-cpu版本-windows)
    - [确认系统环境信息](#确认系统环境信息)
    - [安装MindSpore](#安装mindspore)
    - [验证是否成功安装](#验证是否成功安装)
    - [升级MindSpore版本](#升级mindspore版本)

<!-- /TOC -->

<a href="https://gitee.com/mindspore/docs/blob/r1.7/install/mindspore_cpu_win_install_pip.md" target="_blank"><img src="https://mindspore-website.obs.cn-north-4.myhuaweicloud.com/website-images/r1.7/resource/_static/logo_source.png"></a>

本文档介绍如何在CPU环境的Windows系统上，使用pip方式快速安装MindSpore。

在确认系统环境信息的过程中，如需了解如何安装第三方依赖软件，可以参考社区提供的实践——[在Windows（CPU）上进行源码编译安装MindSpore](https://www.mindspore.cn/news/newschildren?id=364)中的第三方依赖软件安装相关部分，在此感谢社区成员[lvmingfu](https://gitee.com/lvmingfu)的分享。

## 确认系统环境信息

- 确认安装Windows 10是x86架构64位操作系统。
- 确认安装Python 3.7-3.9之间的任意版本。以下列版本为例：
    - Python 3.7.5版本 64位，下载地址：[官网](https://www.python.org/ftp/python/3.7.5/python-3.7.5-amd64.exe)或[华为云](https://mirrors.huaweicloud.com/python/3.7.5/python-3.7.5-amd64.exe)。
    - Python 3.8.0版本 64位，下载地址：[官网](https://www.python.org/ftp/python/3.8.0/python-3.8.0-amd64.exe)或[华为云](https://mirrors.huaweicloud.com/python/3.8.0/python-3.8.0-amd64.exe)。
    - Python 3.9.0版本 64位，下载地址：[官网](https://www.python.org/ftp/python/3.9.0/python-3.9.0-amd64.exe)或[华为云](https://mirrors.huaweicloud.com/python/3.9.0/python-3.9.0-amd64.exe)。
- 安装Python完毕后，将Python和pip添加到系统环境变量。
    - 添加Python：控制面板->系统->高级系统设置->环境变量。双击系统变量中的Path，将`python.exe`的路径添加进去。
    - 添加pip：`python.exe`同一级目录中的`Scripts`文件夹即为Python自带的pip文件，将其路径添加到系统环境变量中即可。

## 安装MindSpore

首先参考[版本列表](https://www.mindspore.cn/versions)选择想要安装的MindSpore版本，并进行SHA-256完整性校验。以1.7.0版本为例，执行以下命令。

```bash
set MS_VERSION=1.7.0
```

然后根据Python版本执行如下命令安装MindSpore。

```bash
# Python3.7
pip install https://ms-release.obs.cn-north-4.myhuaweicloud.com/%MS_VERSION%/MindSpore/cpu/x86_64/mindspore-%MS_VERSION:-=%-cp37-cp37m-win_amd64.whl --trusted-host ms-release.obs.cn-north-4.myhuaweicloud.com -i https://pypi.tuna.tsinghua.edu.cn/simple
# Python3.8
pip install https://ms-release.obs.cn-north-4.myhuaweicloud.com/%MS_VERSION%/MindSpore/cpu/x86_64/mindspore-%MS_VERSION:-=%-cp38-cp38-win_amd64.whl --trusted-host ms-release.obs.cn-north-4.myhuaweicloud.com -i https://pypi.tuna.tsinghua.edu.cn/simple
# Python3.9
pip install https://ms-release.obs.cn-north-4.myhuaweicloud.com/%MS_VERSION%/MindSpore/cpu/x86_64/mindspore-%MS_VERSION:-=%-cp39-cp39-win_amd64.whl --trusted-host ms-release.obs.cn-north-4.myhuaweicloud.com -i https://pypi.tuna.tsinghua.edu.cn/simple
```

在联网状态下，安装whl包时会自动下载mindspore安装包的依赖项（依赖项详情参见[setup.py](https://gitee.com/mindspore/mindspore/blob/r1.7/setup.py)中的required_package），其余情况需自行安装。运行模型时，需要根据[ModelZoo](https://gitee.com/mindspore/models/tree/master/)中不同模型指定的requirements.txt安装额外依赖，常见依赖可以参考[requirements.txt](https://gitee.com/mindspore/mindspore/blob/r1.7/requirements.txt)。

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
pip install --upgrade mindspore=={version}
```

其中：

- 升级到rc版本时，需要手动指定`{version}`为rc版本号，例如1.5.0rc1；如果升级到正式版本，`=={version}`字段可以缺省。
