﻿# 源码编译方式安装MindSpore CPU版本-Windows

<!-- TOC -->

- [源码编译方式安装MindSpore CPU版本-Windows](#源码编译方式安装mindspore-cpu版本-windows)
    - [确认系统环境信息](#确认系统环境信息)
    - [从代码仓下载源码](#从代码仓下载源码)
    - [编译MindSpore](#编译mindspore)
    - [安装MindSpore](#安装mindspore)
    - [验证是否成功安装](#验证是否成功安装)
    - [升级MindSpore版本](#升级mindspore版本)

<!-- /TOC -->

<a href="https://gitee.com/mindspore/docs/blob/r1.7/install/mindspore_cpu_win_install_source.md" target="_blank"><img src="https://mindspore-website.obs.cn-north-4.myhuaweicloud.com/website-images/r1.7/resource/_static/logo_source.png"></a>

本文档介绍如何在CPU环境的Windows系统上，使用源码编译方法快速安装MindSpore。

详细步骤可以参考社区提供的实践——[在Windows（CPU）上进行源码编译安装MindSpore](https://www.mindspore.cn/news/newschildren?id=364)，在此感谢社区成员[lvmingfu](https://gitee.com/lvmingfu)的分享。

## 确认系统环境信息

- 确认安装Windows 10是x86架构64位操作系统。
- 确认安装[Visual C++ Redistributable for Visual Studio 2015](https://www.microsoft.com/zh-CN/download/details.aspx?id=48145)。
- 确认安装了[git](https://github.com/git-for-windows/git/releases/download/v2.29.2.windows.2/Git-2.29.2.2-64-bit.exe)工具。
    - 如果git没有安装在`ProgramFiles`，需设置环境变量指定`patch.exe`的位置，例如git安装在`D:\git`时，需设置`set MS_PATCH_PATH=D:\git\usr\bin`。
- 确认安装[MinGW-W64 GCC-7.3.0](https://sourceforge.net/projects/mingw-w64/files/Toolchains%20targetting%20Win64/Personal%20Builds/mingw-builds/7.3.0/threads-posix/seh/x86_64-7.3.0-release-posix-seh-rt_v5-rev0.7z)。
    - 安装路径中不能出现中文和日文，安装完成后将安装路径下的`MinGW\bin`添加到系统环境变量。例如安装在`D:\gcc`，则需要将`D:\gcc\MinGW\bin`添加到系统环境变量Path中。
- 确认安装[CMake 3.18.3版本](https://github.com/Kitware/Cmake/releases/tag/v3.18.3)。
    - 安装路径中不能出现中文和日文，安装完成后将`cmake.exe`的路径添加到系统环境变量Path中。
- 确认安装[ActivePerl 5.28.1.2801版本](https://downloads.activestate.com/ActivePerl/releases/5.28.1.2801/ActivePerl-5.28.1.2801-MSWin32-x64-24563874.exe)。
- 确认安装Python 3.7-3.9之间的任意版本。以下列版本为例：
    - Python 3.7.5版本 64位，下载地址：[官网](https://www.python.org/ftp/python/3.7.5/python-3.7.5-amd64.exe)或[华为云](https://mirrors.huaweicloud.com/python/3.7.5/python-3.7.5-amd64.exe)。
    - Python 3.8.0版本 64位，下载地址：[官网](https://www.python.org/ftp/python/3.8.0/python-3.8.0-amd64.exe)或[华为云](https://mirrors.huaweicloud.com/python/3.8.0/python-3.8.0-amd64.exe)。
    - Python 3.9.0版本 64位，下载地址：[官网](https://www.python.org/ftp/python/3.9.0/python-3.9.0-amd64.exe)或[华为云](https://mirrors.huaweicloud.com/python/3.9.0/python-3.9.0-amd64.exe)。
    - 安装路径中不能出现中文等特殊符号，安装完成后需要将`python.exe`的路径添加到系统环境变量Path中，Python自带的pip文件在`python.exe`同级目录的`Scripts`文件夹中，也需要将pip文件的路径添加到系统环境变量Path中。
- 确认安装[wheel 0.32.0及以上版本](https://pypi.org/project/wheel/)。

## 从代码仓下载源码

```bash
git clone https://gitee.com/mindspore/mindspore.git -b r1.7
```

## 编译MindSpore

在源码根目录下执行如下命令：

```bash
call build.bat
```

## 安装MindSpore

```bash
for %x in (output\mindspore*.whl) do pip install %x -i https://pypi.tuna.tsinghua.edu.cn/simple
```

在联网状态下，安装whl包时会自动下载mindspore安装包的依赖项（依赖项详情参见[setup.py](https://gitee.com/mindspore/mindspore/blob/r1.7/setup.py)中的required_package），其余情况需自行安装。运行模型时，需要根据[ModelZoo](https://gitee.com/mindspore/models/tree/r1.7/)中不同模型指定的requirements.txt安装额外依赖，常见依赖可以参考[requirements.txt](https://gitee.com/mindspore/mindspore/blob/r1.7/requirements.txt)。

## 验证是否成功安装

```bash
cd ..
python -c "import mindspore;mindspore.run_check()"
```

执行`import mindspore`应在MindSpore源码根目录之外，因为Windows上的Python将当前目录视为执行环境，可能产生目录查找问题。
如果输出：

```text
MindSpore version: 版本号
The result of multiplication calculation is correct, MindSpore has been installed successfully!
```

说明MindSpore安装成功了。

## 升级MindSpore版本

当需要升级MindSpore版本时，可执行如下命令：

- 直接在线升级

    ```bash
    pip install --upgrade mindspore
    ```

- 本地源码编译升级

    在源码根目录下执行编译脚本`build.sh`成功后，在`output`目录下找到编译生成的whl安装包，然后执行命令进行升级。

    ```bash
    pip install --upgrade mindspore-{version}-{python_version}-win_amd64.whl
    ```
