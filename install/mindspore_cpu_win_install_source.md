# 源码编译方式安装MindSpore

<!-- TOC -->

- [源码编译方式安装MindSpore](#源码编译方式安装mindspore)
    - [确认系统环境信息](#确认系统环境信息)
    - [从代码仓下载源码](#从代码仓下载源码)
    - [编译MindSpore](#编译mindspore)
    - [安装MindSpore](#安装mindspore)
    - [验证是否安装成功](#验证是否安装成功)

<!-- /TOC -->

<a href="https://gitee.com/mindspore/docs/blob/master/install/mindspore_cpu_win_install_source.md" target="_blank"><img src="https://gitee.com/mindspore/docs/raw/master/resource/_static/logo_source.png"></a>

本文档介绍如何在CPU环境的Windows系统上，使用源码编译方法快速安装MindSpore。

## 确认系统环境信息

- 需要确认您的Windows 7/8/10是64位操作系统。
- 确认安装[Visual C++ Redistributable for Visual Studio 2015](https://www.microsoft.com/zh-CN/download/details.aspx?id=48145)。
- 确认安装了`git`，如果git没有安装在`ProgramFiles`，在执行上述命令前，需设置环境变量指定`patch.exe`的位置，例如git安装在`D:\git`时，需设置`set MS_PATCH_PATH=D:\git\usr\bin`。
- 确认安装[MinGW-W64 GCC-7.3.0](https://sourceforge.net/projects/mingw-w64/files/Toolchains%20targetting%20Win64/Personal%20Builds/mingw-builds/7.3.0/threads-posix/seh/x86_64-7.3.0-release-posix-seh-rt_v5-rev0.7z)，安装路径中不能出现中文和日文，安装完成后将安装路径下的`MinGW/bin`添加到系统环境变量。例如安装在`d:/gcc`，则需要将`d:/gcc/MinGW/bin`添加到系统环境变量Path中。
- 确认安装[CMake](https://cmake.org/download/)3.14.1以上版本，路径中不能出现中文和日文，安装完成后将`cmake.exe`的路径添加到系统环境变量Path中。
- 确认安装[ActivePerl](http://downloads.activestate.com/ActivePerl/releases/5.24.3.2404/ActivePerl-5.24.3.2404-MSWin32-x64-404865.exe) 5.24.3.2404。
- 确认安装[Python](https://www.python.org/ftp/python/3.7.5/python-3.7.5-amd64.exe)3.7.5  
    路径中不能出现中文和日文，安装完成后需要将`python.exe`的路径添加到系统环境变量Path中，Python自带的pip文件在python.exe同级目录的`Scripts`文件夹中，也需要将pip文件的路径添加到系统环境变量Path中。

## 从代码仓下载源码

```bash
git clone https://gitee.com/mindspore/mindspore.git -b r1.0
```

## 编译MindSpore

在源码根目录下执行如下命令。

```bash
call build.bat
```

## 安装MindSpore

```bash
pip install build/package/mindspore-{version}-cp37-cp37m-win_amd64.whl
```

> `{version}`表示MindSpore版本号，例如下载1.0.1版本MindSpore时，`{version}`应写为1.0.1。

## 验证是否安装成功

```bash
python -c "import mindspore;print(mindspore.__version__)"
```

如果输出MindSpore版本号，说明MindSpore安装成功了，如果输出`No module named 'mindspore'`说明安装未成功。
