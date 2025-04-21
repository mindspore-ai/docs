# 源码编译方式安装MindSpore CPU版本-Windows

<!-- TOC -->

- [源码编译方式安装MindSpore CPU版本-Windows](#源码编译方式安装mindspore-cpu版本-windows)
    - [确认系统环境信息](#确认系统环境信息)
    - [从代码仓下载源码](#从代码仓下载源码)
    - [编译MindSpore](#编译mindspore)
    - [安装MindSpore](#安装mindspore)
    - [验证是否成功安装](#验证是否成功安装)
    - [升级MindSpore版本](#升级mindspore版本)

<!-- /TOC -->

[![查看源文件](https://mindspore-website.obs.cn-north-4.myhuaweicloud.com/website-images/br_base/resource/_static/logo_source.svg)](https://gitee.com/mindspore/docs/blob/br_base/install/mindspore_cpu_win_install_source.md)

本文档介绍如何在CPU环境的Windows系统上，使用源码编译方法快速安装MindSpore。

## 确认系统环境信息

- 确认安装Windows 10是x86架构64位操作系统。
- 确认安装[Microsoft Visual Studio Community 2019](https://learn.microsoft.com/zh-cn/visualstudio/releases/2019/release-notes)。安装时保证工作负荷勾选 `使用C++的桌面开发` 和 `通用Windows平台开发` 选项，单个组件勾选 `用于Windows的C++CMake工具` 。
- 确认安装了[git](https://github.com/git-for-windows/git/releases/download/v2.29.2.windows.2/Git-2.29.2.2-64-bit.exe)工具。同时将Git目录加到 `Path` 环境变量中，如果git安装在 `D:\Program Files\Git` 时，那么需要把 `D:\Program Files\Git\usr\bin`加入到 `Path` 环境变量中。
- 确认安装[CMake 3.22.2版本](https://cmake.org/files/v3.22/cmake-3.22.2-windows-x86_64.msi)。并将安装路径（不能出现中文等特殊字符）添加到系统环境变量 `Path` 中。
- 确认安装Python（>=3.9.0）。可以从[Python官网](https://www.python.org/downloads/windows/)或者[华为云](https://repo.huaweicloud.com/python/)选择合适的版本进行安装。
- 确认安装[wheel 0.32.0及以上版本](https://pypi.org/project/wheel/)。
- 确认安装[PyYAML](https://pypi.org/project/pyyaml/) (>=6.0 并且 <= 6.0.2)。如果没有安装，可以使用 `pip install pyyaml` 命令安装。
- 确认安装[MSYS2软件](https://www.msys2.org/)。详细请查看[Windows上安装MSYS2软件](https://gitee.com/mindspore/docs/blob/br_base/install/third_party/msys_software_install.md)。
- 确认安装[Numpy](https://pypi.org/project/numpy/) (>=1.19.3 并且 <= 1.26.4)。如果没有安装，可以使用 `pip install numpy` 命令安装。

## 从代码仓下载源码

```bash
git clone https://gitee.com/mindspore/mindspore.git
```

## 编译MindSpore

在源码根目录下执行以下命令：

```bash
call build.bat ms_vs_cpu
```

## 安装MindSpore

```bash
for %x in (output\mindspore*.whl) do pip install %x -i https://pypi.tuna.tsinghua.edu.cn/simple
```

在联网状态下，安装whl包时会自动下载mindspore安装包的依赖项（依赖项详情参见[setup.py](https://gitee.com/mindspore/mindspore/blob/br_base/setup.py)中的required_package），其余情况需自行安装依赖。

## 验证是否成功安装

执行以下命令：

```bash
cd ..
python -c "import mindspore;mindspore.set_device(device_target='CPU');mindspore.run_check()"
```

执行`import mindspore`应在MindSpore源码根目录之外，因为Windows上的Python将当前目录视为执行环境，可能产生目录查找问题。
如果输出：

```text
MindSpore version: 版本号
The result of multiplication calculation is correct, MindSpore has been installed on platform [CPU] successfully!
```

说明MindSpore安装成功了。

## 升级MindSpore版本

在源码根目录下执行编译脚本`build.bat`成功后，在`output`目录下找到编译生成的whl安装包，然后执行以下命令进行升级。

 ```bash
pip install --upgrade mindspore-*.whl
```
