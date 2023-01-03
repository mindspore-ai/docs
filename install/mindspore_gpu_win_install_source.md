# 源码编译方式安装MindSpore GPU版本-Windows

<!-- TOC -->

- [源码编译方式安装MindSpore GPU版本-Windows](#源码编译方式安装mindspore-gpu版本-windows)
    - [确认系统环境信息](#确认系统环境信息)
    - [从代码仓下载源码](#从代码仓下载源码)
    - [编译MindSpore](#编译mindspore)
    - [安装MindSpore](#安装mindspore)
    - [验证是否成功安装](#验证是否成功安装)
    - [升级MindSpore版本](#升级mindspore版本)

<!-- /TOC -->

<a href="https://gitee.com/mindspore/docs/blob/r2.0.0-alpha/install/mindspore_gpu_win_install_source.md" target="_blank"><img src="https://mindspore-website.obs.cn-north-4.myhuaweicloud.com/website-images/master/resource/_static/logo_source.png"></a>

本文档介绍如何在GPU环境的Windows系统上，使用源码编译方法快速安装MindSpore。

## 确认系统环境信息

- 当前支持Windows 10/11，x86架构64位的操作系统。
- 确认安装[Microsoft Visual Studio Community 2019](https://learn.microsoft.com/zh-cn/visualstudio/releases/2019/release-notes)。安装时保证勾选`使用C++的桌面开发`和`通用Windows平台开发`选项。
- 确认安装了[git](https://github.com/git-for-windows/git/releases/download/v2.29.2.windows.2/Git-2.29.2.2-64-bit.exe)工具。同时将Git目录加到`Path`环境变量中，如果git安装在`D:\Program Files\Git`时，那么需要把`D:\Program Files\Git\usr\bin`加入到`Path`环境变量中。
- 确认安装[CMake 3.18.3版本](https://github.com/Kitware/Cmake/releases/tag/v3.18.3)。并将安装路径（不能出现中文和日文）添加到系统环境变量`Path`中。
- 确认安装Python（>=3.7.5）。可以从[Python官网](https://www.python.org/downloads/windows/)或者[华为云](https://repo.huaweicloud.com/python/)选择合适的版本进行安装。
- 确认安装[wheel 0.32.0及以上版本](https://pypi.org/project/wheel/)。
- 确认安装[CUDA 11.1/11.6](https://developer.nvidia.com/cuda-11.1.1-download-archive)以及[cuDNN](https://developer.nvidia.com/cudnn)，cuDNN的安装可以参考Nvidia[官方文档](https://docs.nvidia.com/deeplearning/cudnn/install-guide/index.html)，同时需要将cuDNN的安装目录加入到`CUDNN_HOME`环境变量。

> 注意：建议先安装 Microsoft Visual Studio Community 2019，再安装 CUDA

## 从代码仓下载源码

```bash
git clone https://gitee.com/mindspore/mindspore.git -b r2.0.0-alpha
```

## 编译MindSpore

在源码根目录下执行如下命令：

```bash
set FROM_GITEE=1
call build.bat ms_vs_gpu
```

## 安装MindSpore

```bash
for %x in (output\mindspore_gpu*.whl) do pip install %x -i https://pypi.tuna.tsinghua.edu.cn/simple
```

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

    在源码根目录下执行编译脚本`build.bat`成功后，在`output`目录下找到编译生成的whl安装包，然后执行命令进行升级。

    ```bash
    pip install --upgrade mindspore-{version}-{python_version}-win_amd64.whl
    ```
