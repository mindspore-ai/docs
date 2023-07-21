# Conda方式安装MindSpore CPU版本（Windows）

<!-- TOC -->

- [Conda方式安装MindSpore CPU版本（Windows）](#conda方式安装mindspore-cpu版本windows)
    - [确认系统环境信息](#确认系统环境信息)
    - [安装Conda](#安装conda)
    - [启动Anaconda Prompt](#启动anaconda-prompt)
    - [添加Conda镜像源](#添加conda镜像源)
    - [创建并激活Conda环境](#创建并激活conda环境)
    - [安装MindSpore](#安装mindspore)
    - [验证是否安装成功](#验证是否安装成功)
    - [升级MindSpore版本](#升级mindspore版本)

<!-- /TOC -->

[![查看源文件](https://gitee.com/mindspore/docs/raw/r1.0/resource/_static/logo_source.png)](https://gitee.com/mindspore/docs/blob/r1.0/install/mindspore_cpu_win_install_conda.md)

本文档介绍如何在CPU环境的Windows系统上，使用Conda方式快速安装MindSpore。

## 确认系统环境信息

- 确认安装Windows 10是x86架构64位操作系统。

## 安装Conda

下载并安装对应架构的Conda安装包。

- 官方源下载[X86 Anaconda](https://www.anaconda.com/distribution/) 或 [X86 Miniconda](https://docs.conda.io/en/latest/miniconda.html)
- 清华镜像源下载地址：[X86 Anaconda](https://mirrors.tuna.tsinghua.edu.cn/anaconda/archive/Anaconda3-2020.02-Windows-x86_64.exe)

## 启动Anaconda Prompt

安装Conda后，从Windows“开始”菜单打开“Anaconda Prompt”。

## 添加Conda镜像源

从清华源镜像源下载Conda安装包的可忽略此步操作。

```shell
conda config --add channels https://mirrors.tuna.tsinghua.edu.cn/anaconda/pkgs/free/
conda config --add channels https://mirrors.tuna.tsinghua.edu.cn/anaconda/pkgs/main/
conda config --set show_channel_urls yes
```

## 创建并激活Conda环境

```bash
conda create -n mindspore python=3.7.5
conda activate mindspore
```

## 安装MindSpore

```bash
pip install https://ms-release.obs.cn-north-4.myhuaweicloud.com/{version}/MindSpore/cpu/windows_x64/mindspore-{version}-cp37-cp37m-win_amd64.whl --trusted-host ms-release.obs.cn-north-4.myhuaweicloud.com -i https://pypi.tuna.tsinghua.edu.cn/simple
```

其中：

- 在联网状态下，安装whl包时会自动下载MindSpore安装包的依赖项（依赖项详情参见[requirements.txt](https://gitee.com/mindspore/mindspore/blob/r1.0/requirements.txt)），其余情况需自行安装。  
- `{version}`表示MindSpore版本号，例如下载1.0.1版本MindSpore时，`{version}`应写为1.0.1。

## 验证是否安装成功

```bash
python -c "import mindspore;mindspore.__version__"
```

如果输出MindSpore版本号，说明MindSpore安装成功了，如果输出`No module named 'mindspore'`说明未安装成功。

## 升级MindSpore版本

当需要升级MindSpore版本时，可执行如下命令：

```bash
pip install --upgrade mindspore
```
