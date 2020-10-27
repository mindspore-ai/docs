# Conda方式安装MindSpore

<!-- TOC -->

- [Conda方式安装MindSpore](#conda方式安装mindspore)
    - [确认系统环境信息](#确认系统环境信息)
    - [安装Conda](#安装conda)
    - [启动Anaconda Prompt](#启动anaconda-prompt)
    - [添加Conda清华镜像源](#添加conda清华镜像源)
    - [创建并激活Conda环境](#创建并激活conda环境)
    - [安装MindSpore](#安装mindspore)
    - [验证是否安装成功](#验证是否安装成功)

<!-- /TOC -->

<a href="https://gitee.com/mindspore/docs/blob/master/install/mindspore_cpu_win_install_conda.md" target="_blank"><img src="https://gitee.com/mindspore/docs/raw/master/resource/_static/logo_source.png"></a>

本文档介绍如何在CPU环境的Windows系统上，使用Conda方式快速安装MindSpore。

## 确认系统环境信息

- 需要确认您的Windows 7/8/10是64位操作系统。
- 确认安装[GCC](http://ftp.gnu.org/gnu/gcc/gcc-7.3.0/gcc-7.3.0.tar.gz)为7.3.0版本。

## 安装Conda

- 官方源下载[X86 Anaconda](https://www.anaconda.com/distribution/) 或 [X86 Miniconda](https://docs.conda.io/en/latest/miniconda.html)
- 清华镜像源下载地址：[X86 Anaconda](https://mirrors.tuna.tsinghua.edu.cn/anaconda/archive/Anaconda3-2020.02-Windows-x86_64.exe)

## 启动Anaconda Prompt

安装Conda后从Windows开始菜单打开Anaconda Prompt。

## 添加Conda清华镜像源

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
pip install https://ms-release.obs.cn-north-4.myhuaweicloud.com/{version}/MindSpore/cpu/windows_x64/mindspore-{version}-cp37-cp37m-win_amd64.whl --trusted-host ms-release.obs.cn-north-4.myhuaweicloud.com -i https://mirrors.huaweicloud.com/repository/pypi/simple
```

> `{version}`表示MindSpore版本号，例如下载1.0.1版本MindSpore时，`{version}`应写为1.0.1。

## 验证是否安装成功

```bash
python -c "import mindspore;mindspore.__version__"
```

如果输出MindSpore版本号，说明MindSpore安装成功了，如果输出`No module named 'mindspore'`说明安装未成功。
