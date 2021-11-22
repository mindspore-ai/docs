# 源码编译方式安装MindSpore CPU版本（macOS）

<!-- TOC -->

- [源码编译方式安装MindSpore CPU版本（macOS）](#源码编译方式安装mindspore-cpu版本macos)
    - [确认系统环境信息](#确认系统环境信息)
    - [从代码仓下载源码](#从代码仓下载源码)
    - [编译MindSpore](#编译mindspore)
    - [安装MindSpore](#安装mindspore)
    - [验证安装是否成功](#验证安装是否成功)

<!-- /TOC -->

<a href="https://gitee.com/mindspore/docs/blob/master/install/mindspore_cpu_mac_install_source.md" target="_blank"><img src="https://gitee.com/mindspore/docs/raw/master/resource/_static/logo_source.png"></a>

本文档介绍如何在CPU环境的macOS系统上，使用源码编译方式快速安装MindSpore。

## 确认系统环境信息

- 确认安装macOS，支持x64和ARM64架构，其中macOS 10.15、macOS 11.3是经过验证的。

- 确认安装[Xcode](https://developer.apple.com/xcode/)，其中Xcode 10.2、Xcode 12.5是经过验证的。

- 确认安装Command Line Tools for Xcode。
    如果未安装，使用如下命令下载安装Command Line Tools：

    ```bash
    xcode-select --install
    sudo xcode-select -switch /Library/Developer/CommandLineTools
    ```

- 确认安装Python 3.9版本。如果未安装或者已安装其他版本的Python，可以选择下载并安装：

    - Python 64位，下载地址：[官网](https://www.python.org/downloads/macos/)或[华为云](https://repo.huaweicloud.com/python/)。

- 确认安装[wheel 0.32.0及以上版本](https://pypi.org/project/wheel/)。

- 确认安装[patch 2.5及以上版本](http://ftp.gnu.org/gnu/patch/)。
    - 安装完成后需将patch所在路径添加到系统环境变量中。

## 从代码仓下载源码

```bash
git clone https://gitee.com/mindspore/mindspore.git
```

## 编译MindSpore

在源码根目录下执行如下命令。

```bash
export CC=/usr/bin/clang
export CXX=/usr/bin/clang++
export SDKROOT="`xcrun --show-sdk-path`"
bash build.sh -e cpu -S on -j4
```

其中：  
如果编译机性能较好，可在执行中增加-j{线程数}来增加线程数量。如`bash build.sh -e cpu -S on -j12`。

## 安装MindSpore

```bash
pip install output/mindspore-{version}-{python_version}-any.whl -i https://pypi.tuna.tsinghua.edu.cn/simple
```

如果在安装scipy包时出现编译错误，可以尝试先使用下面的命令安装scipy包，再安装MindSpore包。

```bash
pip install --pre -i https://pypi.anaconda.org/scipy-wheels-nightly/simple scipy
```

其中：

- 在联网状态下，安装whl包时会自动下载mindspore安装包的依赖项（依赖项详情参见[setup.py](https://gitee.com/mindspore/mindspore/blob/master/setup.py)中的required_package），其余情况需自行安装。运行模型时，需要根据[ModelZoo](https://gitee.com/mindspore/models/tree/master/)中不同模型指定的requirements.txt安装额外依赖，常见依赖可以参考[requirements.txt](https://gitee.com/mindspore/mindspore/blob/master/requirements.txt)。
- `{version}`表示MindSpore版本号，例如安装1.6.0版本MindSpore时，`{version}`应写为1.6.0。
- `{python_version}`表示用户的Python版本，Python版本为3.9.0时，则写为`cp39-cp39`。

## 验证安装是否成功

```bash
python -c "import mindspore;mindspore.run_check()"
```

如果输出：

```text
mindspore version: 版本号
The result of multiplication calculation is correct, MindSpore has been installed successfully!
```

说明MindSpore安装成功了。
