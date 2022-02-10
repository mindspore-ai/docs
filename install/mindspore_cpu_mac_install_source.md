# 源码编译方式安装MindSpore CPU版本（macOS）

<!-- TOC -->

- [源码编译方式安装MindSpore CPU版本（macOS）](#源码编译方式安装mindspore-cpu版本macos)
    - [确认系统环境信息](#确认系统环境信息)
    - [从代码仓下载源码](#从代码仓下载源码)
    - [编译MindSpore](#编译mindspore)
    - [安装MindSpore](#安装mindspore)
    - [验证安装是否成功](#验证安装是否成功)
    - [升级MindSpore版本](#升级mindspore版本)

<!-- /TOC -->

<a href="https://gitee.com/mindspore/docs/blob/r1.6/install/mindspore_cpu_mac_install_source.md" target="_blank"><img src="https://gitee.com/mindspore/docs/raw/r1.6/resource/_static/logo_source.png"></a>

本文档介绍如何在macOS系统上使用源码编译方式快速安装MindSpore。

## 确认系统环境信息

- 根据下表中的系统及芯片情况确定合适的Python版本，macOS版本及芯片信息可点击桌面左上角苹果标志->`关于本机`获悉：

    |芯片|计算架构|macOS版本|支持Python版本|
    |-|-|-|-|
    |M1|ARM|11.3|Python 3.9.1+（M1当前不支持3.7, 3.9支持最低版本为3.9.1）|
    |Intel|x86_64|10.15/11.3|Python 3.7.5/Python 3.9.0|

- 确认安装对应的Python版本。如果未安装或者已安装其他版本的Python，可以选择下载并安装：

    - Python 3.7.5 (64-bit)：[官网](https://www.python.org/ftp/python/3.7.5/python-3.7.5-macosx10.9.pkg) 或者 [华为云](https://repo.huaweicloud.com/python/3.7.5/python-3.7.5-macosx10.9.pkg)。
    - Python 3.9.0 (64-bit)：[官网](https://www.python.org/ftp/python/3.9.0/python-3.9.0-macosx10.9.pkg) 或者 [华为云](https://repo.huaweicloud.com/python/3.9.0/python-3.9.0-macosx10.9.pkg)。
    - Python 3.9.1 (64-bit)：[官网](https://www.python.org/ftp/python/3.9.1/python-3.9.1-macos11.0.pkg) 或者 [华为云](https://www.python.org/ftp/python/3.9.1/python-3.9.1-macos11.0.pkg)。

- 确认安装[Xcode](https://xcodereleases.com/) (>=12.4 and <= 13.0) ，12.4(X86)及13.0(M1) 已测试。

- 确认安装`Command Line Tools for Xcode`。如果未安装，使用命令`sudo xcode-select --install`安装Command Line Tools。

- 确认安装[CMake 3.18.3及以上版本](https://cmake.org/download/)。如果没有安装，可以使用`brew install cmake`进行安装。

- 确认安装[patch 2.5](https://ftp.gnu.org/gnu/patch/)。如果没有安装，可以使用`brew install patch`进行安装。

- 确认安装[wheel 0.32.0及以上版本](https://pypi.org/project/wheel/)。如果没有安装，可以使用`pip install wheel` 进行安装。

## 从代码仓下载源码

```bash
git clone https://gitee.com/mindspore/mindspore.git -b r1.6
```

## 编译MindSpore

在源码根目录下执行如下命令。

```bash
export CC=/usr/bin/clang
export CXX=/usr/bin/clang++
bash build.sh -e cpu -S on -j4  # -j 为编译时线程配置，如果CPU性能较好，使用多线程方式编译，参数通常为CPU核数的两倍
```

## 安装MindSpore

```bash
pip install output/mindspore-*.whl -i https://pypi.tuna.tsinghua.edu.cn/simple
```

如果在安装scipy包时出现编译错误，可以尝试先使用下面的命令安装scipy包，再安装MindSpore包。

```bash
pip install --pre -i https://pypi.anaconda.org/scipy-wheels-nightly/simple scipy
```

## 验证安装是否成功

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

- 直接在线升级

    ```bash
    pip install --upgrade mindspore
    ```

- 本地源码编译升级

    在源码根目录下执行编译脚本`build.sh`成功后，在`output`目录下找到编译生成的whl安装包，然后执行命令进行升级。

    ```bash
    pip install --upgrade mindspore-{version}-{python_version}-macosx_{platform_version}_{arch}.whl
    ```