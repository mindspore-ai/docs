# 源码编译方式安装MindSpore CPU版本（macOS）

<!-- TOC -->

- [源码编译方式安装MindSpore CPU版本（macOS）](#源码编译方式安装mindspore-cpu版本macOS)
    - [确认系统环境信息](#确认系统环境信息)
    - [从代码仓下载源码](#从代码仓下载源码)
    - [编译MindSpore](#编译mindspore)
    - [安装MindSpore](#安装mindspore)
    - [验证是否安装成功](#验证是否安装成功)
    - [升级MindSpore版本](#升级mindspore版本)

<!-- /TOC -->

<a href="https://gitee.com/mindspore/docs/blob/r1.1/install/mindspore_cpu_macos_install_source.md" target="_blank"><img src="https://gitee.com/mindspore/docs/raw/r1.1/resource/_static/logo_source.png"></a>

本文档介绍如何在CPU环境的macOS系统上，使用源码编译方法快速安装MindSpore。

## 确认系统环境信息

- 确认安装macOS Catalina是x86架构64位操作系统。
- 确认安装Xcode并配置clang version 11.0.0。
- 确认安装[CMake 3.18.3版本](https://github.com/Kitware/Cmake/releases/tag/v3.18.3)。
    - 安装完成后将CMake添加到系统环境变量。
- 确认安装[Python 3.7.5版本](https://www.python.org/ftp/python/3.7.5/python-3.7.5-macosx10.9.pkg)。
    - 安装完成后需要将Python添加到系统环境变量Path中。
- 确认安装[OpenSSL 1.1.1及以上版本](https://github.com/openssl/openssl.git)。
    - 安装完成后将Openssl添加到环境变量。
- 确认安装[wheel 0.32.0及以上版本](https://pypi.org/project/wheel/)。
- 确认安装git工具。  
    如果未安装，使用如下命令下载安装：

    ```bash
    brew install git
    ```

## 从代码仓下载源码

```bash
git clone https://gitee.com/mindspore/mindspore.git -b r1.1
```

## 编译MindSpore

在源码根目录下执行如下命令：

```bash
bash build.sh -e cpu
```

## 安装MindSpore

```bash
pip install build/package/mindspore-{version}-py37-none-any.whl -i https://pypi.tuna.tsinghua.edu.cn/simple
```

其中：

- 在联网状态下，安装whl包时会自动下载MindSpore安装包的依赖项（依赖项详情参见[requirements.txt](https://gitee.com/mindspore/mindspore/blob/r1.1/requirements.txt)），其余情况需自行安装。  
- `{version}`表示MindSpore版本号，例如下载1.0.1版本MindSpore时，`{version}`应写为1.0.1。

## 验证是否安装成功

```bash
python -c "import mindspore;print(mindspore.__version__)"
```

如果输出MindSpore版本号，说明MindSpore安装成功了，如果输出`No module named 'mindspore'`说明未安装成功。

## 升级MindSpore版本

当需要升级MindSpore版本时，可执行如下命令：

- 直接在线升级

    ```bash
    pip install --upgrade mindspore
    ```

- 本地源码编译升级

    在源码根目录下执行编译脚本`build.sh`成功后，在`build/package`目录下找到编译生成的whl安装包，然后执行命令进行升级。

    ```bash
    pip install --upgrade mindspore-{version}-py37-none-any.whl
    ```
