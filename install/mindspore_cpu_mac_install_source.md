# 源码编译方式安装MindSpore CPU版本-macOS

<!-- TOC -->

- [源码编译方式安装MindSpore CPU版本-macOS](#源码编译方式安装mindspore-cpu版本-macos)
    - [确认系统环境信息](#确认系统环境信息)
    - [安装Python](#安装python)
    - [确认安装Python依赖](#确认安装python依赖)
    - [从代码仓下载源码](#从代码仓下载源码)
    - [编译MindSpore](#编译mindspore)
    - [安装MindSpore](#安装mindspore)
    - [验证安装是否成功](#验证安装是否成功)
    - [升级MindSpore版本](#升级mindspore版本)

<!-- /TOC -->

[![查看源文件](https://mindspore-website.obs.cn-north-4.myhuaweicloud.com/website-images/master/resource/_static/logo_source.svg)](https://atomgit.com/mindspore/docs/blob/master/install/mindspore_cpu_mac_install_source.md)

## 确认系统环境信息

- 确认安装[Xcode](https://xcodereleases.com/)，12.4(X86)及13.4(M1) 已测试。

- 确认安装`Command Line Tools for Xcode`。如果没有安装，可以使用 `sudo xcode-select --install` 命令安装。

- 确认安装[CMake 3.22.3及以上版本](https://cmake.org/download/)。如果没有安装，可以使用 `brew install cmake` 命令安装。

- 确认安装[patch 2.5](https://ftp.gnu.org/gnu/patch/)。如果没有安装，可以使用 `brew install patch` 命令安装。

## 安装Python

请参照[Python官网](https://www.python.org/)自行安装Python，版本要求为3.10-3.12。

安装完成后，可以通过以下命令查看Python版本。

```bash
python --version
```

## 确认安装Python依赖

在用于编译MindSpore的Python环境中，确认下列Python库已经安装：

- 确认安装[wheel 0.32.0及以上版本](https://pypi.org/project/wheel/)。如果没有安装，可以使用 `pip install wheel` 命令安装。

- 确认安装[PyYAML](https://pypi.org/project/pyyaml/) (>=6.0 并且 <= 6.0.2)。如果没有安装，可以使用 `pip install pyyaml` 命令安装。

- 确认安装[autoconf](https://ftp.gnu.org/gnu/autoconf/)。如果没有安装，可以使用 `brew install autoconf` 命令安装。

- 确认安装[Numpy](https://pypi.org/project/numpy/) (>=1.19.3 并且 <= 1.26.4)。如果没有安装，可以使用 `pip install numpy` 命令安装。

## 从代码仓下载源码

```bash
git clone https://atomgit.com/mindspore/mindspore.git
```

## 编译MindSpore

在源码根目录下执行以下命令。

```bash
export CC=/usr/bin/clang
export CXX=/usr/bin/clang++
bash build.sh -e cpu -S on -j4  # -j 为编译时线程配置，如果CPU性能较好，使用多线程方式编译，参数通常为CPU核数的两倍
```

## 安装MindSpore

```bash
# install prerequisites
pip install scipy

pip install output/mindspore-*.whl -i https://repo.huaweicloud.com/repository/pypi/simple/
```

## 验证安装是否成功

执行以下命令：

```bash
python -c "import mindspore;mindspore.set_device(device_target='CPU');mindspore.run_check()"
```

如果输出：

```text
MindSpore version: 版本号
The result of multiplication calculation is correct, MindSpore has been installed on platform [CPU] successfully!
```

说明MindSpore安装成功了。

## 升级MindSpore版本

在源码根目录下执行编译脚本`build.sh`成功后，在`output`目录下找到编译生成的whl安装包，然后执行以下命令进行升级。

 ```bash
pip install --upgrade mindspore-*.whl
```
