# 源码编译方式安装MindSpore CPU版本

<!-- TOC -->

- [源码编译方式安装MindSpore CPU版本](#源码编译方式安装mindspore-cpu版本)
    - [确认系统环境信息](#确认系统环境信息)
    - [从代码仓下载源码](#从代码仓下载源码)
    - [编译MindSpore](#编译mindspore)
    - [安装MindSpore](#安装mindspore)
    - [验证安装是否成功](#验证安装是否成功)
    - [升级MindSpore版本](#升级mindspore版本)

<!-- /TOC -->

<a href="https://gitee.com/mindspore/docs/blob/master/install/mindspore_cpu_install_source.md" target="_blank"><img src="https://gitee.com/mindspore/docs/raw/master/resource/_static/logo_source.png"></a>

本文档介绍如何在CPU环境的Linux系统上，使用源码编译方式快速安装MindSpore。

## 确认系统环境信息

- 确认安装Ubuntu 18.04是64位操作系统。
- 确认安装[GCC](http://ftp.gnu.org/gnu/gcc/gcc-7.3.0/gcc-7.3.0.tar.gz) 7.3.0版本。
- 确认安装[Python](https://www.python.org/ftp/python/3.7.5/Python-3.7.5.tgz) 3.7.5版本。
- 确认安装[OpenSSL](https://github.com/openssl/openssl.git) 1.1.1及以上版本。
    - 在执行上述命令前，需保证已经安装了[OpenSSL](https://github.com/openssl/openssl.git)，并设置环境变量`export OPENSSL_ROOT_DIR=“OpenSSL安装目录”`。
- 确认安装[CMake](https://cmake.org/download/) 3.14.1及以上版本。  
    安装完成后需将CMake所在路径添加到系统环境变量。
- 确认安装[wheel](https://pypi.org/project/wheel/) 0.32.0及以上版本。
- 确认安装[patch](http://ftp.gnu.org/gnu/patch/) 2.5及以上版本。  
    安装完成后需将patch所在路径添加到系统环境变量中。
- 确认安装`git`工具。  
    如果未安装，使用如下命令下载安装：

    ```bash
    apt-get install git
    ```

## 从代码仓下载源码

```bash
git clone https://gitee.com/mindspore/mindspore.git
```

## 编译MindSpore

在源码根目录下执行如下命令。

```bash
bash build.sh -e cpu -j4
```

> 如果编译机性能较好，可在执行中增加-j{线程数}来增加线程数量。如`bash build.sh -e cpu -j12`。

## 安装MindSpore

```bash
chmod +x build/package/mindspore-{version}-cp37-cp37m-linux_{arch}.whl
pip install build/package/mindspore-{version}-cp37-cp37m-linux_{arch}.whl -i https://mirrors.huaweicloud.com/repository/pypi/simple
```

> 在联网状态下，安装whl包时会自动下载MindSpore安装包的依赖项（依赖项详情参见[requirements.txt](https://gitee.com/mindspore/mindinsight/blob/master/requirements.txt)），其余情况需自行安装。  
> `{version}`表示MindSpore版本号，例如下载1.0.1版本MindSpore时，`{version}`应写为1.0.1。  
> `{arch}`表示系统架构，例如使用的Linux系统是x86架构64位时，`{arch}`应写为`x86_64`。如果系统是ARMv8架构64位，则写为`aarch64`。

## 验证安装是否成功

```bash
python -c 'import mindspore;print(mindspore.__version__)'
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
    pip install --upgrade mindspore-{version}-cp37-cp37m-linux_{arch}.whl
    ```
