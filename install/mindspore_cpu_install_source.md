# 源码编译方式安装MindSpore

<!-- TOC -->

- [源码编译方式安装MindSpore](#源码编译方式安装mindspore)
    - [确认系统环境信息](#确认系统环境信息)
    - [从代码仓下载源码](#从代码仓下载源码)
    - [编译MindSpore](#编译mindspore)
    - [安装MindSpore](#安装mindspore)
    - [验证安装是否成功](#验证安装是否成功)

<!-- /TOC -->

<a href="https://gitee.com/mindspore/docs/blob/master/install/mindspore_cpu_install_source.md" target="_blank"><img src="https://gitee.com/mindspore/docs/raw/master/resource/_static/logo_source.png"></a>

本文档介绍如何在CPU环境的Linux系统上，使用源码编译方式快速安装MindSpore。

## 确认系统环境信息

- 确认安装Ubuntu18.04是64位操作系统。
- 确认安装[Python3.7.5](https://www.python.org/ftp/python/3.7.5/Python-3.7.5.tgz)。
- 确认安装[GCC](http://ftp.gnu.org/gnu/gcc/gcc-7.3.0/gcc-7.3.0.tar.gz)7.3.0。
- 确认安装[CMake](https://cmake.org/download/)3.14.1以上版本。  
    安装完成后需将CMake添加到系统环境变量。
- 确认安装[wheel](https://pypi.org/project/wheel/)0.32.0以上版本。
- 确认安装[patch](http://ftp.gnu.org/gnu/patch/)2.5以上版本。  
    安装完成后需将patch添加到系统环境变量中。
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
pip install build/package/mindspore-{version}-cp37-cp37m-linux_{arch}.whl --trusted-host ms-release.obs.cn-north-4.myhuaweicloud.com -i https://mirrors.huaweicloud.com/repository/pypi/simple
```

> `{version}`表示MindSpore版本号，例如下载1.0.1版本MindSpore时，`{version}`应写为1.0.1。  
> `{arch}`表示系统架构，例如使用的Linux系统是x86架构64位时，`{arch}`应写为`x86_64`。如果系统是ARMv8架构64位，则写为`aarch64`。

## 验证安装是否成功

```bash
python -c 'import mindspore;print(mindspore.__version__)'
```

如果输出MindSpore版本号，说明MindSpore安装成功了，如果输出`No module named 'mindspore'`说明未成功安装。
