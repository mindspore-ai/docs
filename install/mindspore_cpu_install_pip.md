# Pip方式安装MindSpore

<!-- TOC -->

- [Pip方式安装MindSpore](#pip方式安装mindspore)
    - [确认系统环境信息](#确认系统环境信息)
    - [下载安装MindSpore](#下载安装mindspore)
    - [查询安装是否成功](#查询安装是否成功)

<!-- /TOC -->

<a href="https://gitee.com/mindspore/docs/blob/master/install/mindspore_cpu_install_pip.md" target="_blank"><img src="https://gitee.com/mindspore/docs/raw/master/resource/_static/logo_source.png"></a>

本文档介绍如何在CPU环境的Linux系统上，使用Pip方式快速安装MindSpore。

## 确认系统环境信息

- 确认安装Ubuntu18.04是64位操作系统。
- 确认安装[GCC](http://ftp.gnu.org/gnu/gcc/gcc-7.3.0/gcc-7.3.0.tar.gz)7.3.0版本。
- 确认安装Python 3.7.5版本。
    如果未安装或者安装其他版本的Python，可从[官网](https://www.python.org/ftp/python/3.7.5/Python-3.7.5.tgz)或者[华为云](https://mirrors.huaweicloud.com/python/3.7.5/Python-3.7.5.tgz)下载Python 3.7.5版本，进行安装。

## 下载安装MindSpore

```bash
pip install https://ms-release.obs.cn-north-4.myhuaweicloud.com/{version}/MindSpore/cpu/ubuntu_x86/mindspore-{version}-cp37-cp37m-linux_{arch}.whl --trusted-host ms-release.obs.cn-north-4.myhuaweicloud.com -i https://mirrors.huaweicloud.com/repository/pypi/simple
```

> `{version}`表示MindSpore版本号，例如下载1.0.1版本MindSpore时，`{version}`应写为1.0.1。  
> `{arch}`表示系统架构，例如使用的Linux系统是x86架构64位时，`{arch}`应写为`x86_64`。如果系统是ARMv8架构64位，则写为`aarch64`。

## 查询安装是否成功

```bash
python -c "import mindspore;print(mindspore.__version__)"
```

如果输出MindSpore版本号，说明MindSpore安装成功了，如果输出`No module named 'mindspore'`说明未成功安装。
