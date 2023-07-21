# pip方式安装MindSpore CPU版本

<!-- TOC -->

- [pip方式安装MindSpore CPU版本](#pip方式安装mindspore-cpu版本)
    - [确认系统环境信息](#确认系统环境信息)
    - [下载安装MindSpore](#下载安装mindspore)
    - [查询安装是否成功](#查询安装是否成功)
    - [升级MindSpore版本](#升级mindspore版本)
    - [安装MindArmour](#安装mindarmour)
    - [安装MindSpore Hub](#安装mindspore-hub)

<!-- /TOC -->

[![查看源文件](https://gitee.com/mindspore/docs/raw/r1.1/resource/_static/logo_source.png)](https://gitee.com/mindspore/docs/blob/r1.1/install/mindspore_cpu_install_pip.md)

本文档介绍如何在CPU环境的Linux系统上，使用pip方式快速安装MindSpore。

## 确认系统环境信息

- 确认安装Ubuntu 18.04是64位操作系统。
- 确认安装[GCC 7.3.0版本](http://ftp.gnu.org/gnu/gcc/gcc-7.3.0/gcc-7.3.0.tar.gz)。
- 确认安装Python 3.7.5版本。  
    - 如果未安装或者已安装其他版本的Python，可从[官网](https://www.python.org/ftp/python/3.7.5/Python-3.7.5.tgz)或者[华为云](https://mirrors.huaweicloud.com/python/3.7.5/Python-3.7.5.tgz)下载Python 3.7.5版本，进行安装。

## 下载安装MindSpore

```bash
pip install https://ms-release.obs.cn-north-4.myhuaweicloud.com/{version}/MindSpore/cpu/{system}/mindspore-{version}-cp37-cp37m-linux_{arch}.whl --trusted-host ms-release.obs.cn-north-4.myhuaweicloud.com -i https://pypi.tuna.tsinghua.edu.cn/simple
```

其中：

- 在联网状态下，安装whl包时会自动下载MindSpore安装包的依赖项（依赖项详情参见[requirements.txt](https://gitee.com/mindspore/mindspore/blob/r1.1/requirements.txt)），其余情况需自行安装。  
- `{version}`表示MindSpore版本号，例如安装1.1.0版本MindSpore时，`{version}`应写为1.1.0。  
- `{arch}`表示系统架构，例如使用的Linux系统是x86架构64位时，`{arch}`应写为`x86_64`。如果系统是ARM架构64位，则写为`aarch64`。  
- `{system}`表示系统，例如使用的Ubuntu系统X86架构，`{system}`应写为`ubuntu_x86`，目前CPU版本可支持以下系统`ubuntu_aarch64`/`ubuntu_x86`。

## 查询安装是否成功

```bash
python -c "import mindspore;print(mindspore.__version__)"
```

如果输出MindSpore版本号，说明MindSpore安装成功了，如果输出`No module named 'mindspore'`说明未成功安装。

## 升级MindSpore版本

当需要升级MindSpore版本时，可执行如下命令：

```bash
pip install --upgrade mindspore
```

## 安装MindArmour

当您进行AI模型安全研究或想要增强AI应用模型的防护能力时，可以选装MindArmour。

具体安装步骤参见[MindArmour](https://gitee.com/mindspore/mindarmour/blob/r1.1/README_CN.md)。

## 安装MindSpore Hub

当您想要快速体验MindSpore预训练模型时，可以选装MindSpore Hub。

具体安装步骤参见[MindSpore Hub](https://gitee.com/mindspore/hub/blob/r1.1/README_CN.md)。
