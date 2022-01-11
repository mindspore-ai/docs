# 安装MindQuantum

<a href="https://gitee.com/mindspore/docs/blob/r1.6/docs/mindquantum/docs/source_zh_cn/mindquantum_install.md" target="_blank"><img src="https://gitee.com/mindspore/docs/raw/r1.6/resource/_static/logo_source.png"></a>

<!-- TOC --->

- [安装MindQuantum](#安装mindquantum)
    - [确认系统环境信息](#确认系统环境信息)
    - [安装方式](#安装方式)
        - [源码安装](#源码安装)
        - [pip安装](#pip安装)
    - [验证是否成功安装](#验证是否成功安装)
    - [Docker安装](#docker安装)
    - [注意事项](#注意事项)

<!-- /TOC -->

## 确认系统环境信息

- 硬件平台确认为Linux系统下的CPU，并支持avx指令集。
- 参考[MindSpore安装指南](https://www.mindspore.cn/install)，完成MindSpore的安装，要求至少1.2.0版本。
- 其余依赖请参见[setup.py](https://gitee.com/mindspore/mindquantum/blob/r0.5/setup.py)。

## 安装方式

可以采用pip安装或者源码编译安装两种方式。

### pip安装

```bash
pip install https://ms-release.obs.cn-north-4.myhuaweicloud.com/{ms_version}/MindQuantum/any/mindquantum-{mq_version}-py3-none-any.whl --trusted-host ms-release.obs.cn-north-4.myhuaweicloud.com -i https://pypi.tuna.tsinghua.edu.cn/simple
```

> - 在联网状态下，安装whl包时会自动下载MindQuantum安装包的依赖项（依赖项详情参见[setup.py](https://gitee.com/mindspore/mindquantum/blob/r0.5/setup.py)），其余情况需自行安装。
> - `{ms_version}`表示与MindQuantum匹配的MindSpore版本号，例如下载0.3.0版本MindQuantum时，`{ms_version}`应写为1.5.0。
> - `{mq_version}`表示MindQuantum版本号，例如下载0.3.0版本MindQuantum时，`{mq_version}`应写为0.3.0。
> - 前往[官网](https://www.mindspore.cn/versions)可查询更多版本安装包。

### 源码安装

1. 从代码仓下载源码

    ```bash
    cd ~
    git clone https://gitee.com/mindspore/mindquantum.git
    ```

2. 编译安装MindQuantum

    ```bash
    cd ~/mindquantum
    python setup.py install --user
    ```

## 验证是否成功安装

执行如下命令，如果没有报错`No module named 'mindquantum'`，则说明安装成功。

```bash
python -c 'import mindquantum'
```

## Docker安装

通过Docker也可以在Mac系统或者Windows系统中使用Mindquantum。具体参考[Docker安装指南](https://gitee.com/mindspore/mindquantum/blob/r0.5/install_with_docker.md).

## 注意事项

运行代码前请设置量子模拟器运行时并行内核数，例如设置并行内核数为4，可运行如下代码：

```bash
export OMP_NUM_THREADS=4
```

对于大型服务器，请根据模型规模合理设置并行内核数以达到最优效果。


