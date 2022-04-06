# 安装MindQuantum

<a href="https://gitee.com/mindspore/docs/blob/master/docs/mindquantum/docs/source_zh_cn/mindquantum_install.md" target="_blank"><img src="https://mindspore-website.obs.cn-north-4.myhuaweicloud.com/website-images/master/resource/_static/logo_source.png"></a>

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

- 参考[MindSpore安装指南](https://www.mindspore.cn/install)，完成MindSpore的安装，要求至少1.4.0版本。
- 其余依赖请参见[setup.py](https://gitee.com/mindspore/mindquantum/blob/master/setup.py)。

## 安装方式

可以采用pip安装或者源码编译安装两种方式。

### pip安装

Linux-x86_64 Python3.7

```bash
pip install https://ms-release.obs.cn-north-4.myhuaweicloud.com/1.6.1/MindQuantum/x86_64/mindquantum-0.5.0-cp37-cp37m-linux_x86_64.whl
```

Windows-x64 Python3.7

```bash
pip install https://ms-release.obs.cn-north-4.myhuaweicloud.com/1.6.1/MindQuantum/x86_64/mindquantum-0.5.0-cp37-cp37m-win_amd64.whl
```

Windows-x64 Python3.9

```bash
pip install https://ms-release.obs.cn-north-4.myhuaweicloud.com/1.6.1/MindQuantum/x86_64/mindquantum-0.5.0-cp39-cp39-win_amd64.whl
```

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

通过Docker也可以在Mac系统或者Windows系统中使用Mindquantum。具体参考[Docker安装指南](https://gitee.com/mindspore/mindquantum/blob/master/install_with_docker.md).

## 注意事项

运行代码前请设置量子模拟器运行时并行内核数，例如设置并行内核数为4，可运行如下代码：

```bash
export OMP_NUM_THREADS=4
```

对于大型服务器，请根据模型规模合理设置并行内核数以达到最优效果。


