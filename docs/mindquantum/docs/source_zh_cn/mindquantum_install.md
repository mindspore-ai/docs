# 安装MindQuantum

<a href="https://gitee.com/mindspore/docs/blob/r1.10/docs/mindquantum/docs/source_zh_cn/mindquantum_install.md" target="_blank"><img src="https://mindspore-website.obs.cn-north-4.myhuaweicloud.com/website-images/r1.10/resource/_static/logo_source.png"></a>

## 确认系统环境信息

- 参考[MindSpore安装指南](https://www.mindspore.cn/install)，完成MindSpore的安装，要求至少1.4.0版本。

## 安装方式

可以采用pip安装或者源码编译安装两种方式。

### pip安装

```bash
pip install mindquantum
```

> - 前往[官网](https://www.mindspore.cn/versions)可查询更多版本安装包。

### 源码安装

1. 从代码仓下载源码

    ```bash
    cd ~
    git clone https://gitee.com/mindspore/mindquantum.git -b r0.8
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

通过Docker也可以在Mac系统或者Windows系统中使用Mindquantum。具体参考[Docker安装指南](https://gitee.com/mindspore/mindquantum/blob/r0.8/install_with_docker.md#).
