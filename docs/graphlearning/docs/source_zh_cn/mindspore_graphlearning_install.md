# 安装 Graph Learning

- [安装教程](#安装教程)
    - [确认系统环境信息](#确认系统环境信息)
    - [安装方式](#安装方式)
        - [pip安装](#pip安装)
        - [源码安装](#源码安装)
    - [验证是否成功安装](#验证是否成功安装)
- [社区](#社区)
    - [治理](#治理)
- [贡献](#贡献)
- [许可证](#许可证)

<a href="https://gitee.com/mindspore/docs/blob/master/docs/graphlearning/docs/source_zh_cn/mindspore_graphlearning_install.md" target="_blank"><img src="https://mindspore-website.obs.cn-north-4.myhuaweicloud.com/website-images/master/resource/_static/logo_source.png"></a>&nbsp;&nbsp;

## 安装指南

### 确认系统环境信息

- 硬件平台确认为Linux系统下的GPU。
- 参考[MindSpore安装指南](https://www.mindspore.cn/install)，完成MindSpore的安装，要求至少1.6.0版本。
- 其余依赖请参见[requirements.txt](https://gitee.com/mindspore/graphlearning/blob/master/requirements.txt)。

### 安装方式

可以采用pip安装或者源码编译安装两种方式。

#### pip安装

```bash
pip install https://ms-release.obs.cn-north-4.myhuaweicloud.com/{version}/GraphLearning/any/mindspore_gl_gpu-{version}-cp37-cp37m-linux_x86_64.whl --trusted-host ms-release.obs.cn-north-4.myhuaweicloud.com -i https://pypi.tuna.tsinghua.edu.cn/simple
```

> - 在联网状态下，安装whl包时会自动下载MindSpore Graph Learning安装包的依赖项（依赖项详情参见[requirements.txt](https://gitee.com/mindspore/graphlearning/blob/master/requirements.txt)），其余情况需自行安装。
> - `{version}`表示MindSpore Graph Learning版本号，例如下载0.1版本MindSpore Graph Learning时，`{version}`应写为0.1。

#### 源码安装

1. 从代码仓下载源码

    ```bash
    git clone https://gitee.com/mindspore/graphlearning.git
    ```

2. 编译安装MindSpore Graph Learning

    ```bash
    cd graphlearning
    bash build.sh
    pip install ./output/mindspore_gl_gpu-*.whl
    ```

### 验证是否成功安装

执行如下命令，如果没有报错`No module named 'mindspore_gl'`，则说明安装成功。

```bash
python -c 'import mindspore_gl'
```
