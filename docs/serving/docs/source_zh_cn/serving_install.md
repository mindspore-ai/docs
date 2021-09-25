# 安装MindSpore Serving

<a href="https://gitee.com/mindspore/docs/blob/r1.5/docs/serving/docs/source_zh_cn/serving_install.md" target="_blank"><img src="https://gitee.com/mindspore/docs/raw/r1.5/resource/_static/logo_source.png"></a>

<!-- TOC -->

- [安装MindSpore Serving](安装#mindspore-serving)
    - [安装](#安装)
        - [pip安装](#pip安装)
        - [源码编译安装](#源码编译安装)
    - [验证是否成功安装](#验证是否成功安装)
    - [配置环境变量](#配置环境变量)

<!-- /TOC -->

## 安装

MindSpore Serving当前仅支持Ascend 310、Ascend 910和Nvidia GPU环境。

MindSpore Serving依赖MindSpore训练推理框架，安装完[MindSpore](https://gitee.com/mindspore/mindspore#%E5%AE%89%E8%A3%85)，再安装MindSpore Serving。可以采用pip安装或者源码编译安装两种方式。

### pip安装

使用pip命令安装，请从[MindSpore Serving下载页面](https://www.mindspore.cn/versions)下载并安装whl包。

 ```shell
pip install https://ms-release.obs.cn-north-4.myhuaweicloud.com/{version}/Serving/{arch}/mindspore_serving-{version}-{python_version}-linux_{arch}.whl --trusted-host ms-release.obs.cn-north-4.myhuaweicloud.com -i https://pypi.tuna.tsinghua.edu.cn/simple
```

> - `{version}`表示MindSpore Serving版本号，例如下载1.1.0版本MindSpore Serving时，`{version}`应写为1.1.0。
> - `{arch}`表示系统架构，例如使用的Linux系统是x86架构64位时，`{arch}`应写为`x86_64`。如果系统是ARM架构64位，则写为`aarch64`。
> - `{python_version}`表示用户的Python版本，Python版本为3.7.5时，`{python_version}`应写为`cp37-cp37m`。Python版本为3.9.0时，则写为`cp39-cp39`。请和当前安装的MindSpore使用的Python环境保持一致。

### 源码编译安装

下载[源码](https://gitee.com/mindspore/serving)，下载后进入`serving`目录。

方式一，指定Serving依赖的已安装或编译的MindSpore包路径，安装Serving：

```shell
sh build.sh -p $MINDSPORE_LIB_PATH
```

其中，`build.sh`为`serving`目录下的编译脚本文件，`$MINDSPORE_LIB_PATH`为MindSpore软件包的安装路径下的`lib`路径，例如，`softwarepath/mindspore/lib`，该路径包含MindSpore运行依赖的库文件。

方式二，直接编译Serving，编译时会配套编译MindSpore的包，需要配置MindSpore编译时的[环境变量](https://gitee.com/mindspore/docs/blob/r1.5/install/mindspore_ascend_install_source.md#配置环境变量) ：

```shell
# GPU
sh build.sh -e gpu
# Ascend 310 and Ascend 910
sh build.sh -e ascend
```

其中，`build.sh`为`serving`目录下的编译脚本文件，编译完后，在`serving/third_party/mindspore/build/package/`目录下找到MindSpore的whl安装包进行安装：

```shell
pip install mindspore_ascend-{version}-{python_version}-linux_{arch}.whl
```

同时在`serving/build/package/`目录下找到Serving的whl安装包进行安装：

```shell
pip install mindspore_serving-{version}-{python_version}-linux_{arch}.whl

```

## 验证是否成功安装

执行以下命令，验证安装结果。导入Python模块不报错即安装成功：

```python
from mindspore_serving import server
```

## 配置环境变量

MindSpore Serving运行需要配置以下环境变量：

- MindSpore Serving依赖MindSpore正确运行，运行MindSpore需要完成[环境变量配置](https://gitee.com/mindspore/docs/blob/r1.5/install/mindspore_ascend_install_pip.md#%E9%85%8D%E7%BD%AE%E7%8E%AF%E5%A2%83%E5%8F%98%E9%87%8F)。
