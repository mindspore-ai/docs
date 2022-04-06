# 安装MindSpore Hub

- [安装MindSpore Hub](#安装mindspore-hub)
    - [确认系统环境信息](#确认系统环境信息)
    - [安装方式](#安装方式)
        - [pip安装](#pip安装)
        - [源码安装](#源码安装)
    - [验证是否成功安装](#验证是否成功安装)

<a href="https://gitee.com/mindspore/docs/blob/master/docs/hub/docs/source_zh_cn/hub_installation.md" target="_blank"><img src="https://mindspore-website.obs.cn-north-4.myhuaweicloud.com/website-images/master/resource/_static/logo_source.png"></a>

## 确认系统环境信息

- 硬件平台支持Ascend、GPU和CPU。
- 确认安装[Python](https://www.python.org/ftp/python/3.7.5/Python-3.7.5.tgz) 3.7.5版本。
- MindSpore Hub与MindSpore的版本需保持一致。
- MindSpore Hub支持使用x86 64位或ARM 64位架构的Linux发行版系统。
- 在联网状态下，安装whl包时会自动下载`setup.py`中的依赖项，其余情况需自行安装。

## 安装方式

可以采用pip安装或者源码安装两种方式。

### pip安装

下载并安装[发布版本列表](https://www.mindspore.cn/versions)中的MindSpore Hub whl包。

```shell
pip install https://ms-release.obs.cn-north-4.myhuaweicloud.com/{version}/Hub/any/mindspore_hub-{version}-py3-none-any.whl --trusted-host ms-release.obs.cn-north-4.myhuaweicloud.com -i https://pypi.tuna.tsinghua.edu.cn/simple
```

> - `{version}`表示MindSpore Hub版本号，例如下载1.3.0版本MindSpore Hub时，`{version}`应写为1.3.0。

### 源码安装

1. 从Gitee下载源码。

   ```bash
   git clone https://gitee.com/mindspore/hub.git
   ```

2. 编译安装MindSpore Hub。

   ```bash
   cd hub
   python setup.py install
   ```

## 验证是否成功安装

在能联网的环境中执行以下命令，验证安装结果。

```python
import mindspore_hub as mshub

model = mshub.load("mindspore/cpu/1.0/lenet_v1_mnist", num_class = 10)
```

如果出现下列提示，说明安装成功：

```text
Downloading data from url https://gitee.com/mindspore/hub/raw/master/mshub_res/assets/mindspore/cpu/1.0/lenet_v1_mnist.md

Download finished!
File size = 0.00 Mb
Checking /home/ma-user/.mscache/mindspore/cpu/1.0/lenet_v1_mnist.md...Passed!
```