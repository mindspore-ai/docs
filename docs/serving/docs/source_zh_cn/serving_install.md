# 安装MindSpore Serving

<a href="https://gitee.com/mindspore/docs/blob/r1.7/docs/serving/docs/source_zh_cn/serving_install.md" target="_blank"><img src="https://mindspore-website.obs.cn-north-4.myhuaweicloud.com/website-images/r1.7/resource/_static/logo_source.png"></a>

## 安装

MindSpore Serving包在各类硬件平台（Nvidia GPU, Ascend 910/310P/310, CPU）上通用，推理任务依赖MindSpore或MindSpore Lite推理框架，我们需要选择一个作为Serving推理后端。当这两个推理后端同时存在的时候，优先使用MindSpore Lite推理框架。

MindSpore和MindSpore Lite针对不同的硬件平台有不同的构建包，每个不同的构建包支持的运行目标设备和模型格式如下表所示：

|推理后端|构建平台|运行目标设备|支持的模型格式|
|---------| --- | --- | -------- |
|MindSpore| Nvidia GPU | Nvidia GPU | `MindIR` |
|  | Ascend | Ascend 910 | `MindIR` |
|  |  | Ascend 310P/310 | `MindIR`, `OM` |
|MindSpore Lite| Nvidia GPU | Nvidia GPU, CPU | `MindIR_Lite` |
|  | Ascend | Ascend 310P/310, CPU | `MindIR`, `MindIR_Lite` |
|  | CPU | CPU | `MindIR_Lite` |

当以[MindSpore](https://www.mindspore.cn/)作为推理后端时，MindSpore Serving当前支持Ascend 910/310P/310和Nvidia GPU环境。其中Ascend 310P/310环境支持`OM`和`MindIR`两种模型格式，Ascend 910和GPU环境仅支持`MindIR`模型格式。

MindSpore的安装和配置可以参考[安装MindSpore](https://gitee.com/mindspore/mindspore#安装)，并根据需要完成[环境变量配置](https://gitee.com/mindspore/docs/blob/r1.7/install/mindspore_ascend_install_pip.md#配置环境变量)。

当以[MindSpore Lite](https://www.mindspore.cn/lite)作为推理后端时，MindSpore Serving当前支持Ascend 310P/310、Nvidia GPU和CPU。当前支持`MindIR`和`MindIR_Lite`模型格式，MindSpore的`MindIR`或其他框架的模型文件需要通过Lite转换工具转换成`MindIR_Lite`模型格式。模型转换时，`Ascend 310`设备和`Ascend 310P`转换出的模型不一致，需要在对应的`Ascend 310`或者`Ascend 310P`设备上运行；Nvidia GPU和CPU环境转换成的`MindIR_Lite`模型仅能在Nvidia GPU和CPU使用。

| 推理后端       | 转换工具运行平台 | `MindIR_Lite`模型运行设备    |
| -------------- | ---------------- | --------------- |
| MindSpore Lite | Nvidia GPU, CPU  | Nvidia GPU, CPU |
|                | Ascend 310       | Ascend 310      |
|                | Ascend 310P       | Ascend 310P      |

MindSpore Lite安装和配置可以参考[MindSpore Lite文档](https://www.mindspore.cn/lite/docs/zh-CN/r1.7/index.html)，通过环境变量`LD_LIBRARY_PATH`指示`libmindspore-lite.so`的安装路径。

MindSpore Serving的安装可以采用pip安装或者源码编译安装两种方式。

### pip安装

使用pip命令安装，请从[MindSpore Serving下载页面](https://www.mindspore.cn/versions)下载并安装whl包。

 ```shell
pip install https://ms-release.obs.cn-north-4.myhuaweicloud.com/{version}/Serving/{arch}/mindspore_serving-{version}-{python_version}-linux_{arch}.whl --trusted-host ms-release.obs.cn-north-4.myhuaweicloud.com -i https://pypi.tuna.tsinghua.edu.cn/simple
```

> - `{version}`表示MindSpore Serving版本号，例如下载1.1.0版本MindSpore Serving时，`{version}`应写为1.1.0。
> - `{arch}`表示系统架构，例如使用的Linux系统是x86架构64位时，`{arch}`应写为`x86_64`。如果系统是ARM架构64位，则写为`aarch64`。
> - `{python_version}`表示用户的Python版本，Python版本为3.7.5时，`{python_version}`应写为`cp37-cp37m`。Python版本为3.9.0时，则写为`cp39-cp39`。请和当前安装的MindSpore Serving使用的Python环境保持一致。

### 源码编译安装

通过[源码](https://gitee.com/mindspore/serving)编译安装。

```shell
git clone https://gitee.com/mindspore/serving.git -b r1.7
cd serving
bash build.sh
```

对于`bash build.sh`，可通过例如`-jn`选项，例如`-j16`，加速编译；可通过`-S on`选项，从gitee而不是github下载第三方依赖。

编译完成后，在`build/package/`目录下找到Serving的whl安装包进行安装：

```python
pip install mindspore_serving-{version}-{python_version}-linux_{arch}.whl
```

## 验证是否成功安装

执行以下命令，验证安装结果。导入Python模块不报错即安装成功：

```python
from mindspore_serving import server
```
