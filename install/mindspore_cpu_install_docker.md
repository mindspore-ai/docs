# Docker方式安装MindSpore CPU版本

<!-- TOC -->

- [Docker方式安装MindSpore CPU版本](#docker方式安装mindspore-cpu版本)
    - [确认系统环境信息](#确认系统环境信息)
    - [获取MindSpore镜像](#获取mindspore镜像)
    - [运行MindSpore镜像](#运行mindspore镜像)
    - [验证是否安装成功](#验证是否安装成功)

<!-- /TOC -->

<a href="https://gitee.com/mindspore/docs/blob/r1.6/install/mindspore_cpu_install_docker.md" target="_blank"><img src="https://gitee.com/mindspore/docs/raw/r1.6/resource/_static/logo_source.png"></a>

[Docker](https://docs.docker.com/get-docker/)是一个开源的应用容器引擎，让开发者打包他们的应用以及依赖包到一个轻量级、可移植的容器中。通过使用Docker，可以实现MindSpore的快速部署，并与系统环境隔离。

本文档介绍如何在CPU环境的Linux系统上，使用Docker方式快速安装MindSpore。

MindSpore的Docker镜像托管在[Huawei SWR](https://support.huaweicloud.com/swr/index.html)上。

目前容器化构建选项支持情况如下：

| 硬件平台   | Docker镜像仓库                | 标签                       | 说明                                       |
| :----- | :------------------------ | :----------------------- | :--------------------------------------- |
| CPU    | `mindspore/mindspore-cpu` | `x.y.z`                  | 已经预安装MindSpore `x.y.z` CPU版本的生产环境。       |
|        |                           | `devel`                  | 提供开发环境从源头构建MindSpore（`CPU`后端）。安装详情请参考<https://www.mindspore.cn/install> 。 |
|        |                           | `runtime`                | 提供运行时环境，未安装MindSpore二进制包（`CPU`后端）。         |

> `x.y.z`对应MindSpore版本号，例如安装1.1.0版本MindSpore时，`x.y.z`应写为1.1.0。

## 确认系统环境信息

- 确认安装基于x86架构的64位Linux操作系统，其中Ubuntu 18.04是经过验证的。
- 确认安装[Docker 18.03或者更高版本](https://docs.docker.com/get-docker/)。

## 获取MindSpore镜像

对于`CPU`后端，可以直接使用以下命令获取最新的稳定镜像：

```bash
docker pull swr.cn-south-1.myhuaweicloud.com/mindspore/mindspore-cpu:{tag}
```

其中：

- `{tag}`对应上述表格中的标签。

## 运行MindSpore镜像

执行以下命令启动Docker容器实例：

```bash
docker run -it swr.cn-south-1.myhuaweicloud.com/mindspore/mindspore-cpu:{tag} /bin/bash
```

其中：

- `{tag}`对应上述表格中的标签。

如需使用可视化调试调优工具MindInsight，需设置`--network`参数为`host`模式，例如:

```bash
docker run -it --network host swr.cn-south-1.myhuaweicloud.com/mindspore/mindspore-cpu:{tag} /bin/bash
```

## 验证是否安装成功

- 如果你安装的是指定版本`x.y.z`的容器。

按照上述步骤进入MindSpore容器后，测试Docker是否正常工作，请运行下面的Python代码并检查输出：

方法一：

```bash
python -c "import mindspore;mindspore.run_check()"
```

如果输出：

```text
MindSpore version: 版本号
The result of multiplication calculation is correct, MindSpore has been installed successfully!
```

至此，你已经成功通过Docker方式安装了MindSpore CPU版本。

方法二：

```python
import numpy as np
import mindspore.context as context
import mindspore.ops as ops
from mindspore import Tensor

context.set_context(mode=context.PYNATIVE_MODE, device_target="CPU")

x = Tensor(np.ones([1,3,3,4]).astype(np.float32))
y = Tensor(np.ones([1,3,3,4]).astype(np.float32))
print(ops.add(x, y))
```

代码成功运行时会输出：

```text
[[[[2. 2. 2. 2.]
   [2. 2. 2. 2.]
   [2. 2. 2. 2.]]

  [[2. 2. 2. 2.]
   [2. 2. 2. 2.]
   [2. 2. 2. 2.]]

  [[2. 2. 2. 2.]
   [2. 2. 2. 2.]
   [2. 2. 2. 2.]]]]
```

至此，你已经成功通过Docker方式安装了MindSpore CPU版本。

- 验证MindInsight安装：

    输入```mindinsight start --port 8080```，如提示启动status为success，则安装成功。

- 如果你安装的是`runtime`标签的容器，需要自行安装MindSpore。

    进入[MindSpore安装指南页面](https://www.mindspore.cn/install)，选择CPU硬件平台、Linux-x86_64操作系统和pip的安装方式，获得安装指南。运行容器后参考安装指南，通过pip方式安装MindSpore CPU版本，并进行验证。

- 如果你安装的是`devel`标签的容器，需要自行编译并安装MindSpore。

    进入[MindSpore安装指南页面](https://www.mindspore.cn/install)，选择CPU硬件平台、Linux-x86_64操作系统和Source的安装方式，获得安装指南。运行容器后，下载MindSpore代码仓并参考安装指南，通过源码编译方式安装MindSpore CPU版本，并进行验证。

如果您想了解更多关于MindSpore Docker镜像的构建过程，请查看[docker repo](https://gitee.com/mindspore/mindspore/blob/r1.6/scripts/docker/README.md)了解详细信息。
