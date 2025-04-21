# Docker方式安装MindSpore Ascend版本

<!-- TOC -->

- [Docker方式安装MindSpore Ascend版本](#docker方式安装mindspore-ascend版本)
    - [确认系统环境信息](#确认系统环境信息)
    - [安装昇腾AI处理器配套软件包](#安装昇腾ai处理器配套软件包)
    - [获取MindSpore镜像](#获取mindspore镜像)
    - [运行MindSpore镜像](#运行mindspore镜像)
    - [验证是否安装成功](#验证是否安装成功)
    - [升级MindSpore版本](#升级mindspore版本)

<!-- /TOC -->

[![查看源文件](https://mindspore-website.obs.cn-north-4.myhuaweicloud.com/website-images/br_base/resource/_static/logo_source.svg)](https://gitee.com/mindspore/docs/blob/br_base/install/mindspore_ascend_install_docker.md)

[Docker](https://docs.docker.com/get-docker/)是一个开源的应用容器引擎，支持将开发者的应用和依赖包打包到一个轻量级、可移植的容器中。通过使用Docker，可以实现MindSpore的快速部署，并与系统环境隔离。

本文档介绍如何在Ascend环境的Linux系统上，使用Docker方式快速安装MindSpore。

MindSpore的Docker镜像托管在[Huawei SWR](https://support.huaweicloud.com/swr/index.html)上。

目前容器化构建选项支持情况如下：

| 硬件平台   | Docker镜像仓库                | 标签                       | 说明                                       |
| :----- | :------------------------ | :----------------------- | :--------------------------------------- |
| Ascend | `mindspore/mindspore-ascend` | `x.y.z` | 已经预安装Ascend Data Center Solution 与对应的MindSpore Ascend x.y.z版本的生产环境。 |

> `x.y.z`对应MindSpore版本号，例如安装2.5.0版本MindSpore时，`x.y.z`应写为2.5.0。

## 确认系统环境信息

- 确认安装基于ARM的Ubuntu 18.04 / CentOS 7.6 64位操作系统。

- 确认安装[Docker 18.03或更高版本](https://docs.docker.com/get-docker/)。

## 安装昇腾AI处理器配套软件包

昇腾软件包提供商用版和社区版两种下载途径：

- 商用版下载需要申请权限，下载链接与安装方式请参考[Ascend Training Solution 24.0.0 安装指引文档](https://support.huawei.com/enterprise/zh/doc/EDOC1100441839)。

- 社区版下载不受限制，下载链接请前往[CANN社区版](https://www.hiascend.com/developer/download/community/result?module=cann)，选择`8.0.RC3.beta1`版本，还需在[固件与驱动](https://www.hiascend.com/hardware/firmware-drivers/community)链接中获取对应的固件和驱动安装包，安装包的选择与安装方式请参照上述的商用版安装指引文档。

安装包默认安装路径为`/usr/local/Ascend`。安装后确认当前用户有权限访问昇腾AI处理器配套软件包的安装路径，若无权限，需要root用户将当前用户添加到`/usr/local/Ascend`所在的用户组。

## 获取MindSpore镜像

对于`Ascend`后端，可以直接使用以下命令获取最新的稳定镜像：

```bash
docker pull swr.cn-south-1.myhuaweicloud.com/mindspore/mindspore-ascend:{tag}
```

其中：

- `{tag}`对应上述表格中的标签。

## 运行MindSpore镜像

执行以下命令，启动Docker容器实例：

```bash
docker run -it -u root --ipc=host \
               --device=/dev/davinci0 \
               --device=/dev/davinci1 \
               --device=/dev/davinci2 \
               --device=/dev/davinci3 \
               --device=/dev/davinci4 \
               --device=/dev/davinci5 \
               --device=/dev/davinci6 \
               --device=/dev/davinci7 \
               --device=/dev/davinci_manager \
               --device=/dev/devmm_svm \
               --device=/dev/hisi_hdc \
               -v /usr/local/Ascend/driver:/usr/local/Ascend/driver \
               -v /var/log/npu/:/usr/slog \
               swr.cn-south-1.myhuaweicloud.com/mindspore/mindspore-ascend:{tag} \
               /bin/bash
```

其中：

- `{tag}`对应上述表格中的标签。

## 验证是否安装成功

按照上述步骤进入MindSpore容器后，测试Docker容器是否正常工作，请执行下面的Python代码并检查输出：

**方法一：**

执行以下命令：

```bash
python -c "import mindspore;mindspore.run_check()"
```

如果输出：

```text
MindSpore version: 版本号
The result of multiplication calculation is correct, MindSpore has been installed on platform [Ascend] successfully!
```

至此，你已经成功通过Docker方式安装了MindSpore Ascend版本。

**方法二：**

执行以下代码：

```python
import numpy as np
import mindspore as ms
import mindspore.ops as ops

ms.set_device("Ascend")
x = ms.Tensor(np.ones([1,3,3,4]).astype(np.float32))
y = ms.Tensor(np.ones([1,3,3,4]).astype(np.float32))
print(ops.add(x, y))
```

代码成功执行时会输出：

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

至此，你已经成功通过Docker方式安装了MindSpore Ascend版本。

## 升级MindSpore版本

当需要升级MindSpore版本时：

- 根据需要升级的MindSpore版本，升级对应的Ascend AI处理器配套软件包。
- 直接使用以下命令获取最新的稳定镜像：

    ```bash
    docker pull swr.cn-south-1.myhuaweicloud.com/mindspore/mindspore-ascend:{tag}
    ```

    其中：

    - `{tag}`对应上述表格中的标签。
