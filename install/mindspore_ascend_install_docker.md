# Docker方式安装MindSpore Ascend版本

<!-- TOC -->

- [Docker方式安装MindSpore Ascend版本](#docker方式安装mindspore-ascend版本)
    - [确认系统环境信息](#确认系统环境信息)
    - [安装昇腾AI处理器配套软件包](#安装昇腾ai处理器配套软件包)
    - [获取MindSpore镜像](#获取mindspore镜像)
    - [运行MindSpore镜像](#运行mindspore镜像)
    - [验证是否安装成功](#验证是否安装成功)
    - [升级MindSpore版本](#升级mindspore版本)
    - [注意事项](#注意事项)

<!-- /TOC -->

[![查看源文件](https://mindspore-website.obs.cn-north-4.myhuaweicloud.com/website-images/r2.7.0/resource/_static/logo_source.svg)](https://gitee.com/mindspore/docs/blob/r2.7.0/install/mindspore_ascend_install_docker.md)

[Docker](https://docs.docker.com/get-docker/)是一个开源的应用容器引擎，支持将开发者的应用和依赖包打包到一个轻量级、可移植的容器中。通过使用Docker，可以实现MindSpore的快速部署，并与系统环境隔离。

本文档介绍如何在Ascend环境的Linux系统上，使用Docker方式快速安装MindSpore。

MindSpore的Docker镜像托管在[Huawei SWR](https://support.huaweicloud.com/swr/index.html)上。

目前容器化构建选项支持情况如下：

| 硬件平台   | Docker命名空间   | Docker镜像名称             | 标签                       | 说明                                       |
| :--------- | :------------------------ | :------------------------ | :----------------------- | :--------------------------------------- |
| Atlas 训练系列  | `mindspore` | `mindspore-ascend-a1` | `x.y.z` | 已经预安装Ascend Data Center Solution 与对应的MindSpore Ascend x.y.z版本的生产环境。 |
| Atlas A2 训练系列 | `mindspore` | `mindspore-ascend-a2` | `x.y.z` | 已经预安装Ascend Data Center Solution 与对应的MindSpore Ascend x.y.z版本的生产环境。 |

> `x.y.z`对应MindSpore版本号，例如安装2.7.0版本MindSpore时，`x.y.z`应写为2.7.0。

## 确认系统环境信息

下表列出了使用Docker方式快速安装MindSpore所需的系统环境。

|软件名称|版本|作用|
|-|-|-|
|Debian系列操作系统 / openEuler系列操作系统|Debian系列：Debian、Ubuntu、veLinux / openEuler系列：openEuler、CentOS、Kylin、BCLinux、UOS V20、AntOS、CTyunOS、CULinux、Tlinux、MTOS|运行MindSporer容器的操作系统|
|[昇腾AI处理器配套软件包](#安装昇腾ai处理器配套软件包)|-|MindSpore使用的Ascend平台AI计算库|
|Docker | Docker 18.03或更高版本 |提供轻量级容器化环境，实现MindSpore及其依赖的隔离部署与跨平台运行|

## 安装昇腾AI处理器配套软件包

昇腾软件包社区版下载链接请前往[CANN社区版](https://www.hiascend.com/developer/download/community/result?module=cann)，推荐优先选择`8.2.RC1`版本，以及在[固件与驱动](https://www.hiascend.com/hardware/firmware-drivers/community)链接中获取对应的固件和驱动安装包，安装包的选择与安装方式请参照[安装指引文档](https://www.hiascend.com/document/detail/zh/CANNCommunityEdition/82RC1/softwareinst/instg/instg_quick.html)。

安装包默认安装路径为`/usr/local/Ascend`。安装后确认当前用户有权限访问昇腾AI处理器配套软件包的安装路径，若无权限，需要root用户将当前用户添加到`/usr/local/Ascend`所在的用户组。

## 获取MindSpore镜像

对于不同架构的Ascend硬件平台后端，可以直接使用以下命令获取最新的稳定镜像：

```bash
docker pull swr.cn-south-1.myhuaweicloud.com/mindspore/{image_name}:{tag}
```

其中：

- `{image_name}` 对应上述表格中的docker镜像名称，使用 Atlas 训练系列产品请下载 `mindspore-ascend-a1` 镜像；Atlas A2 训练系列产品请下载 `mindspore-ascend-a2` 镜像。
- `{tag}`对应上述表格中的标签,如2.7.0。

如果需要使用MindSpore 2.7.0版本，Atlas训练系列硬件的镜像，使用以下命令：

```bash
docker pull swr.cn-south-1.myhuaweicloud.com/mindspore/2.7.0:mindspore-ascend-a1
```

如果需要使用MindSpore 2.7.0版本，Atlas A2训练系列硬件的镜像，使用以下命令：

```bash
docker pull swr.cn-south-1.myhuaweicloud.com/mindspore/2.7.0:mindspore-ascend-a2
```

## 运行MindSpore镜像

执行以下命令，启动Docker容器实例：

```bash
docker run -it --ipc=host \
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
               -v /usr/local/bin/npu-smi:/usr/local/bin/npu-smi \
               -v /usr/local/Ascend/driver:/usr/local/Ascend/driver \
               -v /etc/ascend_install.info:/etc/ascend_install.info \
               -v /var/log/npu/:/usr/slog \
               -v /usr/bin/hccn_tool:/usr/bin/hccn_tool \
               -v /etc/hccn.conf:/etc/hccn.conf \
               swr.cn-south-1.myhuaweicloud.com/mindspore/{image_name}:{tag} \
               /bin/bash
```

其中：

- `{tag}`对应上述表格中的标签，如2.7.0。

## 验证是否安装成功

按照上述步骤进入MindSpore容器后，测试Docker容器是否正常工作，请执行下面的Python代码并检查输出：

**方法一：**

执行以下命令：

```bash
python -c "import mindspore;mindspore.set_device('Ascend');mindspore.run_check()"
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

- 根据需要升级的MindSpore版本以及Ascend硬件平台，升级对应的Ascend AI处理器配套软件包。
- 直接使用以下命令获取最新的稳定镜像：

    ```bash
    docker pull swr.cn-south-1.myhuaweicloud.com/mindspore/{image_name}:{tag}
    ```

    其中：

    - `{tag}`对应上述表格中的标签。
    - `{image_name}` 对应上述表格中的docker镜像名称，使用 Atlas 训练系列产品请下载 `mindspore-ascend-a1` 镜像；Atlas A2 训练系列产品请下载 `mindspore-ascend-a2` 镜像。

## 注意事项

- 在非root用户模式下创建容器时，必须确保目标NPU设备未被其他非root容器占用。启动后可以执行 `npu-smi info` 命令验证设备状态，若目标NPU设备已被其他非root容器占用，则会出现以下报错，可以在创建容器时加上 `-u root --privileged`。

```text
    DrvMngGetConsoleLogLevel failed. (g_conLogLevel=3)
    dcmi model initialized failed, because the device is used. ret is -802
```