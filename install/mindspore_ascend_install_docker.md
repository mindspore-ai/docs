# Docker方式安装MindSpore Ascend版本

<!-- TOC -->

- [Docker方式安装MindSpore Ascend版本](#docker方式安装mindspore-ascend版本)
    - [支持的Tags及Dockerfile使用方法](#支持的tags及dockerfile使用方法)
        - [Tag规范](#tag规范)
        - [镜像仓库地址](#镜像仓库地址)
    - [快速开始](#快速开始)
        - [获取MindSpore镜像](#获取mindspore镜像)
        - [运行MindSpore容器](#运行mindspore容器)
        - [构建参数](#构建参数)
        - [构建MindSpore镜像](#构建mindspore镜像)
        - [运行自构建MindSpore容器](#运行自构建mindspore容器)
        - [验证是否安装成功](#验证是否安装成功)
        - [升级MindSpore版本](#升级mindspore版本)
        - [注意事项](#注意事项)
        - [如何二次开发](#如何二次开发)
    - [支持的硬件](#支持的硬件)
    - [许可证](#许可证)

<!-- /TOC -->

[![查看源文件](https://mindspore-website.obs.cn-north-4.myhuaweicloud.com/website-images/r2.10.0/resource/_static/logo_source.svg)](https://atomgit.com/mindspore/docs/blob/r2.10.0/install/mindspore_ascend_install_docker.md)

[Docker](https://docs.docker.com/get-docker/)是一个开源的应用容器引擎，支持将开发者的应用和依赖包打包到一个轻量级、可移植的容器中。通过使用Docker，可以实现MindSpore的快速部署，并与系统环境隔离。

本文档介绍如何在Ascend环境的Linux系统上，使用Docker方式快速安装MindSpore。

## 支持的Tags及Dockerfile使用方法

### Tag规范

Tag 遵循以下格式：

```text
<MindSpore 版本号>-<硬件信息（芯片）>-<操作系统>-<Python 版本>
```

| 字段          | 示例值          | 说明                             |
|-------------|---------------|--------------------------------|
| MindSpore 版本号 | 2.9.0         | 对应 MindSpore 官方发布 Tag 中的版本标识        |
| 硬件信息（芯片）  | 参考下文 | 昇腾芯片型号标识                      |
| 操作系统      | ubuntu22.04 / openeuler24.03 | 基础镜像所使用的操作系统发行版及版本号      |
| Python 版本   | py3.11        | 镜像内置 Python 大版本号                |

> Tips: 系统架构通过 Docker Manifest 自动识别，无需在 Tag 中指定。

### 镜像仓库地址

MindSpore Ascend 镜像托管在华为云 SWR 镜像仓库：

```text
swr.cn-south-1.myhuaweicloud.com/mindspore/mindspore
```

**完整镜像示例：**

```text
swr.cn-south-1.myhuaweicloud.com/mindspore/mindspore:2.9.0-910b-ubuntu22.04-py3.11
```

## 快速开始

### 获取MindSpore镜像

对于不同架构的Ascend硬件平台后端，可以直接使用以下命令获取最新的稳定镜像：

```bash
docker pull swr.cn-south-1.myhuaweicloud.com/mindspore/mindspore:<MindSpore 版本号>-<硬件信息（芯片）>-<操作系统>-<Python 版本>
```

如果需要使用MindSpore 2.9.0版本，Atlas A2训练系列硬件，Ubuntu 22.04操作系统的镜像，使用以下命令：

```bash
docker pull swr.cn-south-1.myhuaweicloud.com/mindspore/mindspore:2.9.0-910b-ubuntu22.04-py3.11
```

如果需要使用MindSpore 2.9.0版本，Atlas A3训练系列硬件，OpenEuler 24.04操作系统的镜像，使用以下命令：

```bash
docker pull swr.cn-south-1.myhuaweicloud.com/mindspore/mindspore:2.9.0-a3-openeuler24.03-py3.11
```

### 运行MindSpore容器

```bash
docker run \
    --privileged \
    --name mindspore_container \
    --device /dev/davinci1 \
    --device /dev/davinci_manager \
    --device /dev/devmm_svm \
    --device /dev/hisi_hdc \
    -v /usr/local/dcmi:/usr/local/dcmi \
    -v /usr/local/bin/npu-smi:/usr/local/bin/npu-smi \
    -v /usr/local/Ascend/driver/lib64/:/usr/local/Ascend/driver/lib64/ \
    -v /usr/local/Ascend/driver/version.info:/usr/local/Ascend/driver/version.info \
    -v /etc/ascend_install.info:/etc/ascend_install.info \
    -it swr.cn-south-1.myhuaweicloud.com/mindspore/mindspore:2.9.0-a3-openeuler24.03-py3.11 bash
```

### 构建参数

| 参数               | 说明                               | 必填 | 参考来源              | 示例值                                                |
|------------------|----------------------------------|----|-------------------|----------------------------------------------------|
| CANN_VERSION     | 昇腾 CANN 工具包版本                    | 是  | CANN 镜像标签         | 9.0.0                                              |
| CHIP_ARCH        | 昇腾芯片架构标识                         | 是  | Tag 规范             | 参考下文                                   |
| OS_SYSTEM        | 基础镜像操作系统及版本                      | 是  | Tag 规范             | ubuntu22.04 / openeuler24.03                      |
| PY_VERSION       | 基础镜像内置 Python 版本                 | 是  | Tag 规范             | py3.11                           |
| MINDSPORE_VERSION | MindSpore 版本号                       | 是  | [MindSpore 仓库发行版](https://atomgit.com/mindspore/mindspore/releases)         | 2.9.0                                              |
| PIP_INDEX_URL    | pip 安装源地址（默认华为云源）                 | 否  | PyPI 镜像源          | https://mirrors.huaweicloud.com/repository/pypi/simple |

### 构建MindSpore镜像

```bash
docker build \
    --build-arg CANN_VERSION=9.0.0 \
    --build-arg CHIP_ARCH=910b \
    --build-arg OS_SYSTEM=ubuntu22.04 \
    --build-arg PY_VERSION=py3.11 \
    --build-arg MINDSPORE_VERSION=2.9.0 \
    --build-arg PIP_INDEX_URL=https://mirrors.huaweicloud.com/repository/pypi/simple \
    -t mindspore:2.9.0-910b-ubuntu22.04-py3.11 \
    -f Dockerfile .
```

### 运行自构建MindSpore容器

```bash
docker run \
    --privileged \
    --name mindspore_container \
    --device /dev/davinci1 \
    --device /dev/davinci_manager \
    --device /dev/devmm_svm \
    --device /dev/hisi_hdc \
    -v /usr/local/dcmi:/usr/local/dcmi \
    -v /usr/local/bin/npu-smi:/usr/local/bin/npu-smi \
    -v /usr/local/Ascend/driver/lib64/:/usr/local/Ascend/driver/lib64/ \
    -v /usr/local/Ascend/driver/version.info:/usr/local/Ascend/driver/version.info \
    -v /etc/ascend_install.info:/etc/ascend_install.info \
    -it mindspore:{tag} bash
```

其中：

- `{tag}`对应上述构建MindSpore镜像操作中指定的标签，例如`mindspore:2.9.0-910b-ubuntu22.04-py3.11`。

### 验证是否安装成功

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

### 升级MindSpore版本

当需要升级MindSpore版本时：

- 根据需要升级的MindSpore版本以及Ascend硬件平台，升级对应的Ascend AI处理器配套软件包。
- 直接使用以下命令获取最新的稳定镜像：

    ```bash
    docker pull swr.cn-south-1.myhuaweicloud.com/mindspore/mindspore:<MindSpore 版本号>-<硬件信息（芯片）>-<操作系统>-<Python 版本>
    ```

### 注意事项

- 在非root用户模式下创建容器时，必须确保目标NPU设备未被其他非root容器占用。启动后可以执行 `npu-smi info` 命令验证设备状态，若目标NPU设备已被其他非root容器占用，则会出现以下报错，可以在创建容器时加上 `-u root --privileged`。

```text
    DrvMngGetConsoleLogLevel failed. (g_conLogLevel=3)
    dcmi model initialized failed, because the device is used. ret is -802
```

### 如何二次开发

```bash
# 以 MindSpore 镜像为基础镜像，叠加用户软件
FROM swr.cn-south-1.myhuaweicloud.com/mindspore/mindspore:2.9.0-910b-ubuntu22.04-py3.11

RUN apt update -y && \
    apt install -y gcc g++

# 安装额外依赖
RUN pip install pandas scikit-learn

# 复制用户代码
WORKDIR /workspace
COPY . /workspace

CMD ["python", "train.py"]
```

## 支持的硬件

| 芯片系列 | 产品示例 | 架构 |
|---|---|---|
| 昇腾 A2 | Atlas 800T A2、Atlas 900 A2 PoD | 自动识别 (ARM64/x86_64) |
| 昇腾 A3 | Atlas 800T A3 | 自动识别 (ARM64/x86_64) |

> Tips: 使用 `docker manifest inspect` 命令可以查看镜像支持的系统架构。

## 许可证

查看这些镜像中包含的 MindSpore 的[许可证信息](https://atomgit.com/mindspore/mindspore/blob/v2.10/LICENSE)。

与所有容器镜像一样，预装软件包（Python、系统库等）可能受其自身许可证约束。
