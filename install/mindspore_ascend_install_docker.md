# Docker方式安装MindSpore Ascend 910版本

<!-- TOC -->

- [Docker方式安装MindSpore Ascend 910版本](#docker方式安装mindspore-ascend-910版本)
    - [确认系统环境信息](#确认系统环境信息)
    - [安装昇腾AI处理器配套软件包](#安装昇腾ai处理器配套软件包)
    - [获取MindSpore镜像](#获取mindspore镜像)
    - [运行MindSpore镜像](#运行mindspore镜像)
    - [验证是否安装成功](#验证是否安装成功)
    - [升级MindSpore版本](#升级mindspore版本)

<!-- /TOC -->

<a href="https://gitee.com/mindspore/docs/blob/r1.7/install/mindspore_ascend_install_docker.md" target="_blank"><img src="https://mindspore-website.obs.cn-north-4.myhuaweicloud.com/website-images/r1.7/resource/_static/logo_source.png"></a>

[Docker](https://docs.docker.com/get-docker/)是一个开源的应用容器引擎，让开发者打包他们的应用以及依赖包到一个轻量级、可移植的容器中。通过使用Docker，可以实现MindSpore的快速部署，并与系统环境隔离。

本文档介绍如何在Ascend 910环境的Linux系统上，使用Docker方式快速安装MindSpore。

MindSpore的Ascend 910镜像托管在[Ascend Hub](https://ascend.huawei.com/ascendhub/#/main)上。

目前容器化构建选项支持情况如下：

| 硬件平台   | Docker镜像仓库                | 标签                       | 说明                                       |
| :----- | :------------------------ | :----------------------- | :--------------------------------------- |
| Ascend | `public-ascendhub/mindspore-modelzoo` | `x.y.z` | 已经预安装与Ascend Data Center Solution `x.y.z` 版本共同发布的MindSpore的生产环境。 |

> `x.y.z`对应Atlas Data Center Solution版本号，可以在Ascend Hub页面获取。

## 确认系统环境信息

- 确认安装Ubuntu 18.04/CentOS 7.6是64位操作系统。

- 确认安装[Docker 18.03或更高版本](https://docs.docker.com/get-docker/)。

## 安装昇腾AI处理器配套软件包

昇腾软件包提供商用版和社区版两种下载途径：

- 商用版下载需要申请权限，下载链接与安装方式请参考[Ascend Data Center Solution 22.0.RC1安装指引文档](https://support.huawei.com/enterprise/zh/doc/EDOC1100246310)。

- 社区版下载不受限制，下载链接请前往[CANN社区版](https://www.hiascend.com/software/cann/community-history)，选择`5.1.RC1.alpha005`版本，以及在[固件与驱动](https://www.hiascend.com/hardware/firmware-drivers?tag=community)链接中获取对应的固件和驱动安装包，安装包的选择与安装方式请参照上述的商用版安装指引文档。

- 在完成安装基础驱动与配套软件包的基础上，确认安装CANN软件包中的toolbox实用工具包，即Ascend-cann-toolbox-{version}.run，该工具包提供了Ascend NPU容器化支持的Ascend Docker runtime工具。

安装包默认安装路径为`/usr/local/Ascend`。安装后确认当前用户有权限访问昇腾AI处理器配套软件包的安装路径，若无权限，需要root用户将当前用户添加到`/usr/local/Ascend`所在的用户组。

## 获取MindSpore镜像

1. 登录[Ascend Hub镜像中心](https://ascend.huawei.com/ascendhub/#/home)，注册并激活账号，获取登录指令和下载指令。
2. 获取下载权限后，进入[MindSpore镜像下载页面](https://ascendhub.huawei.com/#/detail/mindspore-modelzoo)，获取登录与下载指令并执行：

    ```bash
    docker login -u {username} -p {password} {url}
    docker pull ascendhub.huawei.com/public-ascendhub/mindspore-modelzoo:{tag}
    ```

    其中：

    - `{username}` `{password}` `{url}` 代表用户的登录信息与镜像服务器信息，均为注册并激活账号后自动生成，在对应MindSpore镜像页面复制登录命令即可获取。
    - `{tag}` 对应Atlas Data Center Solution版本号，同样可以在MindSpore镜像下载页面复制下载命令获取。

## 运行MindSpore镜像

执行以下命令启动Docker容器实例：

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
               -v /usr/local/Ascend/latest/driver:/usr/local/Ascend/latest/driver \
               -v /usr/local/Ascend/latest/add-ons/:/usr/local/Ascend/latest/add-ons/ \
               -v /var/log/npu/:/usr/slog \
               ascendhub.huawei.com/public-ascendhub/mindspore-modelzoo:{tag} \
               /bin/bash
```

其中：

- `{tag}`对应Atlas Data Center Solution版本号，在MindSpore镜像下载页面自动获取。

如需使用MindInsight，需设置`--network`参数为”host”模式, 例如:

```bash
docker run -it -u root --ipc=host \
               --network host \
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
               -v /usr/local/Ascend/latest/driver:/usr/local/Ascend/latest/driver \
               -v /usr/local/Ascend/latest/add-ons/:/usr/local/Ascend/latest/add-ons/ \
               -v /var/log/npu/:/usr/slog \
               ascendhub.huawei.com/public-ascendhub/mindspore-modelzoo:{tag} \
               /bin/bash
```

## 验证是否安装成功

按照上述步骤进入MindSpore容器后，测试Docker容器是否正常工作，请运行下面的Python代码并检查输出：

方法一：

```bash
python -c "import mindspore;mindspore.run_check()"
```

如果输出：

```text
MindSpore version: 版本号
The result of multiplication calculation is correct, MindSpore has been installed successfully!
```

至此，你已经成功通过Docker方式安装了MindSpore Ascend 910版本。

方法二：

```python
import numpy as np
import mindspore as ms
import mindspore.ops as ops

ms.set_context(device_target="Ascend")
x = ms.Tensor(np.ones([1,3,3,4]).astype(np.float32))
y = ms.Tensor(np.ones([1,3,3,4]).astype(np.float32))
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

至此，你已经成功通过Docker方式安装了MindSpore Ascend 910版本。

验证MindInsight安装：

输入```mindinsight start --port 8080```, 如提示启动status为success，则安装成功。

## 升级MindSpore版本

当需要升级MindSpore版本时：

- 根据需要升级的MindSpore版本，升级对应的Ascend AI处理器配套软件包。
- 再次登录[Ascend Hub镜像中心](https://ascend.huawei.com/ascendhub/#/home)获取最新docker版本的下载命令，并执行：

    ```bash
    docker pull ascendhub.huawei.com/public-ascendhub/mindspore-modelzoo:{tag}
    ```

    其中：

    - `{tag}`对应Atlas Data Center Solution版本号，同样可以在MindSpore镜像下载页面自动获取。
