# 安装

[![查看源文件](https://mindspore-website.obs.cn-north-4.myhuaweicloud.com/website-images/master/resource/_static/logo_source.svg)](https://gitee.com/mindspore/docs/blob/master/docs/mindformers/docs/source_zh_cn/quick_start/install.md)

## 确认版本匹配关系

当前支持的硬件为[Atlas 800T A2](https://www.hiascend.com/hardware/ai-server?tag=900A2)训练服务器。

当前套件建议使用的Python版本为3.10。

| MindFormers | MindSpore | CANN | 固件与驱动 | 镜像链接 |
|:-----------:|:---------:|:----:|:-----:|:----:|
|    在研版本     |   在研版本    | 在研版本 | 在研版本  | 不涉及  |

**当前MindFormers建议使用如上的软件配套关系。**

历史版本配套关系：

|                     MindFormers                      |                 MindSpore                  |                                                     CANN                                                     |                                  固件与驱动                                   |                                 镜像链接                                 |
|:----------------------------------------------------:|:------------------------------------------:|:------------------------------------------------------------------------------------------------------------:|:------------------------------------------------------------------------:|:--------------------------------------------------------------------:|
| [1.3.0](https://pypi.org/project/mindformers/1.3.0/) | [2.4.0](https://www.mindspore.cn/install/) | [8.0.RC3.beta1](https://www.hiascend.com/developer/download/community/result?module=cann&cann=8.0.RC3.beta1) | [24.1.RC3](https://www.hiascend.com/hardware/firmware-drivers/community) | [Link](http://mirrors.cn-central-221.ovaijisuan.com/detail/154.html) |
| [1.2.0](https://pypi.org/project/mindformers/1.2.0/) | [2.3.0](https://www.mindspore.cn/install/) | [8.0.RC2.beta1](https://www.hiascend.com/developer/download/community/result?module=cann&cann=8.0.RC2.beta1) | [24.1.RC2](https://www.hiascend.com/hardware/firmware-drivers/community) | [Link](http://mirrors.cn-central-221.ovaijisuan.com/detail/138.html) |

## 安装依赖软件

1. 安装固件与驱动：通过[版本匹配关系](https://www.mindspore.cn/mindformers/docs/zh-CN/dev/quick_start/install.html#%E7%89%88%E6%9C%AC%E5%8C%B9%E9%85%8D%E5%85%B3%E7%B3%BB)中的固件与驱动链接下载安装包，参考[昇腾官方教程](https://www.hiascend.com/document/detail/zh/quick-installation/24.0.RC1/quickinstg_train/800_9000A2/quickinstg_800_9000A2_0007.html)进行安装。

2. 安装CANN和MindSpore：使用官方提供的Docker镜像（镜像中已包含CANN、MindSpore，无需手动安装）或者按照MindSpore官网的[手动安装](https://www.mindspore.cn/install/)章节进行安装。

## 安装MindFormers

目前在研版本仅支持源码编译安装，用户可以执行如下命令安装MindFormers：

```bash
git clone -b dev https://gitee.com/mindspore/mindformers.git
cd mindformers
bash build.sh
```

## 验证是否成功安装

判断MindFormers是否安装成功可以执行以下代码：

```python
import mindformers as mf
mf.run_check()
```

出现以下类似结果，证明安装成功：

```text
- INFO - All checks passed, used **** seconds, the environment is correctly set up!
```