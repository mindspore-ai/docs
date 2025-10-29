# 安装指南

[![查看源文件](https://mindspore-website.obs.cn-north-4.myhuaweicloud.com/website-images/r2.7.1/resource/_static/logo_source.svg)](https://gitee.com/mindspore/docs/blob/r2.7.1/docs/mindformers/docs/source_zh_cn/installation.md)

## 确认版本匹配关系

当前支持的硬件为Atlas 800T A2、Atlas 800I A2、Atlas 900 A3 SuperPoD。

当前套件建议使用的Python版本为3.11.4。

| MindSpore Transformers |                 MindSpore                 |                                                      CANN                                                      |                                                      固件与驱动                                                      |
|:----------------------:|:-----------------------------------------:|:--------------------------------------------------------------------------------------------------------------:|:---------------------------------------------------------------------------------------------------------------:|
|         1.7.0          | [2.7.1](https://www.mindspore.cn/install) | [8.3.RC1](https://www.hiascend.com/document/detail/zh/canncommercial/83RC1/softwareinst/instg/instg_0000.html) | [25.3.RC1](https://www.hiascend.com/document/detail/zh/canncommercial/83RC1/softwareinst/instg/instg_0000.html) |

**当前MindSpore Transformers建议使用如上的软件配套关系。**

历史版本配套关系：

| MindSpore Transformers |                   MindSpore                   |                                                      CANN                                                      |                                                      固件与驱动                                                      |
|:----------------------:|:---------------------------------------------:|:--------------------------------------------------------------------------------------------------------------:|:---------------------------------------------------------------------------------------------------------------:|
|         1.6.0          |   [2.7.0](https://www.mindspore.cn/install)   | [8.2.RC1](https://www.hiascend.com/document/detail/zh/canncommercial/82RC1/softwareinst/instg/instg_0000.html) |  [25.2.0](https://www.hiascend.com/document/detail/zh/canncommercial/82RC1/softwareinst/instg/instg_0000.html)  |
|         1.5.0          | [2.6.0-rc1](https://www.mindspore.cn/install) | [8.1.RC1](https://www.hiascend.com/document/detail/zh/canncommercial/81RC1/softwareinst/instg/instg_0000.html) | [25.0.RC1](https://www.hiascend.com/document/detail/zh/canncommercial/81RC1/softwareinst/instg/instg_0000.html) |
|         1.3.2          |  [2.4.10](https://www.mindspore.cn/versions)  |   [8.0.0](https://www.hiascend.com/document/detail/zh/canncommercial/800/softwareinst/instg/instg_0000.html)   |   [24.1.0](https://www.hiascend.com/document/detail/zh/canncommercial/800/softwareinst/instg/instg_0000.html)   |
|         1.3.0          |  [2.4.0](https://www.mindspore.cn/versions)   | [8.0.RC3](https://www.hiascend.com/document/detail/zh/canncommercial/80RC3/softwareinst/instg/instg_0000.html) | [24.1.RC3](https://www.hiascend.com/document/detail/zh/canncommercial/80RC3/softwareinst/instg/instg_0000.html) |
|         1.2.0          |  [2.3.0](https://www.mindspore.cn/versions)   | [8.0.RC2](https://www.hiascend.com/document/detail/zh/canncommercial/80RC2/softwareinst/instg/instg_0000.html) | [24.1.RC2](https://www.hiascend.com/document/detail/zh/canncommercial/80RC2/softwareinst/instg/instg_0000.html) |

## 安装依赖软件

1. 安装固件与驱动：通过[版本匹配关系](https://www.mindspore.cn/mindformers/docs/zh-CN/r1.7.0/installation.html#%E7%A1%AE%E8%AE%A4%E7%89%88%E6%9C%AC%E5%8C%B9%E9%85%8D%E5%85%B3%E7%B3%BB)中的固件与驱动链接下载安装包，参考[昇腾官方教程](https://www.hiascend.com/document/detail/zh/canncommercial/81RC1/softwareinst/instg/instg_0000.html?Mode=PmIns&InstallType=local&OS=Ubuntu&Software=cannToolKit)进行安装。

2. 安装CANN和MindSpore：按照MindSpore官网的[手动安装](https://www.mindspore.cn/install/)章节进行安装。

## 安装MindSpore Transformers

MindSpore Transformers支持源码编译安装和pip安装两种方式。

### 源码编译方式安装

用户可以执行如下命令编译并安装MindSpore Transformers：

```bash
git clone -b r1.7.0 https://gitee.com/mindspore/mindformers.git
cd mindformers
bash build.sh
```

### pip方式安装

用户可以执行如下命令安装MindSpore Transformers：

```bash
pip install https://ms-release.obs.cn-north-4.myhuaweicloud.com/2.7.1/MindFormers/any/mindformers-1.7.0-py3-none-any.whl --trusted-host ms-release.obs.cn-north-4.myhuaweicloud.com -i https://repo.huaweicloud.com/repository/pypi/simple
```

> - 此安装方式需要访问公网，如果您处于内网环境，请确保网络连接配置正确。
> - 此方式只安装了MindSpore Transformers基础软件包，模型文件和脚本等请从MindSpore Transformers gitee仓库中获取。

## 验证是否成功安装

要验证MindSpore Transformers是否安装成功，可以执行以下代码：

```bash
python -c "import mindformers as mf;mf.run_check()"
```

出现以下类似结果，说明安装成功：

```text
- INFO - All checks passed, used **** seconds, the environment is correctly set up!
```
