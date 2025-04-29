# 安装

[![查看源文件](https://mindspore-website.obs.cn-north-4.myhuaweicloud.com/website-images/r2.6.0rc1/resource/_static/logo_source.svg)](https://gitee.com/mindspore/docs/blob/r2.6.0rc1/docs/mindformers/docs/source_zh_cn/quick_start/install.md)

## 确认版本匹配关系

当前支持的硬件为[Atlas 800T A2](https://www.hiascend.com/hardware/ai-server?tag=900A2)训练服务器。

当前套件建议使用的Python版本为3.11.4。

| MindSpore Transformers |                   MindSpore                   |  CANN   |  固件与驱动   | 镜像链接 |
|:----------------------:|:---------------------------------------------:|:-------:|:--------:|:----:|
|         1.5.0          | [2.6.0-rc1](https://www.mindspore.cn/install) | 8.1.RC1 | 25.0.RC1 | 即将发布 |

**当前MindSpore Transformers建议使用如上的软件配套关系。**

历史版本配套关系：

| MindSpore Transformers |                 MindSpore                  |                                                     CANN                                                     |                                                    固件与驱动                                                    |                                 镜像链接                                 |
|:----------------------:|:------------------------------------------:|:------------------------------------------------------------------------------------------------------------:|:-----------------------------------------------------------------------------------------------------------:|:--------------------------------------------------------------------:|
|         1.3.2          | [2.4.10](https://www.mindspore.cn/install) |  [8.0.0](https://www.hiascend.com/document/detail/zh/canncommercial/800/softwareinst/instg/instg_0000.html)  | [24.1.0](https://www.hiascend.com/document/detail/zh/canncommercial/800/softwareinst/instg/instg_0000.html) | [Link](http://mirrors.cn-central-221.ovaijisuan.com/detail/168.html) |
|         1.3.0          | [2.4.0](https://www.mindspore.cn/versions) | [8.0.RC3.beta1](https://www.hiascend.com/developer/download/community/result?module=cann&cann=8.0.RC3.beta1) |                  [24.1.RC3](https://www.hiascend.com/hardware/firmware-drivers/community)                   | [Link](http://mirrors.cn-central-221.ovaijisuan.com/detail/154.html) |
|         1.2.0          | [2.3.0](https://www.mindspore.cn/versions) | [8.0.RC2.beta1](https://www.hiascend.com/developer/download/community/result?module=cann&cann=8.0.RC2.beta1) |                  [24.1.RC2](https://www.hiascend.com/hardware/firmware-drivers/community)                   | [Link](http://mirrors.cn-central-221.ovaijisuan.com/detail/138.html) |

## 安装依赖软件

1. 安装固件与驱动：通过[版本匹配关系](https://www.mindspore.cn/mindformers/docs/zh-CN/r1.5.0/quick_start/install.html#%E7%A1%AE%E8%AE%A4%E7%89%88%E6%9C%AC%E5%8C%B9%E9%85%8D%E5%85%B3%E7%B3%BB)中的固件与驱动链接下载安装包，参考[昇腾官方教程](https://www.hiascend.com/document) - 《CANN软件安装》进行安装。

2. 安装CANN和MindSpore：使用官方提供的Docker镜像（镜像中已包含CANN、MindSpore，无需手动安装）或者按照MindSpore官网的[手动安装](https://www.mindspore.cn/install/)章节进行安装。

## 安装MindSpore Transformers

目前仅支持源码编译安装，用户可以执行如下命令安装MindSpore Transformers：

```bash
git clone -b v1.5.0 https://gitee.com/mindspore/mindformers.git
cd mindformers
bash build.sh
```

## 验证是否成功安装

判断MindSpore Transformers是否安装成功可以执行以下代码：

```bash
python -c "import mindformers as mf;mf.run_check()"
```

出现以下类似结果，证明安装成功：

```text
- INFO - All checks passed, used **** seconds, the environment is correctly set up!
```