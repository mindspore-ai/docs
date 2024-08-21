# 安装

[![查看源文件](https://mindspore-website.obs.cn-north-4.myhuaweicloud.com/website-images/master/resource/_static/logo_source.svg)](https://gitee.com/mindspore/docs/blob/master/docs/mindspore/source_zh_cn/mindformers/quick_start/install.md)

## 版本匹配关系

当前支持的硬件为[Atlas 800T A2](https://www.hiascend.com/hardware/ai-server?tag=900A2)训练服务器。

当前套件建议使用的Python版本为3.9。

| MindFormers |                 MindSpore                  |                                                                                                                                           CANN                                                                                                                                            |                                  驱动固件                                  |                                 镜像链接                                  |
|:-----------:|:------------------------------------------:|:-----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------:|:----------------------------------------------------------------------:|:---------------------------------------------------------------------:|
|   r1.3.0    | [2.3.0](https://www.mindspore.cn/install/) | 8.0.RC2.beta1 <br/> [aarch64](https://ascend-repo.obs.cn-east-2.myhuaweicloud.com/CANN/CANN%208.0.RC2/Ascend-cann-toolkit_8.0.RC2_linux-aarch64.run) <br/> [x86_64](https://ascend-repo.obs.cn-east-2.myhuaweicloud.com/CANN/CANN%208.0.RC2/Ascend-cann-toolkit_8.0.RC2_linux-x86_64.run) | [driver](https://www.hiascend.com/hardware/firmware-drivers/community) | [image](http://mirrors.cn-central-221.ovaijisuan.com/detail/138.html) |

**当前MindFormers仅支持如上的软件配套关系**。

## 安装使用说明

MindFormers提供了灵活的使用选项，以适应不同的使用场景和用户需求，用户可以根据自己的具体需求选择最合适的安装方式：

### 手动安装：手动安装相关依赖

**安装步骤**

（可以参考[Docker方式安装MindSpore](https://www.mindspore.cn/install)，里面有从驱动固件到CANN到MindSpore的安装说明）

1. 安装驱动固件：点击上文版本匹配关系中的驱动固件链接根据需要自行选择社区版或商用版进行下载；
2. 安装CANN：点击上文版本匹配关系中的CANN链接即可下载安装包（CANN和驱动固件的安装都需与使用的机器匹配，请注意识别机器型号，选择对应架构的版本）；
3. 安装MindSpore：点击上文版本匹配关系中的MindSpore链接，获取相应安装命令进行安装；
4. 安装MindFormers：源码编译安装，用户可以执行如下命令安装MindFormers：

    ```shell  
    git clone -b dev https://gitee.com/mindspore/mindformers.git
    cd mindformers
    bash build.sh
    ```

### 使用镜像：在Docker中使用MindFormers

镜像中已包含CANN、MindSpore、MindFormers，无需手动安装各种依赖。

**安装步骤**

1. 安装驱动固件：点击上文版本匹配关系中的驱动固件链接根据需要自行选择社区版或商用版进行下载；
2. 安装镜像：点击上文版本匹配关系中的镜像链接选择镜像进行下载。

## 安装验证

判断MindFormers是否安装成功可以执行以下命令：

```shell
import mindformers as mf
mf.run_check('critical')
```

出现以下类似结果，证明安装成功：

```shell
- INFO - MindFormers version: 1.2.0
- INFO - MindSpore version: 2.3.0
- INFO - Ascend-cann-toolkit version: 8.0.RC2
- INFO - Ascend driver version: 24.1.rc2
- INFO - All checks passed, used **** seconds, the environment is correctly set up!
```