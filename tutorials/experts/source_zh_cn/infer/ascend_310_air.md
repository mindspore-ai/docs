# Ascend 310 AI处理器上使用AIR模型进行推理

`Ascend` `推理应用`

<a href="https://gitee.com/mindspore/docs/blob/master/tutorials/experts/source_zh_cn/infer/ascend_310_air.md" target="_blank"><img src="https://mindspore-website.obs.cn-north-4.myhuaweicloud.com/website-images/master/resource/_static/logo_source.png"></a>

## 概述

Ascend 310是面向边缘场景的高能效高集成度AI处理器。Atlas 200开发者套件又称Atlas 200 Developer Kit（以下简称Atlas 200 DK），是以Atlas 200 AI加速模块为核心的开发者板形态的终端类产品，集成了海思Ascend 310 AI处理器，可以实现图像、视频等多种数据分析与推理计算，可广泛用于智能监控、机器人、无人机、视频服务器等场景。

本教程介绍如何在Atlas 200 DK上使用MindSpore基于AIR模型文件执行推理，主要包括以下流程：

1. 开发环境准备，包括制作Atlas 200 DK的SD卡 、配置Python环境和刷配套开发软件包。

2. 导出AIR模型文件，这里以ResNet-50模型为例。

3. 使用ATC工具将AIR模型文件转成OM模型。

4. 编译推理代码，生成可执行`main`文件。

5. 加载保存的OM模型，执行推理并查看结果。

> 你可以在这里找到完整可运行的样例代码：<https://gitee.com/mindspore/docs/tree/master/docs/sample_code/acl_resnet50_sample> 。

## 开发环境准备

### 硬件准备

- 一个操作系统为Ubuntu的服务器或PC机，用于为Atlas 200 DK制作SD卡启动盘和开发环境部署。
- 一张SD卡，建议容量不低于16G。

### 软件包准备

配置开发环境需要的脚本和软件包如下5类，共7个文件。

1. 制卡入口脚本：[make_sd_card.py](https://gitee.com/ascend/tools/blob/master/makesd/for_1.0.9.alpha/make_sd_card.py)

2. 制作SD卡操作系统脚本：[make_ubuntu_sd.sh](https://gitee.com/ascend/tools/blob/master/makesd/for_1.0.9.alpha/make_ubuntu_sd.sh)

3. Ubuntu操作系统镜像包：[ubuntu-18.04.xx-server-arm64.iso](http://cdimage.ubuntu.com/ubuntu/releases/18.04/release/ubuntu-18.04.6-server-arm64.iso)

4. 开发者板驱动包与运行包：

    - `Ascend310-driver-*{software version}*-ubuntu18.04.aarch64-minirc.tar.gz`

    - `Ascend310-aicpu_kernels-*{software version}*-minirc.tar.gz`

    - `Ascend-acllib-*{software version}*-ubuntu18.04.aarch64-minirc.run`

5. 安装开发套件包：`Ascend-Toolkit-*{version}*-arm64-linux_gcc7.3.0.run`

其中，

- 前3项可以参考[Atlas 200 DK 开发者套件使用指南](https://support.huaweicloud.com/usermanual-A200dk_3000/atlas200dk_02_0011.html)获取。
- 其余软件包建议从[固件与驱动](https://ascend.huawei.com/#/hardware/firmware-drivers)中获取，在该页面中选择产品系列和产品型号为`Atlas 200 DK`，选中需要的文件，即可下载。

### 制作SD卡

读卡器通过USB与Ubuntu服务器连接，通过制卡脚本制作SD卡。具体操作参见[操作步骤](https://support.huaweicloud.com/usermanual-A200dk_3000/atlas200dk_02_0011.html#section2)。

### 连接Atlas 200 DK开发板与Ubuntu服务器

Atlas 200 DK开发者板支持通过USB端口或者网线与Ubuntu服务器进行连接。具体操作参见[连接Atlas 200 DK开发者板与Ubuntu服务器](https://support.huaweicloud.com/usermanual-A200dk_3000/atlas200dk_02_0013.html)。

### 配置Python环境

安装Python以及gcc等软件，具体操作参见[安装依赖](https://support.huaweicloud.com/usermanual-A200dk_3000/atlas200dk_02_0016.html#section4)。

### 安装开发套件包

安装开发套件包`Ascend-Toolkit-*{version}*-arm64-linux_gcc7.3.0.run`，具体操作参见[安装开发套件包](https://support.huaweicloud.com/usermanual-A200dk_3000/atlas200dk_02_0017.html)。

## 推理目录结构介绍

创建目录放置推理代码工程，例如`/home/HwHiAiUser/Ascend/ascend-toolkit/20.0.RC1/acllib_linux.arm64/sample/acl_execute_model/acl_resnet50_sample`，其中`inc`、`src`、`test_data`可以从官网示例下载[样例代码](https://gitee.com/mindspore/docs/tree/master/docs/sample_code/acl_resnet50_sample)获取，`model`目录用于存放接下来导出的`AIR`模型文件和转换后的`OM`模型文件，`out`目录用于存放执行编译生成的可执行文件和输出结果目录，推理代码工程目录结构如下:

```text
└─acl_resnet50_sample
    ├── inc
    │   ├── model_process.h                   //声明资源初始化/销毁相关函数的头文件
    │   ├── sample_process.h                  //声明模型处理相关函数的头文件
    │   ├── utils.h                           //声明公共函数（例如：文件读取函数）的头文件
    ├── model
    │   ├── resnet50_export.air               //AIR模型文件
    │   ├── resnet50_export.om                //转换后的OM模型文件
    ├── src
    │   ├── acl.json                          //系统初始化的配置文件
    │   ├── CMakeLists.txt                    //编译脚本
    │   ├── main.cpp                          //主函数，图片分类功能的实现文件
    │   ├── model_process.cpp                 //模型处理相关函数的实现文件
    │   ├── sample_process.cpp                //资源初始化/销毁相关函数的实现文件
    │   ├── utils.cpp                         //公共函数（例如：文件读取函数）的实现文件
    ├── test_data
    │   ├── test_data_1x3x224x224_1.bin       //输入样本数据1
    │   ├── test_data_1x3x224x224_2.bin       //输入样本数据2
    ├── out
    │   ├── main                              //编译生成的可执行文件
    │   ├── result                            //输出结果目录
```

> 输出结果目录`acl_resnet50_sample/out/result`需先创建好再执行推理操作。

## 导出AIR模型文件

在Ascend 910的机器上训练好目标网络，并保存为CheckPoint文件，通过网络和CheckPoint文件导出对应的AIR格式模型文件，导出流程参见[导出AIR格式文件](https://www.mindspore.cn/tutorials/experts/zh-CN/master/parallel/save_load.html#air)。

> 这里提供使用ResNet-50模型导出的示例AIR文件[resnet50_export.air](https://mindspore-website.obs.cn-north-4.myhuaweicloud.com:443/sample_resources/acl_resnet50_sample/resnet50_export.air)。

## 将AIR模型文件转成OM模型

登录Atlas 200 DK开发者板环境，创建`model`目录放置AIR文件`resnet50_export.air`，例如`/home/HwHiAiUser/Ascend/ascend-toolkit/20.0.RC1/acllib_linux.arm64/sample/acl_execute_model/acl_resnet50_sample/model`，并进入该路径下，设置如下环境变量。其中，`install_path`需指定为实际安装路径。

```bash
export install_path=/home/HwHiAiUser/Ascend/ascend-toolkit/20.0.RC1
export PATH=/usr/local/python3.7.5/bin:${install_path}/atc/ccec_compiler/bin:${install_path}/atc/bin:$PATH
export PYTHONPATH=${install_path}/atc/python/site-packages/te:${install_path}/atc/python/site-packages/topi:$PYTHONPATH
export LD_LIBRARY_PATH=${install_path}/atc/lib64:$LD_LIBRARY_PATH
export ASCEND_OPP_PATH=${install_path}/opp
```

以`resnet50_export.air`为例，执行如下命令进行模型转换，在当前目录生成`resnet50_export.om`文件。

```bash
/home/HwHiAiUser/Ascend/ascend-toolkit/20.0.RC1/atc/bin/atc --framework=1 --model=./resnet50_export.air --output=./resnet50_export --input_format=NCHW --soc_version=Ascend310
```

其中：

- `--model`：原始模型文件的路径。
- `--output`：转换得到的OM模型文件的路径。
- `--input_format`：输入数据格式。

ATC工具详细资料可在[昇腾社区开发者文档](https://ascend.huawei.com/#/document?tag=developer)中选择相应CANN版本后，查找《ATC工具使用指南》章节查看。

## 编译推理代码

进入工程目录`acl_resnet50_sample`，设置如下环境变量：

```bash
export DDK_PATH=/home/HwHiAiUser/Ascend/ascend-toolkit/20.0.RC1
export NPU_HOST_LIB=/home/HwHiAiUser/Ascend/ascend-toolkit/20.0.RC1/acllib_linux.arm64/lib64/stub/
```

> `CMakeLists.txt`文件中`acllib`包的`include`的目录需要指定正确，否则会报`acl/acl.h`找不到的错误。`CMakeLists.txt`文件中指定`include`目录的代码位置如下，如果与实际安装目录不符，需要修改。

```text
...
#Header path

 include_directories(

  ${INC_PATH}/acllib_linux.arm64/include/

  ../

 )
...
```

执行如下命令创建编译目录：

```bash
mkdir -p build/intermediates/minirc
```

然后切换至编译目录：

```bash
cd build/intermediates/minirc
```

执行`cmake`命令：

```bash
cmake ../../../src -DCMAKE_CXX_COMPILER=aarch64-linux-gnu-g++ -DCMAKE_SKIP_RPATH=TRUE
```

再执行`make`命令编译即可。

```bash
make
```

编译完成后，在`acl_resnet50_sample/out`下会生成可执行`main`文件。

## 执行推理并查看结果

将生成的OM模型文件`resnet50_export.om`拷贝到`acl_resnet50_sample/out`目录下（和可执行`main`文件同路径），并确认`acl_resnet50_sample/test_data`目录中已经准备好输入数据样本，就可以执行推理了。

值得注意的是，需要设置如下环境变量，否则会导致推理不成功。

```bash
export LD_LIBRARY_PATH=/home/HwHiAiUser/Ascend/acllib/lib64/
```

进入到`acl_resnet50_sample/out`目录下，如果当前目录下`result`目录不存在，需要执行`mkdir result`命令创建该目录，然后执行如下命令进行推理。

```bash
./main  ./resnet50_export.om  ../test_data
```

执行成功后，可以看到推理结果如下，打印了`top5`的概率标签，并且输出结果会以`.bin`文件的格式保存在`acl_resnet50_sample/out/result`目录中。

```text
[INFO]  acl init success
[INFO]  open device 0 success
[INFO]  create context success
[INFO]  create stream success
[INFO]  get run mode success
[INFO]  load model ./resnet50_export.om success
[INFO]  create model description success
[INFO]  create model output success
[INFO]  start to process file:../test_data/test_data_1x3x224x224_1.bin
[INFO]  model execute success
[INFO]  top 1: index[2] value[0.941406]
[INFO]  top 2: index[3] value[0.291992]
[INFO]  top 3: index[1] value[0.067139]
[INFO]  top 4: index[0] value[0.013519]
[INFO]  top 5: index[4] value[-0.226685]
[INFO]  output data success
[INFO]  dump data success
[INFO]  start to process file:../test_data/test_data_1x3x224x224_2.bin
[INFO]  model execute success
[INFO]  top 1: index[2] value[0.946289]
[INFO]  top 2: index[3] value[0.296143]
[INFO]  top 3: index[1] value[0.072083]
[INFO]  top 4: index[0] value[0.014549]
[INFO]  top 5: index[4] value[-0.225098]
[INFO]  output data success
[INFO]  dump data success
[INFO]  unload model success, modelId is 1
[INFO]  execute sample success
[INFO]  end to destroy stream
[INFO]  end to destroy context
[INFO]  end to reset device is 0
[INFO]  end to finalize acl
```
