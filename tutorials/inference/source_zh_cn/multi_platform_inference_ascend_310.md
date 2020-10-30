# Atlas 200 DK使用MindSpore流程

`Linux` `Ascend` `Atlas 200` `初级` `中级` `高级` 

## 概述

`Atlas 200`开发者套件又称`Atlas 200 Developer Kit`（以下简称`Atlas 200 DK`），是以`Atlas 200` ` AI`加速模块（型号 3000）为核心的开发者板形态的终端类产品。 集成了海思`Ascend 310`  `AI`处理器，可以实现图像、视频等多种数据分析与推理计算，可广泛用于智能监控、机器人、无人机、视频服务器等场景。 

`Atlas 200 DK`使用`MindSpore`主要通过以下流程：

1、开发环境准备，包括制作`Atlas 200 DK`的`SD`卡 、配置`PYTHON`环境和刷配套开发软件包。

2、导出`AIR`模型文件，这里使用`resnet50`模型为例。

3、使用`ATC`工具将AIR模型文件转成`OM`模型，保存 。

4、编译推理代码，生成可执行`main`文件。

5、加载保存的`OM`模型，执行推理并查看结果。

## 开发环境准备

### 软件包准备

配置开发环境需要的脚本和软件包如下5类，共7个文件，都可以从华为云官网中获取，在基础软件下载模块中选选择产品系列和产品型号为：`Atlas 200 DK`，选择需要的文件，即可下载。

https://www.huaweicloud.com/ascend/resource/Software

```
1、制卡入口脚本： make_sd_card.py 

2、制作SD卡操作系统脚本 ：make_ubuntu_sd.sh 

3、Ubuntu操作系统镜像包 ： ubuntu-18.04.xx-server-arm64.iso 

4、开发者板驱动包与运行包 ：

Ascend310-driver-*{software version}*-ubuntu18.04.aarch64-minirc.tar.gz 

Ascend310-aicpu_kernels-*{software version}*-minirc.tar.gz 

Ascend-acllib-*{software version}*-ubuntu18.04.aarch64-minirc.run 

5、安装开发套件包：Ascend-Toolkit-*{version}*-arm64-linux_gcc7.3.0.run
```

 

### 制作SD卡

准备一个服务器（`ubuntu` +` arm`）、一张`SD`卡，建议容量不低于16G。读卡器或者`Atlas 200 DK`会通过`USB`与此`Ubuntu`服务器连接，制作`Atlas 200 DK`的系统启动盘  。制作脚本和系统镜像包等资源获取地址以及具体的安装过程参考如下：

https://support.huaweicloud.com/usermanual-A200dk_3000/atlas200dk_02_0011.html

### 连接Atlas 200 DK与Ubuntu服务器

`Atlas 200 DK`开发者板支持通过`USB`端口或者网线与`Ubuntu`服务器进行连接 。具体连接参考如下：

https://support.huaweicloud.com/usermanual-A200dk_3000/atlas200dk_02_0013.html

### 配置Python环境

这里主要是安装`Python`及其依赖以及`gcc`等软件 ，具体安装流程参考如下：

https://support.huaweicloud.com/usermanual-A200dk_3000/atlas200dk_02_0016.html

### 安装开发套件包

这里需要安装开发套件包 :`Ascend-Toolkit-*{version}*-arm64-linux_gcc7.3.0.run` ，具体安装流程参考如下：

https://support.huaweicloud.com/usermanual-A200dk_3000/atlas200dk_02_0017.html

## 导出AIR模型文件

需要在`Ascend 910`的机器上训练好目标网络，并保存`CheckPoint`文件，通过网络和`CheckPoint`生成对应的`AIR`格式模型文件并导出，导出流程具体参见官网教程： 

 <https://www.mindspore.cn/tutorial/training/zh-CN/master/use/save_model.html>

说明：这里提供使用`resnet50`模型导出的示例`AIR`文件：`resnet50_export.air`

## 将AIR模型文件转成OM模型

登录开发者板环境，创建目录放置`AIR`文件：`resnet50_export.air`，例如`/home/HwHiAiUser/Ascend/ascend-toolkit/20.0.RC1/acllib_linux.arm64/sample/acl_execute_model/acl_resnet50_sample/model`，并进入该路径下，设置环境变量，注意`install_path`指定为实际安装路径: 

```
export install_path=/home/HwHiAiUser/Ascend/ascend-toolkit/20.0.RC1
export PATH=/usr/local/python3.7.5/bin:${install_path}/atc/ccec_compiler/bin:${install_path}/atc/bin:$PATH
export PYTHONPATH=${install_path}/atc/python/site-packages/te:${install_path}/atc/python/site-packages/topi:$PYTHONPATH
export LD_LIBRARY_PATH=${install_path}/atc/lib64:$LD_LIBRARY_PATH
export ASCEND_OPP_PATH=${install_path}/opp
```

以下以`resnet50_export.air`为例，做模型转换，执行如下命令 ，在当前目录生成`resnet50_export.om`文件。

```
/home/HwHiAiUser/Ascend/ascend-toolkit/20.0.RC1/atc/bin/atc --framework=1 --model=./resnet50_export.air --output=./resnet50_export --input_format=NCHW --soc_version=Ascend310
```

`--model` ：原始模型文件的路径

` --output` ：转换得到的`om`模型文件的路径

`--input_format`: 输入数据格式

## 编译推理代码

登录开发者板环境，创建目录放置acl推理代码文件，例如：`/home/HwHiAiUser/Ascend/ascend-toolkit/20.0.RC1/acllib_linux.arm64/sample/acl_execute_model/acl_resnet50_sample`，相关代码可以从官网下载，推理目录结构如下：

```
​```text
└─acl_resnet50_sample
    ├── inc
    │   ├── model_process.h                   //声明资源初始化/销毁相关函数的头文件
    │   ├── sample_process.h                  //声明模型处理相关函数的头文件                
    │   ├── utils.h                           //声明公共函数（例如：文件读取函数）的头文件
    
    ├── model
    │   ├── resnet50_export.air               //AIR模型文件
    │   ├── resnet50_export.om                //om模型文件               
    
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
    │   ├── test_data_1x3x224x224_3.bin       //输入样本数据3
    
    ├── out
    │   ├── main                              //编译生成的可执行文件
    │   ├── result                            //输出结果目录                

​```
```

切换到工程目录：`.../acl_resnet50_sample`，先设置环境变量：

```
export DDK_PATH=/home/HwHiAiUser/Ascend/ascend-toolkit/20.0.RC1 

export NPU_HOST_LIB=/home/HwHiAiUser/Ascend/ascend-toolkit/20.0.RC1/acllib_linux.arm64/lib64/stub/ 
```

注意`CMakeLists.txt`文件中`acllib`包的`include`的目录要指定对确，否则会报`acl/acl.h `找不到的错误，`CMakeLists.txt`的 指定`include`目录的代码位置如下，如果与实际安装目录不符，需要修改。

```c++
... 
#Header path

 include_directories(    

	 ${INC_PATH}/acllib_linux.arm64/include/                                  

	 ../

 ) 
...
```

编译过程，需要先执行如下命令创建编译目录：

```
mkdir -p build/intermediates/minirc 
```

然后切换至编译目录：

```
 cd build/intermediates/minirc 
```

执行`cmake`命令：

```
cmake ../../../src -DCMAKE_CXX_COMPILER=aarch64-linux-gnu-g++ -DCMAKE_SKIP_RPATH=TRUE
```

再执行`make`命令编译即可，

```
make
```

编译完成后，在`.../acl_resnet50_sample/out`下会生成可执行`main`文件。

## 执行推理并查看结果

将前面已经生成的`OM`模型文件`resnet50_export.om` 拷贝到`.../acl_resnet50_sample/out`目录下（和可执行`main`文件同路径下），并确认`.../acl_resnet50_sample/test_data`	已经准备好输入数据样本，就可以执行推理了，值得注意的是，需要设置如下环境变量，否则推理会不成功。

```
export LD_LIBRARY_PATH=/home/HwHiAiUser/Ascend/acllib/lib64/ 
```

切换到`.../acl_resnet50_sample/out`目录下，执行如下命令进行推理，

```
./main  ./resnet50_export.om  ../test_data
```

执行成功后，可以看到推理结果如下，打印了`top5`的概率标签，并且输出结果会以`.bin`文件的格式保存在`.../acl_resnet50_sample/out/result`目录中。

```
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
[INFO]  top 1: index[2] value[0.941895]
[INFO]  top 2: index[3] value[0.296875]
[INFO]  top 3: index[1] value[0.071411]
[INFO]  top 4: index[0] value[0.016006]
[INFO]  top 5: index[4] value[-0.228516]
[INFO]  output data success
[INFO]  dump data success
[INFO]  start to process file:../test_data/test_data_1x3x224x224_3.bin
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

