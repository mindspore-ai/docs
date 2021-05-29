# 在微控制器上执行推理

 `Linux` `IoT` `C++` `模型代码生成` `推理应用` `初级` `中级`

<!-- TOC -->

- [在微控制器上执行推理](#在微控制器上执行推理)
    - [概述](#概述)
    - [获取codegen](#获取codegen)
    - [参数说明](#参数说明)
    - [使用步骤](#使用步骤)
    - [使用CodeGen在STM开发板上执行推理](#使用CodeGen在STM开发板上执行推理)
    - [更多详情](#更多详情)

<!-- /TOC -->

<a href="https://gitee.com/mindspore/docs/blob/master/tutorials/lite/source_zh_cn/use/micro.md" target="_blank"><img src="https://gitee.com/mindspore/docs/raw/master/resource/_static/logo_source.png"></a>

## 概述

相较于移动终端，IoT设备上系统资源有限，对ROM空间占用、运行时内存和功耗要求较高。MindSpore Lite提供代码生成工具codegen，将运行时编译、解释计算图，移至离线编译阶段。仅保留推理所必须的信息，生成极简的推理代码。codegen可对接NNACL和CMSIS算子库，支持生成可在x86/ARM64/ARM32A/ARM32M平台部署的推理代码。

代码生成工具codegen的使用流程如下：

1. 通过MindSpore Lite转换工具[Converter](https://www.mindspore.cn/tutorial/lite/zh-CN/master/use/converter_tool.html)，将训练好的模型文件转换为`*.ms`格式；

2. 通过自动代码生成工具codegen，输入`*.ms`模型自动生成源代码。

![img](../images/lite_codegen.png)

## 获取codegen

自动代码生成工具，可以通过两种方式获取：

1. MindSpore官网下载[Release版本](https://www.mindspore.cn/tutorial/lite/zh-CN/master/use/downloads.html)。
2. 从源码开始[编译构建](https://www.mindspore.cn/tutorial/lite/zh-CN/master/use/build.html)。

> 目前模型生成工具仅支持在Linux x86_64架构下运行。

## 参数说明

详细参数说明如下：

| 参数            | 是否必选 | 参数说明                         | 取值范围                   | 默认值    |
| --------------- | -------- | -------------------------------| -------------------------- | --------- |
| help            | 否       | 打印使用说明信息                 | -                          | -         |
| codePath        | 是       | 生成代码的路径                   | -                          | ./(当前目录)|
| target          | 是       | 生成代码针对的平台               | x86, ARM32M, ARM32A, ARM64 | x86       |
| modelPath       | 是       | 输入模型文件路径                 | -                          | -         |
| supportParallel | 否       | 是否生成支持多线程的代码          | true, false                | false     |
| debugMode       | 否       | 是否以生成调试模式的代码          | true, false                | false     |

> 输入模型文件，需要经过MindSpore Lite Converter工具转换成.ms格式。
>
> os不支持文件系统时，debugMode不可用。
>
> 生成的推理接口详细使用说明，请参考[API文档](https://www.mindspore.cn/doc/api_cpp/zh-CN/master/index.html)。
>
> 以下三个接口暂不支持：
>
> 1. `virtual std::unordered_map<String, mindspore::tensor::MSTensor *> GetOutputs() const = 0;`
> 2. `virtual Vector<tensor::MSTensor *> GetOutputsByNodeName(const String &node_name) const = 0;`
> 3. `virtual int Resize(const Vector<tensor::MSTensor *> &inputs, const Vector<Vector<int>> &dims) = 0;`

## 使用说明

以MNIST分类网络为例：

```bash
./codegen --modelPath=./mnist.ms --codePath=./
```

执行成功后，会在codePath指定的目录下，生成名为mnist的文件夹，内容如下：

```text
mnist
├── benchmark                  # 集成调试相关的例程
│   ├── benchmark.cc
│   ├── calib_output.cc
│   ├── calib_output.h
│   ├── load_input.c
│   └── load_input.h
├── CMakeLists.txt
└── src                        # 源文件
    ├── CMakeLists.txt
    ├── mmodel.h
    ├── net.bin                # 二进制形式的模型权重
    ├── net.c
    ├── net.cmake
    ├── net.h
    ├── session.cc
    ├── session.h
    ├── tensor.cc
    ├── tensor.h
    ├── weight.c
    └── weight.h
```

## 使用CodeGen在STM开发板上执行推理

本教程以在STM32F746单板上编译部署生成模型代码为例，演示了codegen编译模型在Cortex-M平台的使用。更多关于Arm Cortex-M的详情可参见其[官网](https://developer.arm.com/ip-products/processors/cortex-m)。

### STM32F746编译依赖

模型推理代码的编译部署需要在Windows上安装[J-Link](https://www.segger.com/)、[STM32CubeMX](https://www.st.com/content/st_com/en.html)、[GNU Arm Embedded Toolchain](https://developer.arm.com/tools-and-software/open-source-software/developer-tools/gnu-toolchain/gnu-rm)等工具来进行交叉编译。

- [STM32CubeMX Windows版本](https://www.st.com/content/ccc/resource/technical/software/sw_development_suite/group0/0b/05/f0/25/c7/2b/42/9d/stm32cubemx_v6-1-1/files/stm32cubemx_v6-1-1.zip/jcr:content/translations/en.stm32cubemx_v6-1-1.zip) >= 6.0.1

- [GNU Arm Embedded Toolchain](https://developer.arm.com/tools-and-software/open-source-software/developer-tools/gnu-toolchain/gnu-rm/downloads)  >= 9-2019-q4-major-win32

- [J-Link Windows版本](https://www.segger.com/downloads/jlink/) >= 6.56
- [GCC](https://gcc.gnu.org/releases.html) >= 7.3.0
- [CMake](https://cmake.org/download/) >= 3.18.3

### STM32F746工程构建

- 需要组织的工程目录如下：

    ```text
    ├── mnist              # codegen生成的模型推理代码
    ├── include            # 模型推理对外API头文件目录(需要自建)
    └── operator_library   # 模型推理算子相关文件(需要自建)
    ```

> 模型推理对外API头文件可由MindSpore团队发布的[Release包](https://www.mindspore.cn/tutorial/lite/zh-CN/master/use/downloads.html)中获取。
>
> 在编译此工程之前需要预先获取对应平台所需要的算子文件，由于Cortex-M平台工程编译一般涉及到较复杂的交叉编译，此处不提供直接预编译的算子库静态库，而是用户根据模型自行组织文件，自主编译Cortex-M7 、Coretex-M4、Cortex-M3等工程(对应工程目录结构已在示例代码中给出，用户可自主将对应ARM官方的CMSIS源码放置其中即可)。

- 使用codegen编译[MNIST手写数字识别模型](https://download.mindspore.cn/model_zoo/official/lite/mnist_lite/mnist.ms)，生成对应的STM32F46推理代码。具体命令如下：

    ```bash
    ./codegen --codePath=. --modelPath=mnist.ms --target=ARM32M
    ```

- 生成代码工程目录如下：

    ```text
    ├── mnist               # 生成代码的根目录
        ├── benchmark       # 生成代码的benchmark目录
        └── src             # 模型推理代码目录
    ```

- 预置算子静态库的目录如下：

    ```text
    ├── operator_library    # 平台算子库目录
        ├── include         # 平台算子库头文件目录
        └── nnacl           # MindSpore团队提供的平台算子库源文件
        └── wrapper         # MindSpore团队提供的平台算子库源文件
        └── CMSIS           # Arm官方提供的CMSIS平台算子库源文件
    ```

    > 在使用过程中，引入CMSIS v5.7.0 Softmax相关的CMSIS算子文件时，头文件中需要加入`arm_nnfunctions.h`。

#### 代码工程编译

1. 环境测试

    安装好交叉编译所需环境后，需要在Windows环境中依次将其加入到环境变量中。

    ```text
    gcc -v               # 查看GCC版本
    arm-none-eabi-gdb -v # 查看交叉编译环境
    jlink -v             # 查看J-Link版本
    make -v              # 查看Make版本
    ```

    以上命令均成功返回值时，表明环境准备已完成，可以继续进入下一步，否则请务必先安装上述环境。

2. 生成STM32F746单板初始化代码（[详情示例代码](https://gitee.com/mindspore/mindspore/tree/master/mindspore/lite/micro/example/mnist_stm32f746)）

    - 启动 STM32CubeMX，新建project，选择单板STM32F746IG。
    - 成功以后，选择`Makefile` ，`generator code`。
    - 在生成的工程目录下打开`cmd`，执行`make`，测试初始代码是否成功编译。

    ```text
    # make成功结果
    arm-none-eabi-size build/test_stm32f746.elf
      text    data     bss     dec     hex filename
      3660      20    1572    5252    1484 build/test_stm32f746.elf
    arm-none-eabi-objcopy -O ihex build/test_stm32f746.elf build/test_stm32f746.hex
    arm-none-eabi-objcopy -O binary -S build/test_stm32f746.elf build/test_stm32f746.bin
    ```

#### 编译模型

1. 拷贝MindSpore团队提供算子文件以及对应头文件到STM32CubeMX生成的工程目录中。

2. 拷贝codegen生成模型推理代码到 STM32CubeMX生成的代码工程目录中。

    ```text
    ├── .mxproject
    ├── build             # 工程编译输出目录
    ├── Core
    ├── Drivers
    ├── mnist             # codegen生成的cortex-m7 模型推理代码
    ├── Makefile          # 编写工程makefile文件组织mnist && operator_library源文件到工程目录中
    ├── startup_stm32f746xx.s
    ├── STM32F746IGKx_FLASH.ld
    └── test_stm32f746.ioc
    ```

3. 修改makefile文件，组织算子静态库以及模型推理代码，具体makefile文件内容参见[示例](https://gitee.com/mindspore/mindspore/tree/master/mindspore/lite/micro/example/mnist_stm32f746)。

    ```text
    # C includes
    C_INCLUDES =  \
    -ICore/Inc \
    -IDrivers/STM32F7xx_HAL_Driver/Inc \
    -IDrivers/STM32F7xx_HAL_Driver/Inc/Legacy \
    -IDrivers/CMSIS/Device/ST/STM32F7xx/Include \
    -Imnist/operator_library/include \                # 新增，指定算子库头文件目录
    -Imnist/include \                                 # 新增，指定模型推理代码头文件
    -Imnist/src                                       # 新增，指定模型推理代码源文件
    ......
    ```

4. 在工程目录的Core/Src的main.c编写模型调用代码，具体代码新增如下：

    ```cpp
    while (1) {
        /* USER CODE END WHILE */
        SEGGER_RTT_printf(0, "***********mnist test start***********\n");
        const char *model_buffer = nullptr;
        int model_size = 0;
        session::LiteSession *session = mindspore::session::LiteSession::CreateSession(model_buffer, model_size, nullptr);
        Vector<tensor::MSTensor *> inputs = session->GetInputs();
        size_t inputs_num = inputs.size();
        void *inputs_binbuf[inputs_num];
        int inputs_size[inputs_num];
        for (size_t i = 0; i < inputs_num; ++i) {
          inputs_size[i] = inputs[i]->Size();
        }
        // here mnist only have one input data,just hard code to it's array;
        inputs_binbuf[0] = mnist_inputs_data;
        for (size_t i = 0; i < inputs_num; ++i) {
          void *input_data = inputs[i]->MutableData();
          memcpy(input_data, inputs_binbuf[i], inputs_size[i]);
        }
        int ret = session->RunGraph();
        if (ret != lite::RET_OK) {
          return lite::RET_ERROR;
        }
        Vector<String> outputs_name = session->GetOutputTensorNames();
        for (int i = 0; i < outputs_name.size(); ++i) {
          tensor::MSTensor *output_tensor = session->GetOutputByTensorName(outputs_name[i]);
          if (output_tensor == nullptr) {
            return -1;
          }
          float *casted_data = static_cast<float *>(output_tensor->MutableData());
          if (casted_data == nullptr) {
            return -1;
          }
          for (size_t j = 0; j < 10 && j < output_tensor->ElementsNum(); j++) {
            SEGGER_RTT_printf(0, "output: [%d] is : [%d]/100\n", i, casted_data[i] * 100);
          }
        }
        delete session;
        SEGGER_RTT_printf(0, "***********mnist test end***********\n");
    ```

5. 在工程跟目中目录使用管理员权限打开`cmd` 执行 `make`进行编译。

    ```bash
    make
    ```

### STM32F746工程部署

使用J-Link将可执行文件拷贝到单板上并做推理。

```text
jlinkgdbserver           # 启动jlinkgdbserver 选定target device为STM32F746IG
jlinkRTTViewer           # 启动jlinkRTTViewer 选定target devices为STM32F746IG
arm-none-eabi-gdb        # 启动arm-gcc gdb服务
file build/target.elf    # 打开调测文件
target remote 127.0.0.1  # 连接jlink服务器
monitor reset            # 重置单板
monitor halt             # 挂起单板
load                     # 加载可执行文件到单板
c                        # 执行模型推理
```

## 更多详情

### [Linux_x86_64平台编译部署](https://gitee.com/mindspore/mindspore/tree/master/mindspore/lite/micro/example/mnist_x86)

### [Android平台编译部署](https://gitee.com/mindspore/mindspore/tree/master/mindspore/lite/micro/example/mobilenetv2)
