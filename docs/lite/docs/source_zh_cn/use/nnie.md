# 集成NNIE使用说明

<a href="https://gitee.com/mindspore/docs/blob/master/docs/lite/docs/source_zh_cn/use/nnie.md" target="_blank"><img src="https://mindspore-website.obs.cn-north-4.myhuaweicloud.com/website-images/master/resource/_static/logo_source.png"></a>

## 目录结构

### 模型转换工具converter目录结构说明

```text
mindspore-lite-{version}-runtime-linux-x64
└── tools
    └── converter
        └── providers
            └── Hi3516D                # 嵌入式板型号
                ├── libmslite_nnie_converter.so        # 集成NNIE转换的动态库
                ├── libmslite_nnie_data_process.so     # 处理NNIE输入数据的动态库
                ├── libnnie_mapper.so        # 构建NNIE二进制文件的动态库
                └── third_party       # NNIE依赖的三方动态库
                    ├── opencv-4.2.0
                    │   └── libopencv_xxx.so
                    └── protobuf-3.9.0
                        ├── libprotobuf.so
                        └── libprotoc.so
```

上述是NNIE的集成目录结构，转换工具converter的其余目录结构详情，见[模型转换工具](https://www.mindspore.cn/lite/docs/zh-CN/master/use/converter_tool.html)。

### 模型推理工具runtime目录结构说明

```text
mindspore-lite-{version}-linux-aarch32
└── providers
    └── Hi3516D        # 嵌入式板型号
        └── libmslite_nnie.so  # 集成NNIE的动态库
        └── libmslite_proposal.so  # 集成proposal的样例动态库
```

上述是NNIE的集成目录结构，推理工具runtime的其余目录结构详情，见[目录结构](https://www.mindspore.cn/lite/docs/zh-CN/master/use/build.html#目录结构)。

## 工具使用

### 转换工具converter

#### 概述

MindSpore Lite提供离线转换模型功能的工具，将多种类型的模型（当前只支持Caffe）转换为可使用NNIE硬件加速推理的板端专属模型，可运行在Hi3516板上。
通过转换工具转换成的NNIE`ms`模型，仅支持在关联的嵌入式板上，使用转换工具配套的Runtime推理框架执行推理。关于转换工具的更一般说明，可参考[推理模型转换](https://www.mindspore.cn/lite/docs/zh-CN/master/use/converter_tool.html)。

#### 环境准备

使用MindSpore Lite模型转换工具，需要进行如下环境准备工作。

1. [下载](https://www.mindspore.cn/lite/docs/zh-CN/master/use/downloads.html)NNIE专用converter工具，当前仅支持Linux

2. 解压下载的包

     ```bash
     tar -zxvf mindspore-lite-{version}-linux-x64.tar.gz
     ```

     {version}是发布包的版本号。

3. 将转换工具需要的动态链接库加入环境变量LD_LIBRARY_PATH

    ```bash
    export LD_LIBRARY_PATH=${LD_LIBRARY_PATH}:${PACKAGE_ROOT_PATH}/tools/converter/lib:${PACKAGE_ROOT_PATH}/runtime/lib:${PACKAGE_ROOT_PATH}/tools/converter/providers/Hi3516D/third_party/opencv-4.2.0:${PACKAGE_ROOT_PATH}/tools/converter/providers/Hi3516D/third_party/protobuf-3.9.0
    ```

    ${PACKAGE_ROOT_PATH}是解压得到的文件夹路径。

#### 扩展配置

在转换阶段，为了能够加载扩展模块，用户需要配置扩展动态库路径。扩展相关的参数有`plugin_path`，`disable_fusion`。参数的详细介绍如下所示：

| 参数 | 属性 | 功能描述 | 参数类型 | 默认值 | 取值范围 |
| ---- | ---- | -------- | -------- | ------ | -------- |
| plugin_path | 可选 | 第三方库加载路径 | String | - | 如有多个请用`;`分隔 |
| disable_fusion | 可选 | 是否关闭融合优化 | String | off | off、on |

发布件中已为用户生成好默认的配置文件（converter.cfg）。文件内保存着NNIE动态库的相对路径，用户需要依据实际情况，决定是否需要手动修改该配置文件。该配置文件内容如下：

```ini
[registry]
plugin_path=../providers/Hi3516D/libmslite_nnie_converter.so
```

#### NNIE配置

NNIE模型可以使用NNIE硬件以提高模型运行速度，用户还需要配置NNIE自身的配置文件。用户需参照海思提供的《HiSVP 开发指南》中表格`nnie_mapper 配置选项说明`来进行配置，以nnie.cfg指代此配置文件：

nnie.cfg文件的示例参考如下：

```text
[net_type] 0
[image_list] ./input_nchw.txt
[image_type] 0
[norm_type] 0
[mean_file] null
```

> `input_nchw.txt`为被转换CAFFE模型的浮点文本格式的输入数据，详情请参照《HiSVP 开发指南》中的`image_list`说明。在配置文件中，配置选项caffemodel_file、prototxt_file、is_simulation、instructions_name不可配置，其他选项功能可正常配置。

#### 执行converter

1. 进入转换目录

    ```bash
    cd ${PACKAGE_ROOT_PATH}/tools/converter/converter
    ```

2. 配置环境变量（可选）

    若已执行第1步，进入到转换目录，则此步无需配置，默认值将使能。若用户未进入转换目录，则需在环境变量中声明转换工具所依赖的so和benchmark二进制执行程序的路径，如下所示：

    ```bash
    export NNIE_MAPPER_PATH=${PACKAGE_ROOT_PATH}/tools/converter/providers/Hi3516D/libnnie_mapper.so
    export NNIE_DATA_PROCESS_PATH=${PACKAGE_ROOT_PATH}/tools/converter/providers/Hi3516D/libmslite_nnie_data_process.so
    export BENCHMARK_PATH=${PACKAGE_ROOT_PATH}/tools/benchmark
    ```

    ${PACKAGE_ROOT_PATH}是下载得到的包解压后的路径。

3. 将nnie.cfg拷贝到转换目录并设置如下环境变量

    ```bash
    export NNIE_CONFIG_PATH=./nnie.cfg
    ```

   如果用户实际的配置文件就叫nnie.cfg，且与converter_lite在同级路径上，则可不用配置。

4. 执行converter，生成NNIE`ms`模型

    ```bash
    ./converter_lite --fmk=CAFFE --modelFile=${model_name}.prototxt --weightFile=${model_name}.caffemodel --configFile=./converter.cfg --outputFile=${model_name}
    ```

    ${model_name}为模型文件名称，运行后的结果显示为：

     ```text
     CONVERTER RESULT SUCCESS:0
     ```

     用户若想了解converter_lite转换工具的相关参数，可参考[参数说明](https://www.mindspore.cn/lite/docs/zh-CN/master/use/converter_tool.html#参数说明)。

### 推理工具runtime

#### 概述

得到转换模型后，可在关联的嵌入式板上，使用板子配套的Runtime推理框架执行推理。MindSpore Lite提供benchmark基准测试工具，它可以对MindSpore Lite模型前向推理的执行耗时进行定量分析（性能），还可以通过指定模型输出进行可对比的误差分析（精度）。
关于推理工具的一般说明，可参考[benchmark](https://www.mindspore.cn/lite/docs/zh-CN/master/use/benchmark_tool.html)。

#### 环境准备

以下为示例用法，用户可根据实际情况进行等价操作。

1. [下载](https://www.mindspore.cn/lite/docs/zh-CN/master/use/downloads.html)NNIE专用模型推理工具，当前仅支持Hi3516D

2. 解压下载的包

     ```bash
     tar -zxvf mindspore-lite-{version}-linux-aarch32.tar.gz
     ```

     {version}是发布包的版本号。

3. 在Hi3516D板上创建存放目录

   登陆板端，创建工作目录

   ```bash
   mkdir /user/mindspore          # 存放benchmark执行文件及模型
   mkdir /user/mindspore/lib      # 存放依赖库文件
   ```

4. 传输文件

   向Hi3516D板端传输benchmark工具、模型、so库。其中libmslite_proposal.so为MindSpore Lite提供的proposal算子实现样例so，若用户模型里含有自定义的proposal算子，用户需参考[proposal算子使用说明](#proposal算子使用说明)生成libnnie_proposal.so替换该so文件，以进行正确推理。

   ```bash
   scp libmindspore-lite.so libmslite_nnie.so libmslite_proposal.so root@${device_ip}:/user/mindspore/lib
   scp benchmark ${model_path} root@${device_ip}:/user/mindspore
   ```

   ${model_path}为转换后ms模型文件路径

5. 设置动态库路径

   NNIE模型的推理，还依赖海思提供NNIE相关板端动态库，包括：libnnie.so、libmpi.so、libVoiceEngine.so、libupvqe.so、libdnvqe.so。

   用户需在板端保存这些so，并将路径传递给LD_LIBRARY_PATH环境变量。
   在示例中，这些so位于/usr/lib下，用户需按实际情况进行配置：

   ```bash
   export LD_LIBRARY_PATH=/user/mindspore/lib:/usr/lib:${LD_LIBRARY_PATH}
   ```

6. 设置配置项（可选）

   若用户模型含有proposal算子，需根据proposal算子实现情况，配置MAX_ROI_NUM环境变量：

   ```bash
   export MAX_ROI_NUM=300    # 单张图片支持roi区域的最大数量，范围：正整数，默认值：300。
   ```

   若用户模型为循环或lstm网络，需根据实际网络运行情况，配置TIME_STEP环境变量，其他要求[见多图片batch运行及多step运行](#多图片batch运行及多step运行)：

   ```bash
   export TIME_STEP=1        # 循环或lstm网络运行的step数，范围：正整数，默认值：1。
   ```

   若板端含有多个NNIE硬件，用户可通过CORE_IDS环境变量指定模型运行在哪个NNIE设备上，
   若模型被分段（用户可用netron打开模型，观察模型被分段情况），可依序分别配置每个分段运行在哪个设备上，未被配置分段运行在最后被配置的NNIE设备上：

   ```bash
   export CORE_IDS=0         # NNIE运行内核id，支持模型分段独立配置，使用逗号分隔(如export CORE_IDS=1,1)，默认值：0
   ```

7. 构建图片输入（可选）

   若converter导出模型时喂给mapper的校正集用的是图片，则传递给benchmark的输入需是int8的输入数据，即需要把图片转成int8传递给benchmark。
   这里采用python给出转换示范样例：

   ``` python
   import sys
   import cv2

   def usage():
       print("usage:\n"
             "example: python generate_input_bin.py xxx.img BGR 224 224\n"
             "argv[1]: origin image path\n"
             "argv[2]: RGB_order[BGR, RGB], should be same as nnie mapper config file's [RGB_order], default is BGR\n"
             "argv[3]: input_h\n"
             "argv[4]: input_w"
             )

   def main(argvs):
       if argvs[1] == "-h":
           usage()
           print("EXIT")
           exit()
       img_path = argvs[1]
       rgb_order = argvs[2]
       input_h = int(argvs[3])
       input_w = int(argvs[4])
       img = cv2.imread(img_path)
       if rgb_order == "RGB":
           img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
       img_hwc = cv2.resize(img, (input_w, input_h))
       outfile_name = "1_%s_%s_3_nhwc.bin" %(argvs[3], argvs[4])
       img_hwc.tofile(outfile_name)
       print("Generated " + outfile_name + " file success in current dir.")

   if __name__ == "__main__":
       if len(sys.argv) == 1:
           usage()
           print("EXIT")
           exit()
       elif len(sys.argv) != 5:
           print("Input argument is invalid.")
           usage()
           print("EXIT")
           exit()
       else:
           main(sys.argv)
   ```

#### 执行benchmark

```text
cd /user/mindspore
./benchmark --modelFile=${model_path}
```

${model_path}为转换后ms模型文件路径

执行该命令，会生成模型的随机输入，并执行前向推理。有关benchmark的其他使用详情，如耗时分析与推理误差分析等，见[Benchmark使用](https://www.mindspore.cn/lite/docs/zh-CN/master/use/benchmark_tool.html)。

有关模型的输入数据格式要求，见[SVP工具链相关功能支持及注意事项（可选）](#SVP工具链相关功能支持及注意事项（可选）)。

## 集成使用

有关集成使用详情，见[集成c++接口](https://www.mindspore.cn/lite/docs/zh-CN/master/use/runtime_cpp.html)。

## SVP工具链相关功能支持及注意事项（高级选项）

在模型转换时，由NNIE_CONFIG_PATH环境变量声明的nnie.cfg文件，提供原先SVP工具链相关功能，支持除caffemodel_file、prototxt_file、is_simulation、instructions_name外其他字段的配置，相关注意实现如下：

### 板端运行输入Format须是NHWC

  转换后的`ms`模型只接受NHWC格式的数据输入，若image_type被声明为0，则接收NHWC格式的float32数据，若image_type被声明为1，则接收NHWC的uint8数据输入。

### image_list说明

  nnie.cfg中image_list字段含义与原先不变，当image_type声明为0时，按行提供chw格式数据，无论原先模型是否是nchw输入。

### image_type限制

  MindSpore Lite不支持image_type为3和5时的网络输入，用户设为0或1。

### image_list和roi_coordinate_file个数说明

  用户只需提供与模型输入个数相同数量的image_list，若模型中含有ROI Pooling或PSROI Pooling层，用户需提供roi_coordinate_file，数量与顺序和prototxt内的ROI Pooling或PSROI Pooling层的个数与顺序对应。

### prototxt中节点名_cpu后缀支持

  SVP工具链中，可通过在prototxt文件的节点名后使用_cpu后缀来，声明cpu自定义算子。MindSpore Lite中忽略_cpu后缀，不做支持。用户若想重定义MindSpore Lite已有的算子实现或新增新的算子，可通过[自定义算子注册](https://www.mindspore.cn/lite/docs/zh-CN/master/use/register_kernel.html)的方式进行注册。

### prototxt中Custom算子支持

  SVP工具链中，通过在prototxt中声明custom层，实现推理时分段，并由用户实现cpu代码。在MindSpore Lite中，用户需在Custom层中增加op_type属性，并通过[自定义算子注册](https://www.mindspore.cn/lite/docs/zh-CN/master/use/register_kernel.html)的方式进行在线推理代码的注册。

  Custom层的修改样例如下：

  ```text
  layer {
    name: "custom1"
    type: "Custom"
    bottom: "conv1"
    top: "custom1_1"
    custom_param {
      type: "MY_CUSTOM"
      shape {
          dim: 1
          dim: 256
          dim: 64
          dim: 64
      }
  }
  }
  ```

  在该示例中定义了一个MY_CUSTOM类型的自定义算子，推理时用户需注册一个类型为MY_CUSTOM的自定义算子。

### prototxt中top域的_report后缀支持

  MindSpore Lite在转换NNIE模型时，会将大部分的算子融合为NNIE运行的二进制文件，用户无法观察到中间算子的输出，通过在top域上添加”_report“后缀，转换构图时会将中间算子的输出添加到融合后的层输出中，若原先该算子便有输出（未被融合），则维持不变。

  在推理运行时，用户可通过[回调运行](https://www.mindspore.cn/lite/docs/zh-CN/master/use/runtime_cpp.html#回调运行)得到中间算子输出。

  MindSpore Lite解析_report的相应规则，及与[inplace机制](#inplace机制)的冲突解决，参照《HiSVP 开发指南》中的定义说明。

### inplace机制

  使用Inplace层写法，可运行芯片高效模式。转换工具默认将Prototxt中符合芯片支持Inplace层的所有层进行改写，用户如需关闭该功能，可通过如下环境声明：

  ```bash
  export NNIE_DISABLE_INPLACE_FUSION=off         # 设置为on或未设置时，使能Inplace自动改写
  ```

  当自动改写被关闭时，若需对个别层使能芯片高效模式，可手动改写Prototxt里面的相应层。

### 多图片batch运行及多step运行

  用户若需同时前向推理多个输入数据（多个图片），可通过[输入维度Resize](https://www.mindspore.cn/lite/docs/zh-CN/master/use/runtime_cpp.html#输入维度resize)将模型输入的第一维resize为输入数据个数。NNIE模型只支持对第一个维度（'n'维）进行resize，其他维度（'hwc'）不可变。

  对于循环或lstm网络，用户需根据step值，配置TIME_STEP环境变量，同时resize模型输入。
  设一次同时前向推理的数据的个数为input_num，对于序列数据输入的节点resize为input_num * step，非序列数据输入的节点resize为input_num。

  含有proposal算子的模型，不支持batch运行，不支持resize操作。

### 节点名称的变动

  模型转换为NNIE模型后，各节点名称可能发生变化，用户可通过netron打开模型，得到变化后的节点名。

### proposal算子使用说明

  MindSpore Lite提供Proposal算子的样例代码，在该样例中，以[自定义算子注册](https://www.mindspore.cn/lite/docs/zh-CN/master/use/register_kernel.html)的方式实现proposal算子及该算子infer shape的注册。用户可将其修改为自身模型匹配的实现后，进行集成使用。
  > 你可以在这里下载完整的样例代码：
  >
  > <https://gitee.com/mindspore/docs/tree/master/docs/sample_code/nnie_proposal>

### 分段机制说明及8段限制

  由于NNIE芯片支持的算子限制，在含有NNIE芯片不支持的算子时，需将模型分段为可支持层与不可支持层。
  板端芯片支持最多8段的可支持层，当分段后的可支持层数量大于8段时，模型将无法运行，用户可通过netron观察Custom算子（其属性中含有type:NNIE），得到转换后的NNIE支持层数量。
