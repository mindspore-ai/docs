# 集成Ascend使用说明

[![查看源文件](https://mindspore-website.obs.cn-north-4.myhuaweicloud.com/website-images/master/resource/_static/logo_source.svg)](https://gitee.com/mindspore/docs/blob/master/docs/lite/docs/source_zh_cn/advanced/third_party/ascend_info.md)

> - 端侧推理集成Ascend后端版本将于后续弃用，Ascend后端相关使用请参考云侧推理版本文档。
> - [云侧推理版本编译](https://mindspore.cn/lite/docs/zh-CN/master/mindir/build.html)
> - [云侧模型转换工具](https://mindspore.cn/lite/docs/zh-CN/master/mindir/converter.html)
> - [云侧基准测试工具](https://mindspore.cn/lite/docs/zh-CN/master/mindir/benchmark.html)

本文档介绍如何在Ascend环境的Linux系统上，使用MindSpore Lite 进行推理，以及动态shape功能的使用。目前，MindSpore Lite支持Atlas 200/300/500推理产品和Atlas推理系列产品芯片。

## 环境准备

### 确认系统环境信息

- 确认安装64位操作系统，[glibc](https://www.gnu.org/software/libc/)>=2.17，其中Ubuntu 18.04/CentOS 7.6/EulerOS 2.8是经过验证的。

- 确认安装[GCC 7.3.0版本](https://gcc.gnu.org/releases.html)。

- 确认安装[CMake 3.18.3及以上版本](https://cmake.org/download/)。
    - 安装完成后将CMake所在路径添加到系统环境变量。

- 确认安装Python 3.7.5或3.9.0版本。如果未安装或者已安装其他版本的Python，可以选择下载并安装：

    - Python 3.7.5版本 64位，下载地址：[官网](https://www.python.org/ftp/python/3.7.5/Python-3.7.5.tgz)或[华为云](https://mirrors.huaweicloud.com/python/3.7.5/Python-3.7.5.tgz)。
    - Python 3.9.0版本 64位，下载地址：[官网](https://www.python.org/ftp/python/3.9.0/Python-3.9.0.tgz)或[华为云](https://mirrors.huaweicloud.com/python/3.9.0/Python-3.9.0.tgz)。

- 如果您的环境为ARM架构，请确认当前使用的Python配套的pip版本>=19.3。

- 确认安装昇腾AI处理器配套软件包。

    - 昇腾软件包提供商用版和社区版两种下载途径：

        1. 商用版下载需要申请权限，下载链接与安装方式请参考[Ascend Data Center Solution 22.0.RC3安装指引文档](https://support.huawei.com/enterprise/zh/doc/EDOC1100280094)。

        2. 社区版下载不受限制，下载链接请前往[CANN社区版](https://www.hiascend.com/software/cann/community-history)，选择`5.1.RC2.alpha007`版本，以及在[固件与驱动](https://www.hiascend.com/hardware/firmware-drivers?tag=community)链接中获取对应的固件和驱动安装包，安装包的选择与安装方式请参照上述的商用版安装指引文档。
    - 安装包默认安装路径为`/usr/local/Ascend`。安装后确认当前用户有权限访问昇腾AI处理器配套软件包的安装路径，若无权限，需要root用户将当前用户添加到`/usr/local/Ascend`所在的用户组。
    - 安装昇腾AI处理器配套软件所包含的whl包。如果之前已经安装过昇腾AI处理器配套软件包，需要先使用如下命令卸载对应的whl包。

        ```bash
        pip uninstall te topi -y
        ```

        默认安装路径使用以下指令安装。如果安装路径不是默认路径，需要将命令中的路径替换为安装路径。

        ```bash
        pip install /usr/local/Ascend/ascend-toolkit/latest/lib64/topi-{version}-py3-none-any.whl
        pip install /usr/local/Ascend/ascend-toolkit/latest/lib64/te-{version}-py3-none-any.whl
        ```

### 配置环境变量

安装好Ascend软件包之后，需要导出Runtime相关环境变量，下述命令中`LOCAL_ASCEND=/usr/local/Ascend`的`/usr/local/Ascend`表示配套软件包的安装路径，需注意将其改为配套软件包的实际安装路径。

```bash
# control log level. 0-EBUG, 1-INFO, 2-WARNING, 3-ERROR, 4-CRITICAL, default level is WARNING.
export GLOG_v=2

# Conda environmental options
LOCAL_ASCEND=/usr/local/Ascend # the root directory of run package

# lib libraries that the run package depends on
export LD_LIBRARY_PATH=${LOCAL_ASCEND}/ascend-toolkit/latest/lib64:${LOCAL_ASCEND}/driver/lib64:${LOCAL_ASCEND}/ascend-toolkit/latest/opp/op_impl/built-in/ai_core/tbe/op_tiling:${LD_LIBRARY_PATH}

# Environment variables that must be configured
export TBE_IMPL_PATH=${LOCAL_ASCEND}/ascend-toolkit/latest/opp/op_impl/built-in/ai_core/tbe            # TBE operator implementation tool path
export ASCEND_OPP_PATH=${LOCAL_ASCEND}/ascend-toolkit/latest/opp                                       # OPP path
export PATH=${LOCAL_ASCEND}/ascend-toolkit/latest/compiler/ccec_compiler/bin/:${PATH}                  # TBE operator compilation tool path
export PYTHONPATH=${TBE_IMPL_PATH}:${PYTHONPATH}                                                       # Python library that TBE implementation depends on
```

## 执行converter工具

MindSpore Lite提供离线转换模型功能的工具，将多种类型的模型（Caffe、ONNX、TensorFlow、MindIR）转换为可在Ascend硬件上推理的模型。
首先，通过转换工具转换成的`ms`模型；然后，使用转换工具配套的Runtime推理框架执行推理，具体流程如下：

1. [下载](https://www.mindspore.cn/lite/docs/zh-CN/master/use/downloads.html)Ascend专用converter工具，当前仅支持Linux。

2. 解压下载的包

     ```bash
     tar -zxvf mindspore-lite-{version}-linux-x64.tar.gz
     ```

   {version}是发布包的版本号。

3. 将转换工具需要的动态链接库加入环境变量LD_LIBRARY_PATH

    ```bash
    export LD_LIBRARY_PATH=${LD_LIBRARY_PATH}:${PACKAGE_ROOT_PATH}/tools/converter/lib
    ```

   ${PACKAGE_ROOT_PATH}是解压得到的文件夹路径。

4. 进入转换目录

    ```bash
    cd ${PACKAGE_ROOT_PATH}/tools/converter/converter
    ```

5. 配置configFile(可选)

    用户可以通过此选项配置用于转模型时的Ascend Option选项，配置文件采用INI的风格，针对Ascend场景，可配置的参数为[acl_option_cfg_param]，参数的详细介绍如下表1所示，针对Ascend初始化可以通过acl_init_options参数进行配置，针对Ascend构图可以通过acl_build_options参数进行配置。

6. 执行converter，生成Ascend`ms`模型

    ```bash
    ./converter_lite --fmk=ONNX --modelFile=${model_name}.onnx --outputFile=${model_name}
    ```

    ${model_name}为模型文件名称，运行后的结果显示为：

    ```text
    CONVERT RESULT SUCCESS:0
    ```

    用户若想了解converter_lite转换工具的相关参数，可参考[参数说明](https://www.mindspore.cn/lite/docs/zh-CN/master/converter/converter_tool.html#参数说明)。

    说明：当原始模型输入shape不确定时，converter工具转换模型时要指定inputShape，同时configFile配置acl_option_cfg_param中input_shape_vector参数，取值相同，命令如下：

    ```bash
    ./converter_lite --fmk=ONNX --modelFile=${model_name}.onnx --outputFile=${model_name} --inputShape="input:1,64,64,1" --configFile="./config.txt"
    ```

    其中，config.txt内容如下:

    ```cpp
    [acl_option_cfg_param]
    input_shape_vector="[1,64,64,1]"
    ```

表1：配置[acl_option_cfg_param]参数

| 参数                        | 属性  | 功能描述                                                       | 参数类型 | 取值说明 |
| -------------------------- | ---- | ------------------------------------------------------------ | -------- | ------ |
| `input_format`             | 可选 | 指定模型输入format。 | String | 可选有`"NCHW"`、`"NHWC"` |
| `input_shape_vector`       | 可选 | 指定模型输入Shape， 按模型输入次序排列，用`；`隔开。 | String | 例如: `"[1,2,3,4];[4,3,2,1]"` |
| `precision_mode`           | 可选 | 配置模型精度模式。    | String | 可选有`"force_fp16"`、`"allow_fp32_to_fp16"`、`"must_keep_origin_dtype"`或者`"allow_mix_precision"`，默认为`"force_fp16"`|
| `op_select_impl_mode`      | 可选 | 配置算子选择模式。    | String | 可选有`"high_performance"`和`"high_precision"`，默认为`"high_performance"` |
| `dynamic_batch_size`       | 可选 | 指定[动态BatchSize](#动态batch-size)参数。 | String | `"2,4"`|
| `dynamic_image_size`       | 可选 | 指定[动态分辨率](#动态分辨率)参数。  | String | `"96,96;32,32"` |
| `fusion_switch_config_file_path` | 可选 | 配置[融合规则开关配置](https://www.hiascend.com/document/detail/zh/canncommercial/700/devtools/auxiliarydevtool/aoepar_16_034.html)文件路径及文件名。 | String   | -      |
| `insert_op_config_file_path` | 可选 | 模型插入[AIPP](https://www.hiascend.com/document/detail/zh/CANNCommunityEdition/81RC1alpha001/devaids/devtools/atc/atlasatc_16_0016.html)算子 | String  | [AIPP](https://www.hiascend.com/document/detail/zh/CANNCommunityEdition/81RC1alpha001/devaids/devtools/atc/atlasatc_16_0016.html)配置文件路径 |

## 推理工具runtime

converter得到转换模型后，使用配套的Runtime推理框架执行推理。有关使用Runtime执行推理详情见[使用Runtime执行推理（C++）](https://www.mindspore.cn/lite/docs/zh-CN/master/infer/runtime_cpp.html)。

## 执行benchmark

MindSpore Lite提供benchmark基准测试工具，它可以对MindSpore Lite模型前向推理的执行耗时进行定量分析（性能），还可以通过指定模型输出进行可对比的误差分析（精度）。
关于推理工具的一般说明，可参考[benchmark](https://www.mindspore.cn/lite/docs/zh-CN/master/tools/benchmark_tool.html)。

- 测性能

    ```bash
    ./benchmark --device=Ascend310 --modelFile=./models/test_benchmark.ms --timeProfiling=true
    ```

- 测精度

    ```bash
    ./benchmark --device=Ascend310 --modelFile=./models/test_benchmark.ms --inDataFile=./input/test_benchmark.bin --inputShapes=1,32,32,1 --accuracyThreshold=3 --benchmarkDataFile=./output/test_benchmark.out
    ```

    有关环境变量设置，将`libmindspore-lite.so`（目录为`mindspore-lite-{version}-{os}-{arch}/runtime/lib`）的`so`库所在的目录加入`${LD_LIBRARY_PATH}`。

## 高级特性

### 动态shape特性

在某些推理场景，如检测出目标后再执行目标识别网络，由于目标个数不固定导致目标识别网络输入BatchSize不固定。如果每次推理都按照最大的BatchSize或最大分辨率进行计算，会造成计算资源浪费。因此，推理需要支持动态BatchSize和动态分辨率的场景，Lite在Atlas 200/300/500推理产品上推理支持动态BatchSize和动态分辨率场景，在convert阶段通过configFile配置[acl_option_cfg_param]动态参数，转成`ms`模型，推理时使用model的[resize](https://www.mindspore.cn/lite/docs/zh-CN/master/infer/runtime_cpp.html#输入维度resize)功能，改变输入shape。

#### 动态Batch size

- 参数名

    dynamic_batch_size

- 功能

    设置动态batch档位参数，适用于执行推理时，每次处理图片数量不固定的场景，该参数需要与input_shape_vector配合使用，不能与dynamic_image_size同时使用。

- 取值

    最多支持100档配置，每一档通过英文逗号分隔，每个档位数值限制为：[1~2048]。例如配置文件中参数配置如下：

    ```cpp
    [acl_option_cfg_param]
    input_shape_vector="[-1,32,32,4]"
    dynamic_batch_size="2,4"
    ```

    其中，input_shape中的"-1"表示设置动态batch，档位可取值为"2,4"，即支持档位0: [2,32,32,4]，档位1: [4,32,32,4]。

- converter

    ```bash
    ./converter_lite --fmk=ONNX --modelFile=${model_name}.onnx --inputShape="input:4,32,32,4" --configFile=./config.txt --outputFile=${model_name}
    ```

    说明：使能动态BatchSize时，需要指定inputShape，值为最大档位对应的shape，即上节中档位1的值；同时通过configFile配置[acl_option_cfg_param]动态batch size，即上节示例中配置内容。

- 推理

    使能动态BatchSize，进行模型推理时，输入shape只能选择converter时设置的档位值，想切换到其他档位对应的输入shape，使用model [resize](https://www.mindspore.cn/lite/docs/zh-CN/master/infer/runtime_cpp.html#输入维度resize)功能。

- 注意事项

    1）若用户执行推理业务时，每次处理的图片数量不固定，则可以通过配置该参数来动态分配每次处理的图片数量。例如用户执行推理业务时需要每次处理2张、4张、8张图片，则可以配置为2、4、8，申请了档位后，模型推理时会根据实际档位申请内存。<br/>
    2）如果用户设置的档位数值过大或档位过多，可能会导致模型编译失败，此时建议用户减少档位或调低档位数值。<br/>
    3）如果用户设置的档位数值过大或档位过多，在运行环境执行推理时，建议执行swapoff -a命令关闭swap交换区间作为内存的功能，防止出现由于内存不足，将swap交换空间作为内存继续调用，导致运行环境异常缓慢的情况。<br/>

#### 动态分辨率

- 参数名

    dynamic_image_size

- 功能

    设置输入图片的动态分辨率参数。适用于执行推理时，每次处理图片宽和高不固定的场景，该参数需要与input_shape_vector配合使用，不能与dynamic_batch_size同时使用。

- 取值

    最多支持100档配置，每一档通过英文分号分隔。例如： "imagesize1_height,imagesize1_width;imagesize2_height,imagesize2_width"，指定的参数必须放在双引号中，每一组参数中间使用英文分号分隔。例如配置文件中参数配置如下：

    ```cpp
    [acl_option_cfg_param]
    input_format="NCHW"
    input_shape_vector="[2,3,-1,-1]"
    dynamic_image_size="64,64;96,96"
    ```

    其中，input_shape中的"-1"表示设置动态分辨率，即支持档位0: [2,3,64,64]，档位1: [2,3,96,96]。

- converter

    ```bash
    ./converter_lite --fmk=ONNX --modelFile=${model_name}.onnx --inputShape="input:2,3,96,96" --configFile=./config.txt --outputFile=${model_name}
    ```

    说明：使能动态BatchSize时，需要指定inputShape，值为最大档位对应的shape，即上节中档位1的值；同时通过configFile配置[acl_option_cfg_param]动态分辨率，即上节示例中配置内容。

- 推理

    使能动态分辨率，进行模型推理时，输入shape只能选择converter时设置的档位值，想切换到其他档位对应的输入shape，使用model的[resize](https://www.mindspore.cn/lite/docs/zh-CN/master/infer/runtime_cpp.html#输入维度resize)功能。

- 注意事项

    1）如果用户设置的分辨率数值过大或档位过多，可能会导致模型编译失败，此时建议用户减少档位或调低档位数值。<br/>
    2）如果用户设置了动态分辨率，实际推理时，使用的数据集图片大小需要与具体使用的分辨率相匹配。<br/>
    3）如果用户设置的分辨率数值过大或档位过多，在运行环境执行推理时，建议执行swapoff -a命令关闭swap交换区间作为内存的功能，防止出现由于内存不足，将swap交换空间作为内存继续调用，导致运行环境异常缓慢的情况。<br/>

## 算子支持

算子支持见[Lite 算子支持](https://www.mindspore.cn/lite/docs/zh-CN/master/reference/operator_list_lite.html)。
