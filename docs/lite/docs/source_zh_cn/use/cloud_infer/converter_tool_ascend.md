# Ascend配置文件说明

<a href="https://gitee.com/mindspore/docs/blob/master/docs/lite/docs/source_zh_cn/use/cloud_infer/converter_tool_ascend.md" target="_blank"><img src="https://mindspore-website.obs.cn-north-4.myhuaweicloud.com/website-images/master/resource/_static/logo_source.png"></a>

## 概述

本文档介绍云侧模型转换工具在Ascend后端指定configFile参数时，配置文件的选项说明。

## 配置文件

表1：配置[ascend_context]参数

| 参数                        | 属性  | 功能描述                                                       | 参数类型 | 取值说明 |
| -------------------------- | ---- | ------------------------------------------------------------ | -------- | ------ |
| `input_format`             | 可选 | 指定模型输入format。 | String | 可选有`"NCHW"`、`"NHWC"`、`"ND"` |
| `input_shape`       | 可选 | 指定模型输入Shape，input_name必须是转换前的网络模型中的输入名称，按输入次序排列，用`；`隔开。 | String | 例如：`"input1:[1,64,64,3];input2:[1,256,256,3]"` |
| `dynamic_dims`       | 可选 | 指定动态BatchSize和动态分辨率参数。 | String | 见[动态shape配置](#动态shape配置) |
| `precision_mode`           | 可选 | 配置模型精度模式。    | String | 可选有`"enforce_fp32"`，`"preferred_fp32"`，`"enforce_fp16"`，`"enforce_origin"`或者`"preferred_optimal"`，默认为`"enforce_fp16"`|
| `op_select_impl_mode`      | 可选 | 配置算子选择模式。    | String | 可选有`"high_performance"`和`"high_precision"`，默认为`"high_performance"` |
| `output_type`       | 可选 | 指定网络输出数据类型。  | String | 可选有`"FP16"`、`"FP32"`、`"UINT8"` |
| `fusion_switch_config_file_path` | 可选 | 配置[融合规则开关配置](https://www.hiascend.com/document/detail/zh/canncommercial/601/inferapplicationdev/atctool/atctool_0078.html)文件路径及文件名。 | String   | 指定融合规则开关的配置文件      |
| `insert_op_config_file_path` | 可选 | 模型插入[AIPP](https://www.hiascend.com/document/detail/zh/canncommercial/601/inferapplicationdev/atctool/atctool_0018.html)算子 | String  | [AIPP](https://www.hiascend.com/document/detail/zh/canncommercial/601/inferapplicationdev/atctool/atctool_0021.html)配置文件路径 |
| `aoe_mode` | 可选 | [AOE](https://www.hiascend.com/document/detail/zh/canncommercial/601/devtools/auxiliarydevtool/aoe_16_001.html)自动调优模式 | String  | 可选有"subgraph turing"、"operator turing"或者"subgraph turing、operator turing"，默认不使能 |

## 动态shape配置

在某些推理场景，如检测出目标后再执行目标识别网络，由于目标个数不固定导致目标识别网络输入BatchSize不固定。如果每次推理都按照最大的BatchSize或最大分辨率进行计算，会造成计算资源浪费。因此，推理需要支持动态BatchSize和动态分辨率的场景，Lite在Ascend上推理支持动态BatchSize和动态分辨率场景，在convert阶段通过congFile配置[ascend_context]中dynamic_dims动态参数，推理时使用model的[Resize](https://www.mindspore.cn/lite/docs/zh-CN/master/use/cloud_infer/runtime_cpp.html#%E5%8A%A8%E6%80%81shape%E8%BE%93%E5%85%A5)功能，改变输入shape。

### 动态Batch size

- 参数名

    dynamic_dims

- 功能

    设置动态batch档位参数，适用于执行推理时，每次处理图片数量不固定的场景，该参数需要与input_shape配合使用，input_shape中-1的位置为动态batch所在的维度。

- 取值

    最多支持100档配置，每一档通过英文逗号分隔，每个档位数值限制为：[1~2048]。例如配置文件中参数配置如下：

    ```
    [ascend_context]
    input_shape=input:[-1,64,64,3]
    dynamic_dims=[1],[2]
    ```

    其中，input_shape中的"-1"表示设置动态batch，档位可取值为"1,2"，即支持档位0：[1,64,64,3]，档位1：[2,64,64,3]。

    若存在多个输入，不同输入对应的挡位需要一致，并用`;`隔开。

    ```
    [ascend_context]
    input_shape=input1:[-1,64,64,3];input2:[-1,256,256,3]
    dynamic_dims=[1],[2];[1],[2]
    ```

- converter

    ```bash
    ./converter_lite --fmk=ONNX --modelFile=${model_name}.onnx --configFile=./config.txt --optimize=ascend_oriented --outputFile=${model_name}
    ```

    说明：使能动态BatchSize时，不需要指定inputShape参数，仅需要通过configFile配置[ascend_context]动态batch size，即上节示例中配置内容。

- 推理

    使能动态BatchSize，进行模型推理时，输入shape只能选择converter时设置的档位值，想切换到其他档位对应的输入shape，使用model [Resize](https://www.mindspore.cn/lite/docs/zh-CN/master/use/cloud_infer/runtime_cpp.html#%E5%8A%A8%E6%80%81shape%E8%BE%93%E5%85%A5)功能。

- 注意事项

    1）若用户执行推理业务时，每次处理的图片数量不固定，则可以通过配置该参数来动态分配每次处理的图片数量。例如用户执行推理业务时需要每次处理2张，4张，8张图片，则可以配置为2,4,8，申请了档位后，模型推理时会根据实际档位申请内存。<br/>
    2）如果用户设置的档位数值过大或档位过多，可能会导致模型编译失败，此时建议用户减少档位或调低档位数值。<br/>
    3）如果用户设置的档位数值过大或档位过多，在运行环境执行推理时，建议执行swapoff -a命令关闭swap交换区间作为内存的功能，防止出现由于内存不足，将swap交换空间作为内存继续调用，导致运行环境异常缓慢的情况。<br/>

### 动态分辨率

- 参数名

    dynamic_dims

- 功能

    设置输入图片的动态分辨率参数。适用于执行推理时，每次处理图片宽和高不固定的场景，该参数需要与input_shape配合使用，input_shape中-1的位置为动态分辨率所在的维度。

- 取值

    最多支持100档配置，每一档通过英文逗号分隔。例如： "[imagesize1_height,imagesize1_width],[imagesize2_height,imagesize2_width]"。例如配置文件中参数配置如下：

    ```
    [ascend_context]
    input_format=NHWC
    input_shape=input:[1,-1,-1,3]
    dynamic_dims=[64,64],[19200,960]
    ```

    其中，input_shape中的"-1"表示设置动态分辨率，即支持档位0：[1,64,64,3]，档位1：[1,19200,960,3]。

- converter

    ```bash
    ./converter_lite --fmk=ONNX --modelFile=${model_name}.onnx --configFile=./config.txt --optimize=ascend_oriented --outputFile=${model_name}
    ```

    说明：使能动态BatchSize时，不需要指定inputShape参数，仅需要通过configFile配置[ascend_context]动态分辨率，即上节示例中配置内容。

- 推理

    使能动态分辨率，进行模型推理时，输入shape只能选择converter时设置的档位值，想切换到其他档位对应的输入shape，使用model的[Resize](https://www.mindspore.cn/lite/docs/zh-CN/master/use/cloud_infer/runtime_cpp.html#%E5%8A%A8%E6%80%81shape%E8%BE%93%E5%85%A5)功能。

- 注意事项

    1）如果用户设置的分辨率数值过大或档位过多，可能会导致模型编译失败，此时建议用户减少档位或调低档位数值。<br/>
    2）如果用户设置了动态分辨率，实际推理时，使用的数据集图片大小需要与具体使用的分辨率相匹配。<br/>
    3）如果用户设置的分辨率数值过大或档位过多，在运行环境执行推理时，建议执行swapoff -a命令关闭swap交换区间作为内存的功能，防止出现由于内存不足，将swap交换空间作为内存继续调用，导致运行环境异常缓慢的情况。<br/>

## AOE自动调优

AOE是一款专门为Davinci平台打造的计算图形性能自动调优工具。Lite使能AOE的能力，是在converter阶段集成AOE离线可执行程序，对图进行性能调优，生成知识库，并保存离线模型。该功能支持子图调优和算子调优。具体使用流程如下：

1. 配置环境变量

    ``${LOCAL_ASCEND}``为昇腾软件包安装所在路径

    ```bash
    export LOCAL_ASCEND=/usr/local/Ascend
    source ${LOCAL_ASCEND}/latest/bin/setenv.bash
    ```

    确认环境中AOE可执行程序可被找到并运行：

    ```bash
    aoe -h
    ```

2. 指定知识库路径

    AOE调优会生成算子知识库，默认的路径为

    ```bash
    ${HOME}/Ascend/latest/data/aoe/custom/graph(op)/${soc_version}
    ```

    （可选）也可通过``export TUNE_BANK_PATH``环境变量来自定义知识库路径。

3. 清除缓存

    为了模型编译能命中AOE生成的知识库，在使能AOE之前，最好先删除编译缓存，以免缓存复用，以昇腾310P环境，用户为root为例，删除``/root/atc_data/kernel_cache/Ascend310P3``和``/root/atc_data/fuzzy_kernel_cache/Ascend310P3``目录。

4. 配置文件指定选项

    在转换工具config配置文件中``[ascend_context]``指定AOE调优模式，如下举例中，会先执行子图调优，再执行算子调优。

    ```bash
    [ascend_context]
    aoe_mode="subgraph tuning, operator tuning"
    ```

> - 性能提升结果会因不同环境存在差异，实际时延减少百分比不完全等同于调优日志中所展示的结果。
> - AOE调优会在执行任务的当前目录下产生``aoe_workspace``目录，用于保存调优前后的模型，用于性能提升对比，以及调优所必须的过程数据和结果文件。该目录会占用额外磁盘空间，如500MB左右的原始模型会占用2~10GB的磁盘空间，视模型大小，算子种类结构，输入shape的大小等因素浮动。因此建议预留足够的磁盘空间，否则可能导致调优失败。
> - ``aoe_workspace``目录需要手动删除来释放磁盘空间。