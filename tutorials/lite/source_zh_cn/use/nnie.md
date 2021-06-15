# 集成NNIE使用说明

`NNIE` `Linux` `环境准备` `中级` `高级`

<!-- TOC -->

- [集成NNIE使用说明](#集成NNIE使用说明)
    - [目录结构](#目录结构)
        - [模型转换工具converter目录结构说明](#模型转换工具converter目录结构说明)
        - [模型推理工具runtime目录结构说明](#模型推理工具Runtime目录结构说明)
    - [工具使用](#工具使用)
        - [转换工具converter](#转换工具converter)
        - [推理工具runtime](#推理工具runtime)
    - [集成使用](#集成使用)

<!-- /TOC -->

<a href="https://gitee.com/mindspore/docs/blob/master/tutorials/lite/source_zh_cn/use/nnie.md" target="_blank"><img src="https://gitee.com/mindspore/docs/raw/master/resource/_static/logo_source.png"></a>

## 目录结构

### 模型转换工具converter目录结构说明

```text
mindspore-lite-{version}-inference-linux-x64
└── tools
    └── converter
        └── providers
            └── 3516D                # 嵌入式板型号
                ├── libmslite_nnie_converter.so        # 集成nnie的动态库
                ├── libmslite_nnie_data_process.so     # 处理nnie输入数据的动态库
                ├── libnnie_mapper.so        # 构建nnie wk文件的动态库
                └── third_party       # nnie依赖的三方动态库
                    ├── opencv-4.2.0
                    │   └── libopencv_xxx.so
                    └── protobuf-3.9.0
                        ├── libprotobuf.so
                        └── libprotoc.so
```

上述是nnie的集成目录结构，转换工具converter的其余目录结构详情，见[模型转换工具converter目录结构说明](https://www.mindspore.cn/tutorial/lite/zh-CN/master/use/build.html#converter)。

### 模型推理工具runtime目录结构说明

```text
mindspore-lite-{version}-linux-aarch32
└── providers
    └── 3516D        # 嵌入式板型号
        └── libmslite_nnie.so  # 集成nnie的动态库
```

上述是nnie的集成目录结构，推理工具runtime的其余目录结构详情，见[Runtime及其他工具目录结构说明](https://www.mindspore.cn/tutorial/lite/zh-CN/master/use/build.html#runtime)。

## 工具使用

### 转换工具converter

1. 进入**版本发布件根路径**。

  ```text
  cd mindspore-lite-{version}-inference-linux-x64
  ```

  若用户未进入**版本发件件根路径**，后续配置用户需按实际情况进行等价设置。

2. converter配置文件。

   用户创建后缀为.cfg的converter配置文件（以converter.cfg指代），文件内容如下：

   ```text
   plugin_path=./tools/converter/providers/3516D/libmslite_nnie_converter.so    # 用户请设置绝对路径
   ```

3. nnie配置文件。

   用户需参照HiSVP开发指南（nnie提供）自行配置（以nnie.cfg指代）。
   设定如下环境变量：

   ```shell
   export NNIE_CONFIG_PATH=nnie.cfg
   ```

4. converter环境变量设置。

   ```shell
   export NNIE_MAPPER_PATH=./tools/converter/providers/3516D/libnnie_mapper.so
   export NNIE_DATA_PROCESS_PATH=./tools/converter/providers/3516D/libmslite_nnie_data_process.so
   export LD_LIBRARY_PATH=${LD_LIBRARY_PATH}:./tools/converter/lib:./tools/converter/providers/3516D/third_party/opencv-4.2.0:./tools/converter/providers/3516D/third_party/protobuf-3.9.0
   ```

5. benchmark环境变量设置。

  运行于x86_64系统上的benchmark是用来生成校正集的，以供nnie学习量化参数。用户需设置以下环境变量:

   ```shell
   export LD_LIBRARY_PATH=${LD_LIBRARY_PATH}:./inference/lib
   export BENCHMARK_PATH=./tools/benchmark
   ```

6. 执行converter,当前只支持caffe。

   ```text
   ./tools/converter/converter/converter_lite --fmk=CAFFE --modelFile=${model_name}.prototxt --weightFile=${model_name}.caffemodel --configFile=converter.cfg --outputFile=${model_name}
   ```

   参数modelFile、weightFile、configFile、outputFile用户按实际情况进行设置。
   当用户在mindspore-lite-{version}-inference-linux-x64/tools/converter/converter目录下时，环境变量NNIE_MAPPER_PATH、NNIE_DATA_PROCESS_PATH、BENCHMARK_PATH可不设置。

### 推理工具runtime

以下是示例用法，用户可根据实际情况进行等价操作。

1. 3516D板目录创建。

   ```text
   mkdir /user/mindspore          # 存放非库文件
   mkdir /user/mindspore/lib      # 存放库文件
   ```

2. 传输文件。

   ```text
   scp libmindspore-lite.so libmslite_nnie.so libnnie_proposal.so root@${device_ip}:/user/mindspore/lib
   scp benchmark ${model_name}.ms root@${device_ip}:/user/mindspore
   ```

3. 设置动态库路径。

   ```shell
   export LD_LIBRARY_PATH=${LD_LIBRARY_PATH}:/user/mindspore/lib       # 此处未设置nnie依赖的动态库，用户需按实际情况进行配置。
   ```

4. 设置配置项（可选）。

   ```shell
   export TIME_STEP=1        # 循环或lstm网络运行的step数，范围：正整数，默认直：1
   export MAX_ROI_NUM=300    # 单张图片支持roi区域的最大数量，范围：正整数，默认直：300
   export CORE_IDS=0         # nnie运行内核id，支持多个，逗号分隔，范围：[0,7],默认直：0
   ```

5. 执行benchmark。

   ```text
   ./benchmark --modelFile=/.../${model_name}
   ```

   有关Benchmark使用详情，见[Benchmark使用](https://www.mindspore.cn/tutorial/lite/zh-CN/master/use/benchmark_tool.html)。

## 集成使用

有关集成使用详情，见[集成c++接口](https://www.mindspore.cn/tutorial/lite/zh-CN/master/use/runtime_cpp.html)。
