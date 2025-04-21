# 模型混淆工具

[![查看源文件](https://mindspore-website.obs.cn-north-4.myhuaweicloud.com/website-images/r2.6.0rc1/resource/_static/logo_source.svg)](https://gitee.com/mindspore/docs/blob/r2.6.0rc1/docs/lite/docs/source_zh_cn/tools/obfuscator_tool.md)

## 概述

MindSpore Lite提供一个轻量级的离线模型混淆工具，可用于保护IOT或端侧设备上部署的模型文件的机密性。该工具通过对`ms`模型的网络结构和算子类型进行混淆，使得混淆后模型的计算逻辑变得难以理解。通过混淆工具生成的模型仍然是`ms`格式的，可直接通过Runtime推理框架执行推理（编译时需开启`mindspore/mindspore/lite/CMakeLists.txt`中的`MSLITE_ENABLE_MODEL_OBF`选项）。混淆会导致模型加载时延有轻微的增加，但对推理性能没有影响。

## Linux环境使用说明

### 环境准备

使用MindSpore Lite模型混淆工具，需要进行如下环境准备工作。

- 参考构建文档中的[环境要求](https://www.mindspore.cn/lite/docs/zh-CN/r2.6.0rc1/build/build.html#环境要求)和[编译示例](https://www.mindspore.cn/lite/docs/zh-CN/r2.6.0rc1/build/build.html#编译示例)编译x86_64版本。

### 目录结构

```text
mindspore-lite-{version}-linux-x64
└── tools
    └── obfuscator # 模型混淆工具
        └── msobfuscator          # 可执行程序
```

### 参数说明

MindSpore Lite模型混淆工具提供了多种参数设置，用户可根据需要来选择使用。此外，用户可输入`./msobfuscator --help`获取实时帮助。

下面提供详细的参数说明。

| 参数                        | 是否必选 | 参数说明                                               | 取值范围 | 默认值 |
| --------------------------- | -------- | ------------------------------------------------------ | -------- | ------ |
| `--help`                    | 否       | 打印全部帮助信息。                                     | -        | -      |
| `--modelFile=<MODELFILE>`   | 是       | 输入MindSpore Lite模型的路径。                         | -        | -      |
| `--outputFile=<OUTPUTFILE>` | 是       | 输出模型的路径，不需加后缀，可自动生成`.ms`后缀。      | -        | -      |
| `--obfDegree=<OBFDEGREE>`   | 否       | 设置模型的混淆程度，该值越大，模型中新增的节点和边越多 | \(0，1]  | 0.2    |

> - 支持输入`.ms`模型。
> - 参数名和参数值之间用等号连接，中间不能有空格。
> - 模型混淆会导致模型规模增大，obfDegree的值越大，模型规模增大的越多。

下面选取了几个常用示例，说明转换命令的使用方法。

### 使用示例

- 设置日志打印级别为INFO。

  ```bat
  set GLOG_v=1
  ```

  > 日志级别：0代表DEBUG，1代表INFO，2代表WARNING，3代表ERROR。

- 以MindSpore Lite模型LeNet为例，执行混淆命令。

  ```bash
  ./msobfuscator --modelFile=lenet.ms --outputFile=lenet_obf --obfDegree=0.5
  ```

  结果显示为：

  ```text
  OBFUSCATE MODEL lenet.ms SUCCESS!
  ```

  这表示已经成功将MindSpore Lite模型混淆，获得新文件`lenet_obf.ms`。
