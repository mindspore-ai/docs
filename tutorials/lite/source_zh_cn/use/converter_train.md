# 训练模型转换

`Linux` `环境准备` `模型导出` `模型转换` `中级` `高级`

[![查看源文件](../_static/logo_source.png)](https://gitee.com/mindspore/docs/blob/r1.1/tutorials/lite/source_zh_cn/use/converter_train.md)

## 概述

创建MindSpore端侧模型的步骤：

- 首先基于MindSpore架构使用Python创建网络模型，并导出为`.mindir`文件，参见云端的[保存模型](https://www.mindspore.cn/tutorial/training/zh-CN/r1.1/use/save_model.html#mindir)。
- 然后将`.mindir`模型文件转换成`.ms`文件，`.ms`文件可以导入端侧设备并基于MindSpore端侧框架训练。

## Linux环境

### 环境准备

MindSpore Lite 模型转换工具提供了多个参数，目前工具仅支持Linux系统，环境准备步骤：

- 编译或下载已编译好的模型转换工具。
- 配置模型转换工具的环境变量。

### 参数说明

下表为MindSpore Lite训练模型转换工具使用到的参数：

| 参数                        | 是否必选 | 参数说明                                    | 取值范围    | 默认值 |
| --------------------------- | -------- | ------------------------------------------- | ----------- | ------ |
| `--help`                    | 否       | 打印全部帮助信息                            | -           | -      |
| `--fmk=<FMK>`               | 是       | 输入模型的原始格式                          | MINDIR      | -      |
| `--modelFile=<MODELFILE>`   | 是       | MINDIR模型文件名（包括路径）                | -           | -      |
| `--outputFile=<OUTPUTFILE>` | 是       | 输出模型文件名（包括路径）自动生成`.ms`后缀 | -           | -      |
| `--trainModel=true`         | 是       | 是否是训练模式；如果要训练模型，必须为true  | true, false | false  |

> 参数名称和数值之间使用等号连接且不能有空格。

### 模型转换示例

假设待转换的模型文件为`my_model.mindir`，执行如下转换命令：

```bash
./converter_lite --fmk=MINDIR --trainModel=true --modelFile=my_model.mindir --outputFile=my_model
```

转换成功输出如下：

```text
CONVERTER RESULT SUCCESS:0
```

这表明 MindSpore 模型成功转换为 MindSpore 端侧模型，并生成了新文件`my_model.ms`。如果转换失败输出如下：

```text
CONVERT RESULT FAILED:
```

程序会返回的[错误码](https://www.mindspore.cn/doc/api_cpp/zh-CN/r1.1/errorcode_and_metatype.html)和错误信息。
