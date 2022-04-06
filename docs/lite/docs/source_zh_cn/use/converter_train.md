# 训练模型转换

`Linux` `环境准备` `模型导出` `模型转换` `中级` `高级`

<a href="https://gitee.com/mindspore/docs/blob/master/docs/lite/docs/source_zh_cn/use/converter_train.md" target="_blank"><img src="https://mindspore-website.obs.cn-north-4.myhuaweicloud.com/website-images/master/resource/_static/logo_source.png"></a>

## 概述

创建MindSpore端侧模型的步骤：

- 首先基于MindSpore架构使用Python创建网络模型，并导出为`.mindir`文件，参见云端的[保存模型](https://www.mindspore.cn/docs/programming_guide/zh-CN/master/save_model.html#mindir)。
- 然后将`.mindir`模型文件转换成`.ms`文件，`.ms`文件可以导入端侧设备并基于MindSpore端侧框架训练。

## Linux环境

### 环境准备

MindSpore Lite 模型转换工具提供了多个参数，目前工具仅支持Linux系统，环境准备步骤：

- [编译](https://www.mindspore.cn/lite/docs/zh-CN/master/use/build.html)或[下载](https://www.mindspore.cn/lite/docs/zh-CN/master/use/downloads.html)模型转换工具。
- 将转换工具需要的动态链接库加入环境变量LD_LIBRARY_PATH。

    ```bash
    export LD_LIBRARY_PATH=${PACKAGE_ROOT_PATH}/tools/converter/lib:${LD_LIBRARY_PATH}
    ```

    ${PACKAGE_ROOT_PATH}是编译或下载得到的包解压后的路径。

### 参数说明

下表为MindSpore Lite训练模型转换工具使用到的参数：

| 参数                        | 是否必选 | 参数说明                                    | 取值范围    | 默认值 |
| --------------------------- | -------- | ------------------------------------------- | ----------- | ------ |
| `--help`                    | 否       | 打印全部帮助信息                            | -           | -      |
| `--fmk=<FMK>`               | 是       | 输入模型的原始格式                          | MINDIR      | -      |
| `--modelFile=<MODELFILE>`   | 是       | MINDIR模型文件名（包括路径）                | -           | -      |
| `--outputFile=<OUTPUTFILE>` | 是       | 输出模型文件名（包括路径）自动生成`.ms`后缀 | -           | -      |
| `--trainModel=true`         | 是       | 是否是训练模式；如果要训练模型，必须为true  | true, false | false  |
| `--configFile=<CONFIGFILE>` | 否 | 1）可作为训练后量化配置文件路径；2）可作为扩展功能配置文件路径。  | - | -  |

> 参数名称和数值之间使用等号连接且不能有空格。
>
> `configFile`配置文件采用`key=value`的方式定义相关参数，量化相关的配置参数详见[训练后量化](https://www.mindspore.cn/lite/docs/zh-CN/master/use/post_training_quantization.html)。

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

程序会返回错误码和错误信息。
