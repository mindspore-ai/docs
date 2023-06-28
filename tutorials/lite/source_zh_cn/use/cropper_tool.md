# 使用裁剪工具降低库文件大小

`Linux` `环境准备` `静态库裁剪` `中级` `高级`

<a href="https://gitee.com/mindspore/docs/blob/r1.1/tutorials/lite/source_zh_cn/use/cropper_tool.md" target="_blank"><img src="../_static/logo_source.png"></a>

## 概述

MindSpore Lite提供对Runtime的`libmindspore-lite.a`静态库裁剪工具，能够筛选出`ms`模型中存在的算子，对静态库文件进行裁剪，有效降低库文件大小。

裁剪工具运行环境是x86_64，目前支持对CPU算子的裁剪，即编译方式为`bash build.sh -I arm64 -e cpu`、`bash build.sh -I arm32 -e cpu`、`bash build.sh -I x86_64 -e cpu`中的`libmindspore-lite.a`静态库。

## 环境准备

使用MindSpore Lite裁剪工具，需要进行如下环境准备工作。

- 编译：裁剪工具代码在MindSpore源码的`mindspore/lite/tools/cropper`目录中，参考构建文档中的[环境要求](https://www.mindspore.cn/tutorial/lite/zh-CN/r1.1/use/build.html#id1)和[编译示例](https://www.mindspore.cn/tutorial/lite/zh-CN/r1.1/use/build.html#id3)编译x86_64版本。

- 运行：参考构建文档中的[编译输出](https://www.mindspore.cn/tutorial/lite/zh-CN/r1.1/use/build.html#id4)，获得`cropper`工具。

## 参数说明

使用裁剪工具进行静态库的裁剪，其命令格式如下所示。

```bash
./cropper [--packageFile=<PACKAGEFILE>] [--configFile=<CONFIGFILE>]
          [--modelFile=<MODELFILE>] [--modelFolderPath=<MODELFOLDERPATH>]
          [--outputFile=<MODELFILE>] [--help]
```

下面提供详细的参数说明。

| 参数                                  | 是否必选 | 参数说明                                                     | 参数类型 | 默认值 | 取值范围 |
| ------------------------------------- | -------- | ------------------------------------------------------------ | -------- | ------ | -------- |
| `--packageFile=<PACKAGEFILE>`         | 是       | 需要裁剪的`libmindspore-lite.a`文件路径。                    | String   | -      | -        |
| `--configFile=<CONFIGFILE>`           | 是       | 裁剪工具配置文件的路径，裁剪CPU库需要设置`cropper_mapping_cpu.cfg`文件路径。 | String   | -      | -        |
| `--modelFolderPath=<MODELFOLDERPATH>` | 否       | 模型文件夹路径，根据文件夹中存在的所有`ms`模型进行库裁剪。`modelFile`和`modelFolderPath`参数必须二选一。 | String   | -      | -        |
| `--modelFile=<MODELFILE>`             | 否       | 模型文件路径，根据指定的`ms`模型文件进行库裁剪，多个模型文件采用`,`分割。`modelFile`和`modelFolderPath`参数必须二选一。 | String   | -      | -        |
| `--outputFile=<OUTPUTFILE>`           | 否       | 裁剪完成的`libmindspore-lite.a`库的保存路径，默认覆盖源文件。 | String   | -      | -        |
| `--help`                              | 否       | 打印全部帮助信息。                                           | -        | -      | -        |

> 配置文件`cropper_mapping_cpu.cfg`存在于`mindspore-lite-{version}-inference-linux-x64`包中的`cropper`目录。

## 使用示例

裁剪工具通过解析`ms`模型得到算子列表，并根据配置文件`configFile`中的映射关系来裁剪`libmindspore-lite.a`静态库。模型文件传入方式包括文件夹、文件两种：

- 通过文件夹的方式传入`ms`模型，将模型文件所在的文件夹路径传递给`modelFolderPath`参数，对arm64-cpu的`libmindspore-lite.a`静态库进行裁剪。

```bash
./cropper --packageFile=/mindspore-lite-{version}-inference-android-aarch64/lib/libmindspore-lite.a --configFile=./cropper_mapping_cpu.cfg --modelFolderPath=/model --outputFile=/mindspore-lite/lib/libmindspore-lite.a
```

本例将读取`/model`文件夹中包含的所有`ms`模型，对arm64-cpu的`libmindspore-lite.a`静态库进行裁剪，并将裁剪后的`libmindspore-lite.a`静态库保存到`/mindspore-lite/lib/`目录。

- 通过文件的方式传入`ms`模型，将模型文件所在的路径传递给`modelFile`参数，对arm64-cpu的`libmindspore-lite.a`静态库进行裁剪。

```bash
./cropper --packageFile=/mindspore-lite-{version}-inference-android-aarch64/lib/libmindspore-lite.a --configFile=./cropper_mapping_cpu.cfg --modelFile=/model/lenet.ms,/model/retinaface.ms  --outputFile=/mindspore-lite/lib/libmindspore-lite.a
```

本例将根据`modelFile`传入的`ms`模型，对arm64-cpu的`libmindspore-lite.a`静态库进行裁剪，并将裁剪后的`libmindspore-lite.a`静态库保存到`/mindspore-lite/lib/`目录。
