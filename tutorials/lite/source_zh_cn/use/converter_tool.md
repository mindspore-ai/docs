# 转换为MindSpore Lite模型

[![查看源文件](../_static/logo_source.png)](https://gitee.com/mindspore/docs/blob/r1.0/tutorials/lite/source_zh_cn/use/converter_tool.md)

## 概述

MindSpore Lite提供离线转换模型功能的工具，支持多种类型的模型转换，转换后的模型可用于推理。命令行参数包含多种个性化选项，为用户提供方便的转换途径。

目前支持的输入格式有：MindSpore、TensorFlow Lite、Caffe和ONNX。

## Linux环境使用说明

### 环境准备

使用MindSpore Lite模型转换工具，需要进行如下环境准备工作。

- 编译：模型转换工具代码在MindSpore源码的`mindspore/lite/tools/converter`目录中，参考构建文档中的[环境要求](https://www.mindspore.cn/tutorial/lite/zh-CN/r1.0/use/build.html#id1)和[编译示例](https://www.mindspore.cn/tutorial/lite/zh-CN/r1.0/use/build.html#id3)编译x86_64版本。

- 运行：参考构建文档中的[编译输出](https://www.mindspore.cn/tutorial/lite/zh-CN/r1.0/use/build.html#id4)，获得`converter`工具，并配置环境变量。

### 使用示例

首先，在源码根目录下，输入命令进行编译，可参考`build.md`。
```bash
bash build.sh -I x86_64
```
> 目前模型转换工具仅支持x86_64架构。

下面选取了几个常用示例，说明转换命令的使用方法。

- 以Caffe模型LeNet为例，执行转换命令。

   ```bash
   ./converter_lite --fmk=CAFFE --modelFile=lenet.prototxt --weightFile=lenet.caffemodel --outputFile=lenet
   ```

   本例中，因为采用了Caffe模型，所以需要模型结构、模型权值两个输入文件。再加上其他必需的fmk类型和输出路径两个参数，即可成功执行。

   结果显示为：
   ```
   CONVERTER RESULT SUCCESS:0
   ```
   这表示已经成功将Caffe模型转化为MindSpore Lite模型，获得新文件`lenet.ms`。
   
- 以MindSpore、TensorFlow Lite、ONNX模型格式和感知量化模型为例，执行转换命令。

   - MindSpore模型`model.mindir`
      ```bash
      ./converter_lite --fmk=MINDIR --modelFile=model.mindir --outputFile=model
      ```
   
   - TensorFlow Lite模型`model.tflite`
      ```bash
      ./converter_lite --fmk=TFLITE --modelFile=model.tflite --outputFile=model
      ```
   
   - ONNX模型`model.onnx`
      ```bash
      ./converter_lite --fmk=ONNX --modelFile=model.onnx --outputFile=model
      ```

   - TensorFlow Lite感知量化模型`model_quant.tflite`
      ```bash
      ./converter_lite --fmk=TFLITE --modelFile=model_quant.tflite --outputFile=model --quantType=AwareTraining
      ```

   - 感知量化模型输入输出类型设置为float
   
       ```bash
      ./converter_lite --fmk=TFLITE --modelFile=model_quant.tflite --outputFile=model --quantType=AwareTraining --inferenceType=FLOAT
      ```
   以上几种情况下，均显示如下转换成功提示，且同时获得`model.ms`目标文件。
   
   ```
   CONVERTER RESULT SUCCESS:0
   ```
- 如果转换命令执行失败，程序会返回一个[错误码](https://www.mindspore.cn/doc/api_cpp/zh-CN/r1.0/errorcode_and_metatype.html)。

> 训练后量化示例请参考<https://www.mindspore.cn/tutorial/lite/zh-CN/r1.0/use/post_training_quantization.html>。

### 参数说明

MindSpore Lite模型转换工具提供了多种参数设置，用户可根据需要来选择使用。此外，用户可输入`./converter_lite --help`获取实时帮助。

下面提供详细的参数说明。

| 参数  |  是否必选   |  参数说明  | 取值范围 | 默认值 |
| -------- | ------- | ----- | --- | ---- |
| `--help` | 否 | 打印全部帮助信息。 | - | - |
| `--fmk=<FMK>`  | 是 | 输入模型的原始格式。 | MINDIR、CAFFE、TFLITE、ONNX | - |
| `--modelFile=<MODELFILE>` | 是 | 输入模型的路径。 | - | - |
| `--outputFile=<OUTPUTFILE>` | 是 | 输出模型的路径（不存在时将自动创建目录），不需加后缀，可自动生成`.ms`后缀。 | - | - |
| `--weightFile=<WEIGHTFILE>` | 转换Caffe模型时必选 | 输入模型weight文件的路径。 | - | - |
| `--quantType=<QUANTTYPE>` | 否 | 设置模型的量化类型。 | WeightQuant：训练后量化（权重量化）<br>PostTraining：训练后量化（全量化）<br>AwareTraining：感知量化 | - |
|` --inferenceType=<INFERENCETYPE>` | 否 | 设置感知量化模型输入输出数据类型，如果和原模型不一致则转换工具会在模型前后插转换算子，使得转换后的模型输入输出类型和inferenceType保持一致。 | UINT8、FLOAT、INT8 | FLOAT |
| `--stdDev=<STDDEV> `| 否 | 感知量化模型转换时用于设置输入数据的标准差。 | （0，+∞） | 128 |
| `--mean=<MEAN>` | 否 | 感知量化模型转换时用于设置输入数据的均值。 | [-128, 127] | -0.5 |
| `--bitNum=<BITNUM>` | 否 | 设定训练后量化（权重量化）的比特数，目前仅支持8bit量化 | 8 | 8 |
| `--quantSize=<QUANTSIZE>` | 否 | 设定参与训练后量化（权重量化）的卷积核尺寸阈值，若卷积核尺寸大于该值，则对此权重进行量化 |  （0，+∞） | 0 |
| `--convWeightQuantChannelThreshold=<CONVWEIGHTQUANTCHANNELTHRESHOLD>` | 否 | 设定参与训练后量化（权重量化）的卷积通道数阈值，若卷积通道数大于该值，则对此权重进行量化 | （0，+∞） | 16 | 
| `--config_file=<CONFIGFILE>` | 否 | 训练后量化（全量化）校准数据集配置文件路径  |  - | -  |


> - 参数名和参数值之间用等号连接，中间不能有空格。
> - Caffe模型一般分为两个文件：`*.prototxt`模型结构，对应`--modelFile`参数；`*.caffemodel`模型权值，对应`--weightFile`参数。

## Windows环境使用说明

### 环境准备

使用MindSpore Lite模型转换工具，需要进行如下环境准备工作。

- 获取工具包：下载Windows转换工具的[Zip包](https://www.mindspore.cn/versions)并解压至本地目录，获得`converter`工具。

### 参数说明

参考Linux环境模型转换工具的[参数说明](https://www.mindspore.cn/tutorial/lite/zh-CN/r1.0/use/converter_tool.html#id4)。

### 使用示例

设置日志打印级别为INFO。
```bash
set MSLOG=INFO
```

下面选取了几个常用示例，说明转换命令的使用方法。

- 以Caffe模型LeNet为例，执行转换命令。

   ```bash
   call converter_lite --fmk=CAFFE --modelFile=lenet.prototxt --weightFile=lenet.caffemodel --outputFile=lenet
   ```

   本例中，因为采用了Caffe模型，所以需要模型结构、模型权值两个输入文件。再加上其他必需的fmk类型和输出路径两个参数，即可成功执行。

   结果显示为：
   ```
   CONVERTER RESULT SUCCESS:0
   ```
   这表示已经成功将Caffe模型转化为MindSpore Lite模型，获得新文件`lenet.ms`。
   
- 以MindSpore、TensorFlow Lite、ONNX模型格式和感知量化模型为例，执行转换命令。

   - MindSpore模型`model.mindir`
      ```bash
      call converter_lite --fmk=MINDIR --modelFile=model.mindir --outputFile=model
      ```
   
   - TensorFlow Lite模型`model.tflite`
      ```bash
      call converter_lite --fmk=TFLITE --modelFile=model.tflite --outputFile=model
      ```
   
   - ONNX模型`model.onnx`
      ```bash
      call converter_lite --fmk=ONNX --modelFile=model.onnx --outputFile=model
      ```

   - TensorFlow Lite感知量化模型`model_quant.tflite`
      ```bash
      call converter_lite --fmk=TFLITE --modelFile=model_quant.tflite --outputFile=model --quantType=AwareTraining
      ```

   以上几种情况下，均显示如下转换成功提示，且同时获得`model.ms`目标文件。
   ```
   CONVERTER RESULT SUCCESS:0
   ```   
- 如果转换命令执行失败，程序会返回一个[错误码](https://www.mindspore.cn/doc/api_cpp/zh-CN/r1.0/errorcode_and_metatype.html)。
