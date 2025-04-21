# 三方ONNX模型对接自定义算子

[![查看源文件](https://mindspore-website.obs.cn-north-4.myhuaweicloud.com/website-images/r2.6.0rc1/resource/_static/logo_source.svg)](https://gitee.com/mindspore/docs/blob/r2.6.0rc1/docs/lite/docs/source_zh_cn/mindir/converter_custom.md)

## 概述

MindSpore Lite的[转换工具](https://www.mindspore.cn/lite/docs/zh-CN/r2.6.0rc1/converter/converter_tool.html)除了基本的模型转换功能之外，还支持对接自定义AscendC算子和自定义融合Pass，满足一些特殊场景对模型功能、性能的要求。

## 对接自定义算子

本教程介绍了MindSpore Lite如何对接三方ONNX模型中的自定义算子，来使能云侧转换和推理。本教程建立在已熟悉[转换工具](https://www.mindspore.cn/lite/docs/zh-CN/r2.6.0rc1/converter/converter_tool.html)的基础上。

对ONNX模型中的自定义算子节点有如下要求：

1. ONNX节点名为`Custom`，即ONNX的`NODE_PROPERTIES`中的`type`为`Custom`；

2. 必选属性`input_names`，数据类型为String[]，算子输入名和顺序要与自定义算子原型中的相同；

3. 必选属性`output_names`，数据类型为String[]，算子输出名和顺序要与自定义算子原型中的相同；

4. 必选属性`type`，数据类型为String，算子名要与自定义算子原型中的相同；

5. 可选属性`optional_input_names`，数据类型为String[]，可选输入名是必选输入名的子集；

6. 可选属性`output_num`，数据类型为int，算子有多输出的时且模型中仅使用第一个输出时需要指定；

7. 如果自定义算子原始属性是`bool`类型，但由于ONNX属性中无`bool`类型，所以需要以String类型的方式接入，值可以设置为`True`、`true`、`False`和`false`其中之一。

### 准备工作

1. 本地已安装了自定义算子包；

2. 已按照如上要求修改了ONNX文件中的Custom算子；

3. 已有MindSpore Lite云侧环境。详情请参考[转换工具](https://www.mindspore.cn/lite/docs/zh-CN/r2.6.0rc1/converter/converter_tool.html)。

### 模型转换与推理

按照[转换工具](https://www.mindspore.cn/lite/docs/zh-CN/r2.6.0rc1/converter/converter_tool.html)的流程即可。

## 对接自定义融合Pass

本教程介绍了MindSpore Lite如何编译、使用自定义融合Pass，来使能云侧转换和推理。本教程建立在已熟悉[端侧注册机制](https://www.mindspore.cn/lite/docs/zh-CN/r2.6.0rc1/advanced/third_party/converter_register.html)的基础上。

### 准备工作

1. 已有MindSpore Lite云侧环境。详情请参考[转换工具](https://www.mindspore.cn/lite/docs/zh-CN/r2.6.0rc1/converter/converter_tool.html)。

2. 已有ONNX文件。

3. 实现自定义Pass，代码请参考[example](https://gitee.com/mindspore/mindspore/tree/v2.6.0-rc1/mindspore/lite/examples/converter_acl_custom_pass)。

与[端侧注册机制](https://www.mindspore.cn/lite/docs/zh-CN/r2.6.0rc1/advanced/third_party/converter_register.html)不同的是，注册位置的参数需要更改。

```c++
// register customed Pass
using mindspore::registry::POSITION_ASCEND;
REG_PASS(PassTutorial, opt::PassTutorial)
REG_SCHEDULED_PASS(POSITION_ASCEND, {"PassTutorial"})
```

### 编译、转换、推理

编译流程详见[端侧注册机制](https://www.mindspore.cn/lite/docs/zh-CN/r2.6.0rc1/advanced/third_party/converter_register.html)，转换、推理详见[转换工具](https://www.mindspore.cn/lite/docs/zh-CN/r2.6.0rc1/converter/converter_tool.html)。
