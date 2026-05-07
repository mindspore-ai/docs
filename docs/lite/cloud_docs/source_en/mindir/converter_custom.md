# Adapting Custom Operators for Third-Party ONNX Models

[![View Source on AtomGit](https://mindspore-website.obs.cn-north-4.myhuaweicloud.com/website-images/r2.9.0/resource/_static/logo_source_en.svg)](https://atomgit.com/mindspore/docs/blob/r2.9.0/docs/lite/cloud_docs/source_en/mindir/converter_custom.md)

## Overview

In addition to basic model conversion functions, the [conversion tool](https://www.mindspore.cn/lite/docs/en/r2.9.0/converter/converter_tool.html) of MindSpore Lite supports adapting custom Ascend C operators and custom fusion passes to meet the functional and performance requirements of models in specific scenarios.

## Adapting Custom Operators

This tutorial describes how MindSpore Lite adapts custom operators in third-party ONNX models to enable cloud-side conversion and inference. This tutorial assumes that you are familiar with the [conversion tool](https://www.mindspore.cn/lite/docs/en/r2.9.0/converter/converter_tool.html).

The custom operator nodes in the ONNX model must meet the following requirements:

1. The ONNX node type is `Custom`, that is, the `type` field in `NODE_PROPERTIES` of the ONNX node is set to `Custom`;

2. Required attribute `input_names` of the String[] type. The input names and sequence must be the same as those defined in the custom operator prototype;

3. Required attribute `output_names` of the String[] type. The output names and sequence must be the same as those defined in the custom operator prototype;

4. Required attribute `type` of the String type. The operator name must be the same as that in the custom operator prototype;

5. Optional attribute `optional_input_names` of the String[] type. The optional input names are a subset of the required input names;

6. Optional attribute `output_num` of the int type. This attribute must be specified when the operator has multiple outputs but only the first output is used in the model;

7. If an original attribute of the custom operator is of the `bool` type, the attribute must be transferred as the String type in the ONNX model (because ONNX does not support the bool attribute type). The value can be set to one of `True`, `true`, `False`, and `false`.

### Prerequisites

1. The custom operator package has been installed locally.

2. The Custom operators in the ONNX file have been modified according to the preceding requirements.

3. The MindSpore Lite cloud-side environment is available. For details, see the [conversion tool](https://www.mindspore.cn/lite/docs/en/r2.9.0/converter/converter_tool.html) documentation.

### Model Conversion and Inference

Perform the operations by following the procedure provided in the [conversion tool](https://www.mindspore.cn/lite/docs/en/r2.9.0/converter/converter_tool.html) documentation.

## Adapting Custom Fusion Passes

This tutorial describes how MindSpore Lite compiles and uses custom fusion passes to enable cloud-side conversion and inference. This tutorial assumes that you are familiar with the [device-side registration mechanism](https://www.mindspore.cn/lite/docs/en/r2.9.0/advanced/third_party/converter_register.html).

### Prerequisites

1. The MindSpore Lite cloud-side environment is available. For details, see the [conversion tool](https://www.mindspore.cn/lite/docs/en/r2.9.0/converter/converter_tool.html) documentation.

2. The ONNX model file is available.

3. The custom pass is implemented. For sample code, see [example](https://atomgit.com/mindspore/mindspore-lite/tree/r2.9/mindspore-lite/examples/converter_acl_custom_pass).

Different from the [device-side registration mechanism](https://www.mindspore.cn/lite/docs/en/r2.9.0/advanced/third_party/converter_register.html), the parameters for the registration position need to be modified:

```c++
// register custom Pass
using mindspore::registry::POSITION_ASCEND;
REG_PASS(PassTutorial, opt::PassTutorial)
REG_SCHEDULED_PASS(POSITION_ASCEND, {"PassTutorial"})
```

### Compilation, Conversion, and Inference

For the compilation process, see the [device-side registration mechanism](https://www.mindspore.cn/lite/docs/en/r2.9.0/advanced/third_party/converter_register.html) documentation.
For model conversion and inference, see the [conversion tool](https://www.mindspore.cn/lite/docs/en/r2.9.0/converter/converter_tool.html) documentation.
