# Building Custom Operators Offline

[![View Source On Gitee](https://mindspore-website.obs.cn-north-4.myhuaweicloud.com/website-images/r2.6.0rc1/resource/_static/logo_source_en.svg)](https://gitee.com/mindspore/docs/blob/r2.6.0rc1/docs/lite/docs/source_en/advanced/third_party/converter_register.md)

## Overview

Our [Conversion Tool](https://www.mindspore.cn/lite/docs/en/r2.6.0rc1/converter/converter_tool.html) is a highly flexible tool. In addition to the basic ability of model converter, we have designed a set of registration mechanism, which allows users to expand, including node-parse extension, model-parse extension and graph-optimization extension. The users can combined them as needed to achieve their own intention.

node-parse extension: The users can define the process to parse a certain node of a model by themselves, which only support ONNX, CAFFE, TF and TFLITE. The related interface is [NodeParser](https://www.mindspore.cn/lite/api/en/r2.6.0rc1/generate/classmindspore_converter_NodeParser.html), [NodeParserRegistry](https://www.mindspore.cn/lite/api/en/r2.6.0rc1/generate/classmindspore_registry_NodeParserRegistry.html).
model-parse extension: The users can define the process to parse a model by themselves, which only support ONNX, CAFFE, TF and TFLITE. The related interface is [ModelParser](https://www.mindspore.cn/lite/api/en/r2.6.0rc1/generate/classmindspore_converter_ModelParser.html), [ModelParserRegistry](https://www.mindspore.cn/lite/api/en/r2.6.0rc1/generate/classmindspore_registry_ModelParserRegistry.html).
graph-optimization extension: After parsing a model, a graph structure defined by MindSpore will show up and then, the users can define the process to optimize the parsed graph. The related interfaces are [PassBase](https://www.mindspore.cn/lite/api/en/r2.6.0rc1/generate/classmindspore_registry_PassBase.html), [PassPosition](https://mindspore.cn/lite/api/en/r2.6.0rc1/generate/enum_mindspore_registry_PassPosition-1.html), [PassRegistry](https://www.mindspore.cn/lite/api/en/r2.6.0rc1/generate/classmindspore_registry_PassRegistry.html).

> The node-parse extension needs to rely on the flatbuffers, protobuf and the serialization files of third-party frameworks, at the same time, the version of flatbuffers and the protobuf needs to be consistent with that of the released package, the serialized files must be compatible with that used by the released package. Note that the flatbuffers, protobuf and the serialization files are not provided in the released package, users need to compile and generate the serialized files by themselves. The users can obtain the basic information about [flabuffers](https://gitee.com/mindspore/mindspore/blob/v2.6.0-rc1/cmake/external_libs/flatbuffers.cmake), [probobuf](https://gitee.com/mindspore/mindspore/blob/v2.6.0-rc1/cmake/external_libs/protobuf.cmake), [ONNX prototype file](https://gitee.com/mindspore/mindspore/tree/v2.6.0-rc1/third_party/proto/onnx), [CAFFE prototype file](https://gitee.com/mindspore/mindspore/tree/v2.6.0-rc1/third_party/proto/caffe), [TF prototype file](https://gitee.com/mindspore/mindspore/tree/v2.6.0-rc1/third_party/proto/tensorflow) and [TFLITE prototype file](https://gitee.com/mindspore/mindspore/blob/v2.6.0-rc1/mindspore/lite/tools/converter/parser/tflite/schema.fbs) from the [MindSpore WareHouse](https://gitee.com/mindspore/mindspore/tree/v2.6.0).
>
> MindSpore Lite alse providers a series of registration macros to facilitate user access. These macros include node-parse registration [REG_NODE_PARSER](https://www.mindspore.cn/lite/api/en/r2.6.0rc1/generate/define_node_parser_registry.h_REG_NODE_PARSER-1.html), model-parse registration [REG_MODEL_PARSER](https://www.mindspore.cn/lite/api/en/r2.6.0rc1/generate/define_model_parser_registry.h_REG_MODEL_PARSER-1.html), graph-optimization registration [REG_PASS](https://www.mindspore.cn/lite/api/en/r2.6.0rc1/generate/define_pass_registry.h_REG_PASS-1.html) and graph-optimization scheduled registration [REG_SCHEDULED_PASS](https://www.mindspore.cn/lite/api/en/r2.6.0rc1/generate/define_pass_registry.h_REG_SCHEDULED_PASS-1.html)

The expansion capability of MindSpore Lite conversion tool only support on Linux system currently.

In this chapter, we will show the users a sample of extending MindSpore Lite converter tool, covering the example of expanding node, example of optimizing graph, compiling and linking. The example will help the users understand the extension ability as soon as possible.

> Due to that model-parse extension is a modular extension ability, the chapter will not introduce in details. However, we still provide the users with a simplified unit case for inference.

The chapter takes a [add.tflite](https://download.mindspore.cn/model_zoo/official/lite/quick_start/add.tflite), which only includes an opreator of adding, as an example. We will show the users how to convert the single operator of adding to that of [Custom](https://www.mindspore.cn/lite/docs/en/r2.6.0rc1/advanced/third_party/register_kernel.html#custom-operators) and finally obtain a model which only includs a single operator of custom.

The code related to the example can be obtained from the path [mindspore/lite/examples/converter_extend](https://gitee.com/mindspore/mindspore/tree/v2.6.0-rc1/mindspore/lite/examples/converter_extend).

## Node Extension

1. Self-defined node-parse: The users need to inherit the base class [NodeParser](https://www.mindspore.cn/lite/api/en/r2.6.0rc1/generate/classmindspore_converter_NodeParser.html), and then, choose a interface to override according to model frameworks.

2. Node-parse Registration: The users can directly call the registration interface [REG_NODE_PARSER](https://www.mindspore.cn/lite/api/en/r2.6.0rc1/generate/define_node_parser_registry.h_REG_NODE_PARSER-1.html), so that the self-defined node-parse will be registered in the converter tool of MindSpore Lite.

```c++
class AddParserTutorial : public NodeParser {  // inherit the base class
 public:
  AddParserTutorial() = default;
  ~AddParserTutorial() = default;
  ops::PrimitiveC *Parse(const std::unique_ptr<tflite::OperatorT> &tflite_op,            // override interface
                         const std::unique_ptr<tflite::SubGraphT> &tflite_subgraph,
                         const std::unique_ptr<tflite::ModelT> &tflite_model) override;
};

REG_NODE_PARSER(kFmkTypeTflite, ADD, std::make_shared<AddParserTutorial>());     // call the registration macro
```

For the sample code, please refer to [node_parser](https://gitee.com/mindspore/mindspore/tree/v2.6.0-rc1/mindspore/lite/examples/converter_extend/node_parser).

## Model Extension

For the sample code, please refer to the unit case [ModelParserRegistryTest](https://gitee.com/mindspore/mindspore/blob/v2.6.0-rc1/mindspore/lite/test/ut/tools/converter/registry/model_parser_registry_test.cc).

### Optimization Extension

1. Self-defined Pass: The users need to inherit the base class [PassBase](https://www.mindspore.cn/lite/api/en/r2.6.0rc1/generate/classmindspore_registry_PassBase.html), and override the interface function [Execute](https://www.mindspore.cn/lite/api/en/r2.6.0rc1/generate/classmindspore_dataset_Execute.html).

2. Pass Registration: The users can directly call the registration interface [REG_PASS](https://www.mindspore.cn/lite/api/en/r2.6.0rc1/generate/define_pass_registry.h_REG_PASS-1.html), so that the self-defined pass can be registered in the converter tool of MindSpore Lite.

```c++
class PassTutorial : public registry::PassBase {  // inherit the base class
 public:
  PassTutorial() : PassBase("PassTutorial") {}

  ~PassTutorial() = default;

  bool Execute(const api::FuncGraphPtr &func_graph) override;     // override interface

 private:
  AnfNodePtr CreateCustomOp(const api::FuncGraphPtr func_graph, const CNodePtr &cnode);
};

using mindspore::registry::POSITION_BEGIN;            // choose a scheduling position
REG_PASS(PassTutorial, opt::PassTutorial)             // register PassBase's subclass
REG_SCHEDULED_PASS(POSITION_BEGIN, {"PassTutorial"})  // register scheduling logic
```

For the sample code, please refer to [pass](https://gitee.com/mindspore/mindspore/tree/v2.6.0-rc1/mindspore/lite/examples/converter_extend/pass).

> In the offline phase of conversion, we will infer the basic information of output tensors of each node of the model, including the format, data type and shape. So, in this phase, users need to provide the inferring process of self-defined operator. Here, users can refer to [Operator Infershape Extension](https://www.mindspore.cn/lite/docs/en/r2.6.0rc1/infer/runtime_cpp.html#operator-infershape-extension).

## Example

### Compile

- Environment Requirements

    - System environment: Linux x86_64; Recommend Ubuntu 18.04.02LTS
    - compilation dependencies:
        - [CMake](https://cmake.org/download/) >= 3.18.3
        - [GCC](https://gcc.gnu.org/releases.html) >= 7.3.0

- Compilation preparation

  The release package of MindSpore Lite doesn't provide serialized files of other frameworks, therefore, users need to compile and obtain by yourselves. Here, please refer to [Overview](https://www.mindspore.cn/lite/docs/en/r2.6.0rc1/advanced/third_party/converter_register.html#overview).

  The case is a tflite model, users need to compile [flatbuffers](https://gitee.com/mindspore/mindspore/blob/v2.6.0-rc1/cmake/external_libs/flatbuffers.cmake) and combine the [TFLITE Proto File](https://gitee.com/mindspore/mindspore/blob/v2.6.0-rc1/mindspore/lite/tools/converter/parser/tflite/schema.fbs) to generate the serialized file.

  After generating, users need to create a directory `schema` under the directory of `mindspore/lite/examples/converter_extend` and then place the serialized file in it.

- Compilation and Build

  Execute the script [build.sh](https://gitee.com/mindspore/mindspore/blob/v2.6.0-rc1/mindspore/lite/examples/converter_extend/build.sh) in the directory of `mindspore/lite/examples/converter_extend`. And then, the released package of MindSpore Lite will be downloaded and the demo will be compiled automatically.

  ```bash
  bash build.sh
  ```

  > If the automatic download is failed, users can download the specified package manually, of which the hardware platform is CPU and the system is Ubuntu-x64 [mindspore-lite-{version}-linux-x64.tar.gz](https://www.mindspore.cn/lite/docs/en/r2.6.0rc1/use/downloads.html), After unzipping, please copy the directory of `tools/converter/lib` and `tools/converter/include` to the directory of `mindspore/lite/examples/converter_extend`.
  >
  > After manually downloading and storing the specified file, users need to execute the `build.sh` script to complete the compilation and build process.

- Compilation Result

  The dynamic library `libconverter_extend_tutorial.so` will be generated in the directory of `mindspore/lite/examples/converter_extend/build`.

### Executing Program

1. Copy library

   Copy the dynamic library `libconverter_extend_tutorial.so` to the directory of `tools/converter/lib` of the released package.

2. Enter the conversion directory of the released package.

   ```bash
   cd ${PACKAGE_ROOT_PATH}/tools/converter/converter
   ```

3. Create extension configuration file(converter.cfg, please refer to [Extension Configuration](#extension-configuration)), the content is as follows:

   ```text
   [registry]
   plugin_path=libconverter_extend_tutorial.so      # users need to configure the correct path of the dynamic library
   ```

4. Add the required dynamic library to the environment variable `LD_LIBRARY_PATH`

   ```bash
   export LD_LIBRARY_PATH=${LD_LIBRARY_PATH}:${PACKAGE_ROOT_PATH}/tools/converter/lib
   ```

5. Execute the script

   ```bash
   ./converter_lite --fmk=TFLITE --modelFile=add.tflite --configFile=converter.cfg --outputFile=add_extend
   ```

The model file `add_extend.ms` will be generated, the place of which is up to the parameter `outputFile`.

## Extension Configuration

To load the extension module when converting, users need to configure the path of extended dynamic library. The parameters related to the extension include `plugin_path`, `disable_fusion`, `fusion_blacklists`. The detailed description of the parameters is as follows:

| Parameter         | Attribute | Function Description                         | Parameter Type | Default Value | Value Range                                             |
| ----------------- | --------- | -------------------------------------------- | -------------- | ------------- | ------------------------------------------------------- |
| plugin_path       | Optional  | Third-party library path                     | String         | -             | If there are more than one, please use `;` to separate. |
| disable_fusion    | Optional  | Indicate whether to close fusion             | String         | off           | off or on.                                              |
| fusion_blacklists | Optional  | Specified fusion operator names to be closed | String         | -             | If there are more than one, please use `,` to separate  |

We have generated the default configuration file (converter.cfg). The content is as follows:

```ini
[registry]
plugin_path=libconverter_extend_tutorial.so      # users need to configure the correct path of the dynamic library
```

If the user needs to turn off the specified operator fusions, the fusion configuration of the the specified operator names to be closed are as follows:

```ini
[registry]
# When parameter `disable_fusion` is configured as `off`, the user can turn off the specified operator fusions by configuring parameter `fusion_blacklists`. While parameter `disable_fusion` is configured as `on`, the parameter `fusion_blacklists` does not work.
disable_fusion=off
fusion_blacklists=ConvActivationFusion,MatMulActivationFusion
```

The operator fusion names are as follows:

| No   | the operator fusion name             |
| ---- | ------------------------------------ |
| 1    | AddConcatActivationFusion            |
| 2    | SqueezeFusion                        |
| 3    | TransposeFusion                      |
| 4    | ReshapeReshapeFusion                 |
| 5    | ConvBiasaddFusion                    |
| 6    | ConvBatchNormFusion                  |
| 7    | ConvScaleFusion                      |
| 8    | GroupNormFusion                      |
| 9    | TfNormFusion                         |
| 10   | OnnxLayerNormFusion                  |
| 11   | OnnxLayerNormFusion2                 |
| 12   | BatchMatMulFusion                    |
| 13   | BatchNormToScaleFusion               |
| 14   | SigmoidMulFusion                     |
| 15   | ActivationFusion                     |
| 16   | ConvActivationFusion                 |
| 17   | ConvTupleGetItemFusion               |
| 18   | ConvTupleActivationFusion            |
| 19   | TfliteLstmCellFusion                 |
| 20   | TfLstmCellFusion                     |
| 21   | TfBidirectionGruFusion               |
| 22   | TfGeLUFusion                         |
| 23   | OnnxGeLUFusion                       |
| 24   | TfliteRelPosMultiHeadAttentionFusion |
| 25   | GLUFusion                            |
| 26   | ConstFoldPass                        |
| 27   | AffineFusion                         |
| 28   | AffineActivationFusion               |
| 29   | ConvConvFusion                       |
| 30   | ConvPadFusion                        |
| 31   | MatMulAddFusion                      |
| 32   | MatMulMulFusion                      |
| 33   | TransposeMatMulFusion                |
| 34   | MulAddFusion                         |
| 35   | ScaleActivationFusion                |
| 36   | ScaleScaleFusion                     |
| 37   | FullConnectedFusion                  |
| 38   | FullconnectedAddFusion               |
| 39   | TensorDotFusion                      |
| 40   | MatMulActivationFusion               |

