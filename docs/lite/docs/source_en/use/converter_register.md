# Construct custom kernel by registering conversion tool

`Linux` `Model Converting` `Intermediate` `Expert`

<a href="https://gitee.com/mindspore/docs/blob/master/docs/lite/docs/source_en/use/converter_register.md" target="_blank"><img src="https://mindspore-website.obs.cn-north-4.myhuaweicloud.com/website-images/master/resource/_static/logo_source_en.png"></a>

## Overview

Our [Conversion Tool](https://www.mindspore.cn/lite/docs/en/master/use/converter_tool.html) is a highly flexible tool. In addition to the basic ability of model converter, we have designed a set of registration mechanism, which allows users to expand, including node-parse extension, model-parse extension and graph-optimization extension. The users can combined them as needed to achieve their own intention.

node-parse extension: The users can define the process to parse a certain node of a model by themselves, which only support ONNX, CAFFE, TF and TFLITE. The related interface is [NodeParser](https://www.mindspore.cn/lite/api/en/master/api_cpp/mindspore_converter.html#nodeparser), [NodeParserRegistry](https://www.mindspore.cn/lite/api/en/master/api_cpp/mindspore_registry.html#nodeparserregistry).
model-parse extension: The users can define the process to parse a model by themselves, which only support ONNX, CAFFE, TF and TFLITE. The related interface is [ModelParser](https://www.mindspore.cn/lite/api/en/master/api_cpp/mindspore_converter.html#modelparser), [ModelParserRegistry](https://www.mindspore.cn/lite/api/en/master/api_cpp/mindspore_registry.html#modelparserregistry).
graph-optimization extension: After parsing a model, a graph structure defined by MindSpore will show up and then, the users can define the process to optimize the parsed graph. The related interface is [PassBase](https://www.mindspore.cn/lite/api/en/master/api_cpp/mindspore_registry.html#passbase), [PassPosition](https://www.mindspore.cn/lite/api/en/master/api_cpp/mindspore_registry.html#passposition)、[PassRegistry](https://www.mindspore.cn/lite/api/en/master/api_cpp/mindspore_registry.html#passregistry).

> The node-parse extension needs to rely on the flatbuffers, protobuf and the serialization files of third-party frameworks, at the same time, the version of flatbuffers and the protobuf needs to be consistent with that of the released package, the serialized files must be compatible with that used by the released package. Note that the flatbuffers, protobuf and the serialization files are not provided in the released package, users need to compile and generate the serialized files by themselves. The users can obtain the basic information about [flabuffers](https://gitee.com/mindspore/mindspore/blob/master/cmake/external_libs/flatbuffers.cmake), [probobuf](https://gitee.com/mindspore/mindspore/blob/master/cmake/external_libs/protobuf.cmake), [ONNX prototype file](https://gitee.com/mindspore/mindspore/tree/master/third_party/proto/onnx), [CAFFE prototype file](https://gitee.com/mindspore/mindspore/tree/master/third_party/proto/caffe), [TF prototype file](https://gitee.com/mindspore/mindspore/tree/master/third_party/proto/tensorflow) and [TFLITE prototype file](https://gitee.com/mindspore/mindspore/blob/master/mindspore/lite/tools/converter/parser/tflite/schema.fbs) from the [MindSpore WareHouse](https://gitee.com/mindspore/mindspore/tree/master).

MindSpore Lite alse providers a series of registration macros to facilitate user access. These macros include node-parse registration [REG_NODE_PARSER](https://www.mindspore.cn/lite/api/en/master/api_cpp/mindspore_registry.html#reg-node-parser), model-parse registration [REG_MODEL_PARSER](https://www.mindspore.cn/lite/api/en/master/api_cpp/mindspore_registry.html#reg-model-parser), graph-optimization registration [REG_PASS](https://www.mindspore.cn/lite/api/en/master/api_cpp/mindspore_registry.html#reg-pass) and graph-optimization scheduled registration [REG_SCHEDULED_PASS](https://www.mindspore.cn/lite/api/en/master/api_cpp/mindspore_registry.html#reg-scheduled-pass)

The expansion capability of MindSpore Lite conversion tool only support on Linux system currently.

In this chapter, we will show the users a sample of extending Mindspore Lite converter tool, covering the example of expanding node, example of optimizing graph, compiling and linking. The example will help the users understand the extension ability as soon as possible.

> Due to that model-parse extension is a modular extension ability, the chapter will not introduce in details. However, we still provide the users with a simplified unit case for inference.

The chapter takes a [add.tflite](https://download.mindspore.cn/model_zoo/official/lite/quick_start/add.tflite), which only includes an opreator of adding, as an example. We will show the users how to convert the single operator of adding to that of [Custom](https://www.mindspore.cn/lite/docs/en/master/use/register_kernel.html#custom) and finally, obtain a model which only includs a single operator of custom.

The code related to the example can be obtained from the directory [mindspore/lite/examples/converter_extend](https://gitee.com/mindspore/mindspore/tree/master/mindspore/lite/examples/converter_extend).

## Node Extension

1. Self-defined node-parse: The users need to inherit the base class [NodeParser](https://www.mindspore.cn/lite/api/en/master/api_cpp/mindspore_converter.html#nodeparser), and then, choose a interface to override according to model frameworks.

2. Node-parse Registration: The users can directly call the registration interface [REG_NODE_PARSER](https://www.mindspore.cn/lite/api/en/master/api_cpp/mindspore_registry.html#reg-node-parser), so that the self-defined node-parse will be registered in the converter tool of MindSpore Lite.

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

The sample code, please refer to [node_parser](https://gitee.com/mindspore/mindspore/tree/master/mindspore/lite/examples/converter_extend/node_parser).

## Model Extension

The sample code, please refer to the unit case [ModelParserRegistryTest](https://gitee.com/mindspore/mindspore/blob/master/mindspore/lite/test/ut/tools/converter/registry/model_parser_registry_test.cc)

## Optimization Extension

1. Self-defined Pass: The users need to inherit the base class [PassBase](https://www.mindspore.cn/lite/api/en/master/api_cpp/mindspore_registry.html#passbase), and override the interface function [Execute](https://www.mindspore.cn/lite/api/en/master/api_cpp/mindspore_registry.html#execute)。

2. Pass Registration: The users can directly call the registration interface [REG_PASS](https://www.mindspore.cn/lite/api/en/master/api_cpp/mindspore_registry.html#reg-pass), so that the self-defined pass can be registered in the converter tool of MindSpore Lite.

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

The sample code, please refer to [pass](https://gitee.com/mindspore/mindspore/tree/master/mindspore/lite/examples/converter_extend/pass)。

> In the offline phase of conversion, we will infer the basic information of output tensors of each node of the model, including the format, data type and shape. So, in this phase, users need to provide the inferring process of self-defined operator. Here, users can refer to [Operator Infershape Extension](https://www.mindspore.cn/lite/docs/en/master/use/runtime_cpp.html#operator-infershape-extension)。

## Example

### Compile

- Environment Requirements

    - System environment: Linux x86_64; Recommend Ubuntu 18.04.02LTS
    - compilation dependencies:
        - [CMake](https://cmake.org/download/) >= 3.18.3
        - [GCC](https://gcc.gnu.org/releases.html) >= 7.3.0

- Compilation preparation

  The release package of MindSpore Lite doesn't provide serialized files of other frameworks, therefore, users need to compile and obtain by yourselves. Here, please refer to [Overview](https://www.mindspore.cn/lite/docs/en/master/use/converter_register.html#overview).

  The case is a tflite model, users need to compile [flatbuffers](https://gitee.com/mindspore/mindspore/blob/master/cmake/external_libs/flatbuffers.cmake) and combine the [TFLITE Proto File](https://gitee.com/mindspore/mindspore/blob/master/mindspore/lite/tools/converter/parser/tflite/schema.fbs) to generate the serialized file.

  After generating, users need to create a directory `schema` under the directory of `mindspore/lite/examples/converter_extend` and then place the serialized file in it.

- Compilation and Build

  Execute the script [build.sh](https://gitee.com/mindspore/mindspore/blob/master/mindspore/lite/examples/converter_extend/build.sh) in the directory of `mindspore/lite/examples/converter_extend`. And then, the released package of Mindspore Lite will be downloaded and the demo will be compiled automatically.

  ```bash
  bash build.sh
  ```

  > If the automatic download is failed, users can download the specified package manually, of which the hardware platform is CPU and the system is Ubuntu-x64 [mindspore-lite-{version}-linux-x64.tar.gz](https://www.mindspore.cn/lite/docs/en/master/use/downloads.html), After unzipping, please copy the directory of `tools/converter/lib` and `tools/converter/include` to the directory of `mindspore/lite/examples/converter_extend`.
  >
  > After manually downloading and storing the specified file, users need to execute the `build.sh` script to complete the compilation and build process.

- Compilation Result

  The dynamic library `libconverter_extend_tutorial.so` will be generated in the directory of `mindspore/lite/examples/converter_extend/build`.

### Execute Program

1. Copy library

   Copy the dynamic library `libconverter_extend_tutorial.so` to the directory of `tools/converter/lib` of the released package.

2. Enter the conversion directory of the released package.

   ```bash
   cd ${PACKAGE_ROOT_PATH}/tools/converter/converter
   ```

3. Create extension configuration file(converter.cfg), the content is as follows:

   ```text
   [registry]
   plugin_path=libconverter_extend_tutorial.so      # users need to configure the correct path of the dynamic library
   ```

4. Add the required dynamic library to the environment variable `LD_LIBRARY_PATH`

   ```bash
   export LD_LIBRARY_PATH=${LD_LIBRARY_PATH}:/tools/converter/lib
   ```

5. Execute the script

   ```bash
   ./converter_lite --fmk=TFLITE --modelFile=add.tflite --configFile=converter.cfg --outputFile=add_extend
   ```

The model file `add_extend.ms` will be generated, the place of which is up to the parameter `outputFile`.
