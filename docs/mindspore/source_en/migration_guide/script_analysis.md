# Network Script Analysis

<a href="https://gitee.com/mindspore/docs/blob/master/docs/mindspore/migration_guide/source_en/script_analysis.md" target="_blank"><img src="https://mindspore-website.obs.cn-north-4.myhuaweicloud.com/website-images/master/resource/_static/logo_source_en.png"></a>

## Operator Evaluation

### MindSpore Operator Design

The process of using the MindSpore framework to build a neural network is similar to other frameworks (TensorFlow/PyTorch), but the supported operators are different. It is necessary to find out the missing operators in the MindSpore framework when performing network migration (e.g., migrating from TensorFlow to the MindSpore-ascend platform).

MindSpore API is composed of various Python/C++ API operators, which can be roughly divided into:

- Data framework operator

  Including tensors, basic data types, training gradients, optimizer operators, such as `mindspore.int32`, `mindspore.nn.Cell`, etc.

- Data preprocessing operator

  Including image reading, data type conversion operators, such as `mindspore.dataset.MnistDataset`, etc.

- Network structure operator

  Including convolution and normalization operators used in network construction, such as `mindspore.nn.Conv2d`, `mindspore.nn.Dense`, etc.

The surface layer of the network structure operator is the ME operator, which is the operator API called by the user (e.g., `mindspore.nn.Softmax`), and the ME operator is implemented by calling the TBE operator (C/C++) at the bottom layer.

When counting missing ME operators, you need to find out the corresponding operators of all operators (including data framework classes, data preprocessing, and network structure operators) in the source script in the MindSpore framework (e.g.,`tf.nn.relu` corresponds to MindSpore operator `mindspore.nn.ReLU`). If there is no corresponding operator in MindSpore, it will be counted as missing.

### Querying Operator Mapping Table

Find the network structure and the Python file that implements the training function in the code library (the name is generally train.py model.py, etc.), and find all relevant operators in the script file (including data framework classes, data preprocessing, network structure operators, etc.), and compare with [MindSpore Operator API](https://www.mindspore.cn/docs/en/master/note/operator_list_ms.html) , to find the platform support status of the operator under `mindspore.nn` or `mindspore.ops`.

If the corresponding ME operator cannot be found on this webpage, you can continue to search for the operator name in [MindSpore API List](https://www.mindspore.cn/docs/en/master/index.html).

If the source code is a PyTorch script, you can directly query [MindSpore and PyTorch operator mapping](https://www.mindspore.cn/docs/en/master/note/index.html#operator_api) to find the corresponding MindSpore operator. For the mapping of other frame operators, please refer to the operator naming and function description. Note that for operators with the same function, MindSpore may define a name for this operator differing from other frameworks, and the parameters and functions of operators with the same name may also be different from other frameworks. Please refer to the official description for checking the names.

### Missing Operator Processing Strategy

1. Consider replacing it with other operators: It is necessary to analyze the implementation formula of the operator and examine whether the existing MindSpore operator can be superimposed to achieve the expected goal.
3. Consider using Customized operators: see [Custom Operators (Custom based)](https://www.mindspore.cn/docs/programming_guide/en/master/custom_operator_custom.html).
4. Consider using third-party operators by Customized operators: see [Use Third-Party Operators by Custom Operators](https://www.mindspore.cn/docs/en/master/migration_guide/use_third_party_op.html).
4. Consider temporary circumvention solutions: For example, if a certain loss is not supported, it can be replaced with a loss operator of the same kind that has been supported.
5. Submit suggestions in [MindSpore Community](https://gitee.com/mindspore/mindspore/issues) to develop missing operators.

## Grammar Assessment

MindSpore provides two modes: `GRAPH_MODE` and `PYNATIVE_MODE`.

In PyNative mode, the behavior of the model for **Inference** is same as that of in the general Python code.

When using GRAPH_MODE, or when using PYNATIVE_MODE for **Training**, there are usually grammatical restrictions. In these two cases, it is necessary to perform graph compilation operations on the Python code. In this step, MindSpore has not yet been able to support the complete set of Python syntax, so there will be some restrictions on the implementation of the `construct` function. For specific restrictions, please refer to [MindSpore static graph syntax support](https://www.mindspore.cn/docs/en/master/note/static_graph_syntax_support.html).

### Common Restriction Principles

Compared with the specific syntax description, the common restrictions can be summarized as follows:

- Do not call other Python module , such as numpy and scipy, when building the graph. The related processing should be moved to the `__init__` stage.
- Do not use custom types when building the graph. Instead, use the data types and Python basic types provided by MindSpore. You can use tuple/list combinations based on these types.
- Do not processing multi-threaded, multi-process data when building the graph.

### Common Processing Strategies

1. Use the operators provided by MindSpore to replace the functions of other Python libraries. The processing of constants can be moved to the `__init__` stage.
2. Use basic types for combination, and you can consider increasing the amount of function parameters. There are no restrictions on the input parameters of the function, and variable length input can be used.
3. Avoid multi-threading processing in the network.
