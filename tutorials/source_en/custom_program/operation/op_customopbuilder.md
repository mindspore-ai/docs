# CustomOpBuilder-Based Custom Operators

[![View Source On Gitee](https://mindspore-website.obs.cn-north-4.myhuaweicloud.com/website-images/master/resource/_static/logo_source_en.svg)](https://gitee.com/mindspore/docs/blob/master/tutorials/source_en/custom_program/operation/op_customopbuilder.md)

## Overview

In dynamic graph mode, network workflows are easier to debug, supporting operations like single-operator execution, normal functions/networks, and standalone gradient computations.

While [Custom Primitive-Based Custom Operators](https://www.mindspore.cn/tutorials/en/master/custom_program/op_custom.html) support both static and dynamic graphs, they require extensive definitions. To simplify the integration of custom operators in dynamic graphs, MindSpore has introduced a new Python API, [CustomOpBuilder](https://www.mindspore.cn/docs/en/master/api_python/ops/mindspore.ops.CustomOpBuilder.html), which not only improves usability but also enhances the execution performance of custom operators in dynamic graphs.

When developing operators using C++ interfaces, users need to define the operator function body, including deriving and constructing output tensors, calling and executing device operators, and more. Once the function body is defined, the [pybind11](https://github.com/pybind/pybind11) component can be used to register C++ functions as Python module interfaces.

## Usage Scenarios

- [Developing Forward and Backward Operators Using the Function Interface](https://www.mindspore.cn/tutorials/en/master/custom_program/operation/op_customopbuilder_function.html): Introduces the method of defining custom operator forward and backward propagation functions.