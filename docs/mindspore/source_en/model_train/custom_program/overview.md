# Overview of Custom Higher-Order Programming

[![View Source On Gitee](https://mindspore-website.obs.cn-north-4.myhuaweicloud.com/website-images/r2.4.10/resource/_static/logo_source_en.svg)](https://gitee.com/mindspore/docs/blob/r2.4.10/docs/mindspore/source_en/model_train/custom_program/overview.md)

During the training process, when the advanced methods provided by the framework cannot satisfy certain scenarios of the developer, or when the developer has high performance requirements, a customized method can be used to add or modify certain processes to meet development or debugging requirements.

Currently MindSpore provides some ways to customize higher-order programming, and you can refer to the following guidelines:

## Customizing Operators

When the built-in operators are not sufficient, you can utilize the [Custom](https://www.mindspore.cn/docs/en/r2.4.10/api_python/ops/mindspore.ops.Custom.html#mindspore-ops-custom) principle to quickly and easily define and use different types of custom operators.

## Customizing Neural Network Layers

Typically, the neural network layer interface and function function interface provided by MindSpore are able to meet the model construction requirements, but since the AI field is constantly pushing the boundaries, it is possible to encounter new network structures without built-in modules.

At this point we can customize the neural network layer as needed through the function interface provided by MindSpore, the Primitive operator, and can customize the inverse using the `Cell.bprop` method.

## Customizing Parameter Initialization

MindSpore provides a variety of ways to initialize network parameters and encapsulates the function of parameter initialization in some operators. The main methods are as follows:

- **Initializer Initialization**: Initializer is MindSpore built-in parameter initialization base class, and all built-in parameter initialization methods inherit from this class.
- **String initialization**: MindSpore also provides simple method of the Parameter Initialization, which initializes the string of the method name using the parameter. This method initializes using the default parameters of the Initializer.
- **Custom Parameter Initialization**: When encountering the need for a custom parameter initialization method, you can inherit the Initializer custom parameter initialization method.
- **Cell Traversal Initialization**: Construct the network and instantiate it first, then traverse the Cell and assign values to the parameters.

## Customizing Loss Functions

A number of generic loss functions are provided in the `mindspore.nn` module, and when these generic loss functions do not meet all the requirements, the user can customize the desired loss function.

## Customizing Optimizers

The nn module in MindSpore provides commonly-used optimizers, such as nn.SGD, nn.Adam, and nn.Momentum, and when these optimizers can not meet the development needs, users can customize the optimizer.

## Hook Programming

In order to facilitate users to accurately and quickly debug the deep learning network, MindSpore has designed the Hook function in dynamic graph mode, using which the input and output data as well as the inverse gradient of the middle-layer operators can be captured.
