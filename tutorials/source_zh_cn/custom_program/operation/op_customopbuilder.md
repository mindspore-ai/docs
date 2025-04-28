# 基于CustomOpBuilder的自定义算子

[![查看源文件](https://mindspore-website.obs.cn-north-4.myhuaweicloud.com/website-images/r2.6.0/resource/_static/logo_source.svg)](https://gitee.com/mindspore/docs/blob/r2.6.0/tutorials/source_zh_cn/custom_program/operation/op_customopbuilder.md)

## 概述

动态图模式下，网络流程更容易调试，可以支持执行单算子、普通函数和网络，以及单独求梯度等操作。

[基于Custom原语的自定义算子](https://www.mindspore.cn/tutorials/zh-CN/r2.6.0/custom_program/operation/op_custom_prim.html)虽然可以同时支持静态图和动态图，但是需要定义的内容较多。因此MindSpore针对动态图的自定义算子接入方式做了优化，提供了新的Python API [CustomOpBuilder](https://www.mindspore.cn/docs/zh-CN/r2.6.0/api_python/ops/mindspore.ops.CustomOpBuilder.html) ，在方便用户使用的同时，还能提升动态图自定义算子的执行性能。

用户基于C++接口开发算子，需要定义算子函数体，包括推导并构造输出Tensor，调用执行device算子等功能。定义好算子函数体后，通过[pybind11](https://github.com/pybind/pybind11) 组件即可将C++函数注册成为Python模块接口。

## 使用场景示例

- [通过Function接口开发正反向算子](https://www.mindspore.cn/tutorials/zh-CN/r2.6.0/custom_program/operation/op_customopbuilder_function.html)： 介绍定义自定义算子正向传播函数和反向传播函数的方法。
