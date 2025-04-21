# 编程形态概述

[![查看源文件](https://mindspore-website.obs.cn-north-4.myhuaweicloud.com/website-images/r2.6.0rc1/resource/_static/logo_source.svg)](https://gitee.com/mindspore/docs/blob/r2.6.0rc1/docs/mindspore/source_zh_cn/features/program_form/overview.md)

MindSpore是面向“端-边-云”全场景设计的AI框架，为用户提供AI模型开发、训练、推理的接口，支持用户用原生Python语法开发和调试神经网络，其提供动态图、静态图、动静统一的编程形态，使开发者可以兼顾开发效率和执行性能。

考虑开发灵活性、易用性，MindSpore支持动态图的编程模式，基于MindSpore提供的functional和nn.cell接口，用户可以灵活组装构建所需网络，相关接口按照Python函数库的形态解释执行，并支持微分求导能力。从而易于调试和开发。相关接口按配置支持加速硬件的异步下发执行从而实现异构加速。

同时，基于动态图模式，MindSpore提供@jit的装饰器优化能力，可以指定函数通过@jit装饰优化，装饰部分会被整体解析，构建成C++计算图，进行全局分析，编译优化，从而加速被装饰部分的整体执行性能。这一过程我们也称之为静态化加速。

除了动态图模式，MindSpore进一步提供了[静态图](https://www.mindspore.cn/tutorials/zh-CN/r2.6.0rc1/compile/static_graph.html)的编程模式，相关MindSpore模型构建接口不变，无需添加@jit装饰，MindSpore框架会针对所有开发在nn.cell类中construct函数的定义内容，整体编译解析，构建针对网络的完整静态图，进行模型整图级编译优化与执行。这样能针对整网，基于AI模型训练、推理的特点，进行模型级专有的优化，获取更高的执行性能。
