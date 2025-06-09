自定义算子
============

.. image:: https://mindspore-website.obs.cn-north-4.myhuaweicloud.com/website-images/master/resource/_static/logo_source.svg
    :target: https://gitee.com/mindspore/docs/blob/master/tutorials/source_zh_cn/custom_program/op_custom.rst
    :alt: 查看源文件

.. toctree::
   :maxdepth: 1
   :hidden:

   operation/op_custom_prim
   operation/op_custom_ascendc
   operation/op_custom_aot
   operation/op_custom_julia
   operation/op_custom_adv
   operation/op_customopbuilder
   operation/cpp_api_for_custom_ops
   operation/op_customopbuilder_atb
   operation/op_customopbuilder_function

当开发网络遇到内置算子不足以满足需求时，你可以利用MindSpore的自定义算子功能接入你的算子。当前MindSpore提供了两种方式接入自定义算子，分别是 `基于Custom原语接入 <https://www.mindspore.cn/tutorials/zh-CN/master/custom_program/operation/op_custom_prim.html>`_ 和 `基于CustomOpBuilder接入 <https://www.mindspore.cn/tutorials/zh-CN/master/custom_program/operation/op_customopbuilder.html>`_ 。


.. list-table::
   :widths: 20 40 40
   :header-rows: 1

   * - 接口比较
     - `Custom原语 <https://www.mindspore.cn/tutorials/zh-CN/master/custom_program/operation/op_custom_prim.html>`_
     - `CustomOpBuilder <https://www.mindspore.cn/tutorials/zh-CN/master/custom_program/operation/op_customopbuilder.html>`_
   * - 支持模式
     - 静态图（Graph Mode）和动态图（PyNative Mode）
     - 动态图（PyNative Mode）
   * - 接口功能
     - 提供统一的Custom原语，在算子执行的各个阶段分别调用用户接口。
     - 在线编译和加载自定义算子模块，可以直接应用到网络当中。
   * - 优点
     - 同时支持动态图和静态图，算子调度执行流程与内置算子一致，执行性能高效。
     - 基于C++ tensor开发算子，自定义执行流程，更加直观。
   * - 缺点
     - 接口限制较多，算子执行流程对用户不可见。
     - 开发算子涉及的接口较多，当前暂无简洁高效的C++ API，开发高效算子的难度较大。
   * - 特性等级
     - STABLE
     - BETA
