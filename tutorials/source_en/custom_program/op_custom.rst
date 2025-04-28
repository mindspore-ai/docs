Custom Operators
=================

.. image:: https://mindspore-website.obs.cn-north-4.myhuaweicloud.com/website-images/r2.6.0rc1/resource/_static/logo_source_en.svg
    :target: https://gitee.com/mindspore/docs/blob/r2.6.0rc1/tutorials/source_en/custom_program/op_custom.rst
    :alt: View Source On Gitee

.. toctree::
   :maxdepth: 1
   :hidden:

   operation/op_custom_prim
   operation/op_custom_ascendc
   operation/op_custom_aot
   operation/op_custom_julia
   operation/op_custom_adv
   operation/op_customopbuilder
   operation/op_customopbuilder_function

When built-in operators cannot meet requirements during network development, you can use MindSpore's custom operator functionality to integrate your operators. Currently, MindSpore provides two approaches for integrating custom operators:

- `Custom Primitive-Based Custom Operators <https://www.mindspore.cn/tutorials/en/r2.6.0rc1/custom_program/operation/op_custom_prim.html>`_
- `CustomOpBuilder-Based Custom Operators <https://www.mindspore.cn/tutorials/en/r2.6.0rc1/custom_program/operation/op_customopbuilder.html>`_

.. list-table::
   :widths: 20 40 40
   :header-rows: 1

   * - Interface Comparison
     - `Custom Primitive <https://www.mindspore.cn/tutorials/en/r2.6.0rc1/custom_program/operation/op_custom_prim.html>`_
     - `CustomOpBuilder <https://www.mindspore.cn/tutorials/en/r2.6.0rc1/custom_program/operation/op_customopbuilder.html>`_
   * - Supported Modes
     - Graph Mode and PyNative Mode
     - PyNative Mode
   * - Interface Functions
     - Provides a unified Custom Primitive that calls user interfaces at various stages of operator execution.
     - Compiles and loads custom operator modules online, which can be directly applied to networks.
   * - Advantages
     - Supports both Graph and PyNative mode , with operator scheduling and execution processes consistent with built-in operators, ensuring high performance.
     - Enables operator development based on C++ tensors, offering a more intuitive custom execution process.
   * - Disadvantages
     - Has more interface restrictions, and the operator execution process is not visible to users.
     - Involves multiple interfaces for operator development; currently lacks a concise and efficient C++ API, making the development of high-performance operators challenging.
   * - Feature Level
     - STABLE
     - BETA
