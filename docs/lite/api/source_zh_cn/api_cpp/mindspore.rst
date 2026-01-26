mindspore
============

.. image:: https://mindspore-website.obs.cn-north-4.myhuaweicloud.com/website-images/master/resource/_static/logo_source.svg
    :target: https://atomgit.com/mindspore/docs/blob/master/docs/lite/api/source_zh_cn/api_cpp/mindspore.rst
    :alt: 查看源文件

.. toctree::
   :maxdepth: 1
   :hidden:

   ../generate/classmindspore_AbstractDelegate
   ../generate/classmindspore_Allocator
   ../generate/classmindspore_AscendDeviceInfo
   ../generate/classmindspore_Buffer
   ../generate/classmindspore_Context
   ../generate/classmindspore_CoreMLDelegate
   ../generate/classmindspore_CPUDeviceInfo
   ../generate/classmindspore_Delegate
   ../generate/classmindspore_DelegateModel
   ../generate/classmindspore_DeviceInfoContext
   ../generate/classmindspore_DSPDeviceInfo
   ../generate/classmindspore_GPUDeviceInfo
   ../generate/classmindspore_Graph
   ../generate/classmindspore_IDelegate
   ../generate/classmindspore_KirinNPUDeviceInfo
   ../generate/classmindspore_MixPrecisionCfg
   ../generate/classmindspore_Model
   ../generate/classmindspore_ModelExecutor
   ../generate/classmindspore_ModelParallelRunner
   ../generate/classmindspore_MSTensor
   ../generate/classmindspore_MultiModelRunner
   ../generate/classmindspore_RunnerConfig
   ../generate/classmindspore_Serialization
   ../generate/classmindspore_Status
   ../generate/classmindspore_TrainCfg
   ../generate/enum_mindspore_CompCode-1
   ../generate/enum_mindspore_DataType-1
   ../generate/enum_mindspore_DelegateMode-1
   ../generate/enum_mindspore_Format-1
   ../generate/enum_mindspore_ModelGroupFlag-1
   ../generate/enum_mindspore_SchemaVersion-1
   ../generate/function_mindspore_CharVersion-1
   ../generate/function_mindspore_Version-1   
   ../generate/structmindspore_Key
   ../generate/structmindspore_MSCallBackParam   
   ../generate/structmindspore_QuantParam
   ../generate/typedef_mindspore_KernelIter-1
   ../generate/typedef_mindspore_MSKernelCallBack-1

推理
-----

.. list-table::
   :widths: 30 40 15 15
   :header-rows: 1

   * - 接口名
     - 描述
     - 云侧推理是否支持
     - 端侧推理是否支持
   * - :doc:`class Model <../generate/classmindspore_Model>`
     - MindSpore中的模型，便于计算图管理。
     - √
     - √
   * - :doc:`class ModelExecutor <../generate/classmindspore_ModelExecutor>`
     - 包装多个Model类，用于调度多个Model对象。
     - √
     - ✕
   * - :doc:`class MultiModelRunner <../generate/classmindspore_MultiModelRunner>`
     - 包装多个ModelExecutor类，用于调度多个Model对象。
     - √
     - ✕

运行环境配置
------------

.. list-table::
   :widths: 30 40 15 15
   :header-rows: 1

   * - 接口名
     - 描述
     - 云侧推理是否支持
     - 端侧推理是否支持
   * - :doc:`Class AscendDeviceInfo <../generate/classmindspore_AscendDeviceInfo>`
     - 模型运行在Atlas 200/300/500推理产品、Atlas推理系列产品上的配置。
     - √
     - √
   * - :doc:`Class Context <../generate/classmindspore_Context>`
     - 保存执行中的环境变量。
     - √
     - √
   * - :doc:`Class CPUDeviceInfo <../generate/classmindspore_CPUDeviceInfo>`
     - 模型运行在CPU上的配置。
     - √
     - √
   * - :doc:`Class DeviceInfoContext <../generate/classmindspore_DeviceInfoContext>`
     - 不同硬件设备的环境信息。
     - √
     - √
   * - :doc:`Class DSPDeviceInfo <../generate/classmindspore_DSPDeviceInfo>`
     - 模型运行在DSP上的配置。支持ft04和ft78设备。
     - ✕
     - √
   * - :doc:`Class GPUDeviceInfo <../generate/classmindspore_GPUDeviceInfo>`
     - 模型运行在GPU上的配置。
     - √
     - √
   * - :doc:`Class KirinNPUDeviceInfo <../generate/classmindspore_KirinNPUDeviceInfo>`
     - 模型运行在NPU上的配置。
     - ✕
     - √
   * - :doc:`Enum DelegateMode <../generate/enum_mindspore_DelegateMode-1>`
     - 模型运行的代理模式。
     - √
     - √
   * - :doc:`Struct Key <../generate/structmindspore_Key>`
     - 键。
     - √
     - √

并发推理
--------

.. list-table::
   :widths: 30 40 15 15
   :header-rows: 1

   * - 接口名
     - 描述
     - 云侧推理是否支持
     - 端侧推理是否支持
   * - :doc:`Class RunnerConfig <../generate/classmindspore_RunnerConfig>`
     - 模型并发推理配置参数。
     - √
     - ✕
   * - :doc:`Class ModelParallelRunner <../generate/classmindspore_ModelParallelRunner>`
     - 模型并发推理类。
     - √
     - ✕

张量Tensor相关
-----------------

.. list-table::
   :widths: 30 40 15 15
   :header-rows: 1

   * - 接口名
     - 描述
     - 云侧推理是否支持
     - 端侧推理是否支持
   * - :doc:`Class Allocator <../generate/classmindspore_Allocator>`
     - 内存管理基类。
     - √
     - √
   * - :doc:`Class MSTensor <../generate/classmindspore_MSTensor>`
     - MindSpore中的张量。
     - √
     - √
   * - :doc:`Enum DataType <../generate/enum_mindspore_DataType-1>`
     - MindSpore MSTensor保存的数据支持的类型。
     - √
     - √
   * - :doc:`Enum Format <../generate/enum_mindspore_Format-1>`
     - MindSpore MSTensor保存的数据支持的排列格式。
     - √
     - √
   * - :doc:`Struct QuantParam <../generate/structmindspore_QuantParam>`
     - MSTensor中的一组量化参数。
     - √
     - √

模型分组
----------

.. list-table::
   :widths: 30 40 15 15
   :header-rows: 1

   * - 接口名
     - 描述
     - 云侧推理是否支持
     - 端侧推理是否支持
   * - :doc:`Enum ModelGroupFlag <../generate/enum_mindspore_ModelGroupFlag-1>`
     - 模型分组。
     - √
     - √

状态
----------

.. list-table::
   :widths: 30 40 15 15
   :header-rows: 1

   * - 接口名
     - 描述
     - 云侧推理是否支持
     - 端侧推理是否支持
   * - :doc:`Class Status <../generate/classmindspore_Status>`
     - 返回状态类。
     - √
     - √
   * - :doc:`Enum CompCode <../generate/enum_mindspore_CompCode-1>`
     - 返回计算类别。
     - √
     - √

序列化保存与加载
-----------------

.. list-table::
   :widths: 30 40 15 15
   :header-rows: 1

   * - 接口名
     - 描述
     - 云侧推理是否支持
     - 端侧推理是否支持
   * - :doc:`Class Buffer <../generate/classmindspore_Buffer>`
     - Buffer数据类。
     - √
     - √
   * - :doc:`Class Serialization <../generate/classmindspore_Serialization>`
     - 汇总了模型文件读写的方法。
     - √
     - √

版本查询
---------

.. list-table::
   :widths: 30 40 15 15
   :header-rows: 1

   * - 接口名
     - 描述
     - 云侧推理是否支持
     - 端侧推理是否支持
   * - :doc:`Enum SchemaVersion <../generate/enum_mindspore_SchemaVersion-1>`
     - MindSpore Lite 执行推理时，模型文件的版本。
     - ✕
     - √
   * - :doc:`Function CharVersion <../generate/function_mindspore_CharVersion-1>`
     - 获取字符vector形式的当前版本号。
     - ✕
     - √
   * - :doc:`Function Version <../generate/function_mindspore_Version-1>`
     - 获取字符串形式的当前版本号。
     - ✕
     - √

回调函数
---------

.. list-table::
   :widths: 30 40 15 15
   :header-rows: 1

   * - 接口名
     - 描述
     - 云侧推理是否支持
     - 端侧推理是否支持
   * - :doc:`Struct MSCallBackParam <../generate/structmindspore_MSCallBackParam>`
     - MindSpore回调函数的参数。
     - √
     - √
   * - :doc:`Typedef mindspore::MSKernelCallBack <../generate/typedef_mindspore_MSKernelCallBack-1>`
     - MindSpore回调函数包装器。
     - √
     - √


MindSpore Lite 训练配置
------------------------

.. list-table::
   :widths: 30 40 15 15
   :header-rows: 1

   * - 接口名
     - 描述
     - 云侧推理是否支持
     - 端侧推理是否支持
   * - :doc:`Class TrainCfg <../generate/classmindspore_TrainCfg>`
     - MindSpore Lite训练配置类。
     - ✕
     - √
   * - :doc:`Class MixPrecisionCfg <../generate/classmindspore_MixPrecisionCfg>`
     - MindSpore Lite训练混合精度配置类。
     - ✕
     - √

Delegate三方框架接入机制
------------------------

.. list-table::
   :widths: 30 40 15 15
   :header-rows: 1

   * - 接口名
     - 描述
     - 云侧推理是否支持
     - 端侧推理是否支持
   * - :doc:`Class AbstractDelegate <../generate/classmindspore_AbstractDelegate>`
     - MindSpore Lite接入代理（抽象类）。
     - √
     - ✕
   * - :doc:`Class CoreMLDelegate <../generate/classmindspore_CoreMLDelegate>`
     - MindSpore Lite接入CoreML框架的代理。
     - ✕
     - √          
   * - :doc:`Class Delegate <../generate/classmindspore_Delegate>`
     - MindSpore Lite接入第三方AI框架的代理。
     - ✕
     - √
   * - :doc:`Class DelegateModel <../generate/classmindspore_DelegateModel>`
     - MindSpore Lite Delegate机制封装的模型。
     - ✕
     - √
   * - :doc:`Class IDelegate <../generate/classmindspore_IDelegate>`
     - MindSpore Lite接入代理（模板类）。
     - √
     - ✕     
   * - :doc:`Typedef mindspore::KernelIter <../generate/typedef_mindspore_KernelIter-1>`
     - MindSpore Lite 算子列表的迭代器。
     - ✕
     - √

图容器
----------------

.. list-table::
   :widths: 30 40 15 15
   :header-rows: 1

   * - 接口名
     - 描述
     - 云侧推理是否支持
     - 端侧推理是否支持
   * - :doc:`Class Graph <../generate/classmindspore_Graph>`
     - 图类。
     - ✕
     - √
