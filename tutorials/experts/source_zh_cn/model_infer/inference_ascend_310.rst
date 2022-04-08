Ascend 310 AI处理器上推理
===============================

Ascend 310是面向边缘场景的高能效高集成度AI处理器，支持对MindIR格式和AIR格式模型进行推理。

MindIR格式可由MindSpore CPU、GPU、Ascend 910导出，可运行在GPU、Ascend 910、Ascend 310上，推理前不需要手动执行模型转换，推理时需要安装MindSpore，调用MindSpore C++ API进行推理。

AIR格式仅MindSpore Ascend 910可导出，仅Ascend 310可推理，推理前需使用Ascend CANN中atc工具进行模型转换，推理时不依赖MindSpore，仅需Ascend CANN软件包。

.. toctree::
  :maxdepth: 1

  inference_ascend_310_mindir
  inference_ascend_310_air
