Inference on Ascend 310
===============================

Ascend 310 is a high-efficiency and highly integrated AI processor for edge scenes. It supports to perform inference on MindIR format and AIR format models.

MindIR format can be exported by MindSpore CPU, GPU, Ascend 910, and can be run on GPU, Ascend 910, Ascend 310. There is no need to manually perform model conversion before inference. MindSpore needs to be installed during inference, and MindSpore C++ API is called for inference.

AIR format can only be exported by MindSpore Ascend 910 and only Ascend 310 can infer. Before inference, the atc tool in Ascend CANN needs to be used for model conversion. MindSpore is not required for inference, only Ascend CANN software package is required.

.. toctree::
  :maxdepth: 1

  inference_ascend_310_mindir
  inference_ascend_310_air
