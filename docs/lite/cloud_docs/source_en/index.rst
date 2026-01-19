.. MindSpore documentation master file, created by
   sphinx-quickstart on Thu Aug 17 09:00:00 2020.
   You can adapt this file completely to your liking, but it should at least
   contain the root `toctree` directive.

MindSpore Lite Cloud-side Documentation
==========================================

MindSpore Lite inference comprises two components: cloud-side inference and device-side inference. This document primarily introduces MindSpore Lite cloud-side inference. For device-side inference, please refer to the `Device-side Inference Documentation <https://www.mindspore.cn/lite/docs/en/r2.8.0/index.html>`_ .

Usage Scenarios
------------------

MindSpore Lite cloud-side inference is primarily designed for server-side devices. It offers compatibility with model structures exported from the MindSpore training framework and various open-source formats including ONNX, TFLite, and Pb. This version is applicable to Ascend cards (such as Atlas 300I Duo, Atlas 800I A2, Atlas 800I A3 series) and CPU hardware based on X86/Arm architectures. MindSpore Lite has also implemented targeted optimizations and adaptations for various algorithmic scenarios. Its current features and optimizations primarily focus on multi-modal generation, speech recognition, speech synthesis, autonomous driving, vector models, and traditional computer vision domains.

Advantages
------------

1. MindSpore Lite effectively reduces operator dispatch latency through whole-graph sinking during model inference, thereby enhancing model inference performance;

2. For multi-modal generative algorithm models, MindSpore Lite supports key capabilities including multiple cache mechanisms, quantization, shared memory, and multi-dimensional hybrid parallelism. For Ascend hardware, MindSpore Lite enables user-defined operator integration;

3. For speech-related algorithm models, MindSpore Lite supports key capabilities such as zero-copy I/O data processing;

4. For autonomous driving models, MindSpore Lite supports hybrid scheduling of single operators and subgraphs during inference on Ascend hardware. This ensures subgraph-sinking inference performance while enabling rapid integration of custom operators for autonomous driving applications through hybrid scheduling.

Development Process
-------------------------

.. image:: ./images/lite_runtime.png

Using the MindSpore Lite inference framework primarily involves the following steps:

1. Model loading: You can directly load MindIR models exported from MindSpore training, or convert models exported from third-party frameworks into MindIR format using the MindSpore Lite conversion tool. These converted models can then be loaded via MindSpore Lite's interfaces.

2. Model compilation:

   1. Create a configuration context: By creating a ``Context``, save some essential configuration parameters to guide graph compilation and model execution.

   2. Model loading: Before performing inference, the ``Build`` interface of the ``Model`` must be invoked to load the model. This process parses the cached file into a runtime model.

   3. Graph compilation: After model loading completes, the MindSpore Lite runtime compiles the graph. The model compilation phase consumes significant time, so it is recommended to create the model once, compile it once, and perform multiple inferences.

3. Model inference:

   1. Input data must be padded before model execution.

   2. Execute inference: Use the ``Predict`` function of the ``Model`` for model inference.

   3. Obtain the output: The ``outputs`` parameter in the ``Predict`` interface returns the inference results. By parsing the ``MSTensor`` object, you can obtain the model's inference results along with the output data type and size.

4. Memory release: During the model compilation phase, resources such as resident memory, video memory, and thread pools are allocated. These resources must be released after model inference concludes to prevent resource leaks.

.. toctree::
   :glob:
   :maxdepth: 1
   :caption: Obtain MindSpore Lite
   :hidden:

   use/downloads
   use/build

.. toctree::
   :glob:
   :maxdepth: 1
   :caption: Quick Start
   :hidden:

   quick_start/one_hour_introduction

.. toctree::
   :glob:
   :maxdepth: 1
   :caption: Model Converter
   :hidden:

   mindir/converter_tool
   mindir/converter_python
   mindir/converter_tool_ascend
   mindir/converter_custom

.. toctree::
   :glob:
   :maxdepth: 1
   :caption: Cloud-side Inference
   :hidden:

   mindir/runtime
   mindir/runtime_parallel
   mindir/runtime_distributed

.. toctree::
   :glob:
   :maxdepth: 1
   :caption: Cloud-side Tools
   :hidden:

   mindir/benchmark_tool

.. toctree::
   :glob:
   :maxdepth: 1
   :caption: References
   :hidden:

   reference/operator_lite
   reference/environment_variable_support

.. toctree::
   :maxdepth: 1
   :caption: RELEASE NOTES
   :hidden:

   RELEASE
