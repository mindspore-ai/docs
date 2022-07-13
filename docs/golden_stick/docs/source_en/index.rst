MindSpore Golden Stick
=============================

MindSpore Golden Stick is an open-source model compression algorithm set. It provides a set of user APIs for application algorithms, allowing users to use model compression algorithms such as quantization and pruning in a unified and convenient manner. The MindSpore Golden Stick also provides algorithm developers with basic capabilities of modifying network definitions. It abstracts an IR layer between the algorithm and network definition to shield the specific network definition from algorithm developers so that they can focus on algorithm logic development.

.. raw:: html

   <img src="https://mindspore-website.obs.cn-north-4.myhuaweicloud.com/website-images/master/docs/golden_stick/docs/source_en/images/golden-stick-arch.png" width="700px" alt="" >

Design Guidelines
---------------------------------------

1. Provide unified algorithm APIs to reduce the learning cost of application algorithms.

   There are various types of model compression algorithms, such as the quantization aware training algorithm, pruning algorithm, matrix decomposition algorithm, and knowledge distillation algorithm. In each type of compression algorithm, there are various specific algorithms. For example, LSQ and PACT are both quantization aware training algorithms. Different algorithms are usually applied in different manners, which increases learning costs of a user for applying an algorithm. The MindSpore Golden Stick streamlines and abstracts the algorithm application process and provides a set of unified algorithm APIs to minimize the learning cost of algorithm applications. This also facilitates the exploration of technologies such as automatic model compression (AMC) and neural architecture search (NAS) based on the algorithm ecosystem.

2. Provide the capability of modifying network definitions to reduce algorithm access costs.

   The model compression algorithm is usually designed or optimized for a specific network structure, and seldom focuses on a specific network definition. The MindSpore Golden Stick provides the capability of modifying the front-end network definitions through APIs so that algorithm developers can focus on algorithm implementation without reinventing the wheel for different network definitions. In addition, the MindSpore Golden Stick provides some commissioning capabilities, including network dump, layer-by-layer profiling, algorithm effect analysis, and visualization, to help algorithm developers improve development and research efficiency and help users find algorithms that meet their requirements.

General Process of Applying the MindSpore Golden Stick
------------------------------------------------------

.. raw:: html

   <img src="https://mindspore-website.obs.cn-north-4.myhuaweicloud.com/website-images/master/docs/golden_stick/docs/source_en/images/workflow.png" width="800px" alt="" >

1. Training

During network training, the MindSpore Golden Stick does not have great impact on the original training script logic. As shown in the highlighted part in the preceding figure, only the following two steps need to be added:

- **Optimize the model using the MindSpore Golden Stick:** In the original training process, after the original network is defined and before the model is trained, use the MindSpore Golden Stick to optimize the network structure. Generally, this step is implemented by calling the `apply` API of MindSpore Golden Stick. For details, see [Applying the SimQAT Algorithm](https://mindspore.cn/golden_stick/docs/en/master/quantization/simqat.html).

- **Register the MindSpore Golden Stick callback API:** Register the callback algorithm of the MindSpore Golden Stick with the model to be trained. Generally, in this step, the `callback` function of MindSpore Golden Stick is called to obtain the corresponding callback object and [register the object with the model](https://www.mindspore.cn/tutorials/zh-CN/master/advanced/train/callback.html).

2. Deployment

- **Network conversion:** A network compressed by MindSpore Golden Stick may require additional steps to convert the model compression structure on the network from training to deployment, facilitating model export and deployment. For example, in the quantization aware scenario, a fake quantization node in a network usually needs to be eliminated, and converted into an operator attribute in the network.

.. note::
 - For details about how to apply the MindSpore Golden Stick, see the detailed description and sample code in each algorithm section.
 - For details about the "network training or retraining" step in the process, see [MindSpore Training and Evaluation](https://mindspore.cn/tutorials/zh-CN/master/advanced/train/train_eval.html).
 - For details about the "ms.export" step in the process, see [Exporting MINDIR Model](https://www.mindspore.cn/tutorials/en/master/advanced/train/save.html#export-mindir-model).
 - For details about "model optimization" and "model export" in the process, see [Converting Models for Inference](https://mindspore.cn/lite/docs/en/master/use/converter_tool.html).
 - For details about the "MindSpore inference runtime" step in the process, see [MindSpore Inference Runtime](https://mindspore.cn/lite/docs/en/master/use/runtime.html).

Planning
---------------------------------------

The initial version includes a stable API and provides a linear quantization algorithm, a non-linear quantization algorithm, and a structured pruning algorithm. More algorithms and better network support will be provided in later versions. The commissioning capability will also be provided in later versions. In the future, with more algorithms, the MindSpore Golden Stick will explore capabilities such as automatic model compression (AMC), hardware-aware automated quantization (HAQ), and neural architecture search (NAS).

.. toctree::
   :glob:
   :maxdepth: 1
   :caption: Installation and Deployment

   install

.. toctree::
   :glob:
   :maxdepth: 1
   :caption: Quantization Algorithms

   quantization/overview
   quantization/simqat
   quantization/slb

.. toctree::
   :glob:
   :maxdepth: 1
   :caption: Pruning Algorithms

   pruner/overview
   pruner/scop

.. toctree::
   :maxdepth: 1
   :caption: API References

   mindspore_gs
