MindSpore Golden Stick
=============================

MindSpore Golden Stick is a model compression algorithm set jointly designed and developed by Huawei's Noah team and Huawei's MindSpore team. The architecture diagram of MindSpore Golden Stick is shown in the figure below, which is divided into five parts:

.. raw:: html

   <img src="https://mindspore-website.obs.cn-north-4.myhuaweicloud.com/website-images/master/docs/golden_stick/docs/source_en/images/golden-stick-arch.png" width="700px" alt="" >

1. The underlying MindSpore Rewrite module provides the ability to modify the front-end network. Based on the interface provided by this module, algorithm developers can add, delete, query and modify the nodes and topology relationships of the MindSpore front-end network according to specific rules;

2. Based on MindSpore Rewrite, MindSpore Golden Stick will provide various types of algorithms, such as SimQAT algorithm, SLB quantization algorithm, SCOP pruning algorithm, etc.;

3. At the upper level of the algorithm, MindSpore Golden Stick also plans advanced technologies such as AMC (AutoML for Model Compression), NAS (Neural Architecture Search), and HAQ (Hardware-aware Automated Quantization);

4. In order to facilitate developers to analyze and debug algorithms, MindSpore Golden Stick provides some tools, such as visualization tool, profiler tool, summary tool, etc.;

5. In the outermost layer, MindSpore Golden Stick encapsulates a set of concise user interface.

.. note:
 The architecture diagram is the overall picture of MindSpore Golden Stick, which includes the features that have been implemented in the current version and the capabilities planned in RoadMap. Please refer to release notes for available features in current version.

Design Guidelines
---------------------------------------

In addition to providing rich model compression algorithms, an important design concept of MindSpore Golden Stick is try to provide users with the most unified and concise experience for a wide variety of model compression algorithms in the industry, and reduce the cost of algorithm application for users. MindSpore Golden Stick implements this philosophy through two initiatives:

1. Unified algorithm interface design to reduce user application costs:

   There are many types of model compression algorithms, such as quantization-aware training algorithms, pruning algorithms, matrix decomposition algorithms, knowledge distillation algorithms, etc. In each type of compression algorithm, there are also various specific algorithms, such as LSQ and PACT, which are both quantization-aware training algorithms. Different algorithms are often applied in different ways, which increases the learning cost for users to apply algorithms. MindSpore Golden Stick sorts out and abstracts the algorithm application process, and provides a set of unified algorithm application interfaces to minimize the learning cost of algorithm application. At the same time, this also facilitates the exploration of advanced technologies such as AMC, NAS, and HAQ based on the algorithm ecology.

2. Provide front-end network modification capabilities to reduce algorithm development costs:

   Model compression algorithms are often designed or optimized for specific network structures. For example, perceptual quantization algorithms often insert fake-quantization nodes on the Conv2d, Conv2d + BatchNorm2d, or Conv2d + BatchNorm2d + Relu structures in the network. MindSpore Golden Stick provides the ability to modify the front-end network through API. Based on this ability, algorithm developers can formulate general network transform rules to implement the algorithm logic without needing to implement the algorithm logic for each specific network. In addition, MindSpore Golden Stick also provides some debugging capabilities, including network dump, level-wise profiling, algorithm effect analysis and visualization tool, aiming to help algorithm developers improve development and research efficiency, and help users find algorithms that meet their needs.

General Process of Applying the MindSpore Golden Stick
------------------------------------------------------

.. raw:: html

   <img src="https://mindspore-website.obs.cn-north-4.myhuaweicloud.com/website-images/master/docs/golden_stick/docs/source_en/images/workflow.png" width="800px" alt="" >

1. Training

During network training, the MindSpore Golden Stick does not have great impact on the original training script logic. As shown in the highlighted part in the preceding figure, only the following two steps need to be added:

- **Optimize the network using the MindSpore Golden Stick:** In the original training process, after the original network is defined and before the network is trained, use the MindSpore Golden Stick to optimize the network structure. Generally, this step is implemented by calling the `apply` API of MindSpore Golden Stick. For details, see `Applying the SimQAT Algorithm <https://mindspore.cn/golden_stick/docs/en/master/quantization/simqat.html>`_ .

- **Register the MindSpore Golden Stick callback:** Register the callback of the MindSpore Golden Stick into the model to be trained. Generally, in this step, the `callback` function of MindSpore Golden Stick is called to obtain the corresponding callback object and `register the object into the model <https://www.mindspore.cn/tutorials/zh-CN/master/advanced/model/callback.html>`_ .

2. Deployment

- **Network conversion:** A network compressed by MindSpore Golden Stick may require additional steps to convert the model compression structure from training mode to deployment mode, facilitating model export and deployment. For example, in the quantization aware scenario, a fake quantization node in a network usually needs to be eliminated, and converted into an operator attribute in the network. This capability is not enabled in the current version.

.. note::
 - For details about how to apply the MindSpore Golden Stick, see the detailed description and sample code in each algorithm section.
 - For details about the "network training or retraining" step in the process, see `MindSpore Training and Evaluation <https://mindspore.cn/tutorials/zh-CN/master/advanced/model/train_eval.html>`_ .
 - For details about the "ms.export" step in the process, see `Exporting MINDIR Model <https://www.mindspore.cn/tutorials/en/master/advanced/train/save.html#export-mindir-model>`_ .
 - For details about the "MindSpore infer" step in the process, see `MindSpore Inference Runtime <https://mindspore.cn/tutorials/experts/zh-CN/master/infer/inference.html>`_ .

Roadmap
---------------------------------------

The current release version of MindSpore Golden Stick provides a stable API and provides a linear quantization algorithm, a nonlinear quantization algorithm and a structured pruning algorithm. More algorithms and better network support will be provided in the future version, and debugging capabilities will also be provided in subsequent versions. With the enrichment of algorithms in the future, MindSpore Golden Stick will also explore capabilities such as AMC, HAQ, NAS, etc., so stay tuned.

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

   mindspore_gs.quantization
   mindspore_gs.pruner

.. toctree::
   :glob:
   :maxdepth: 1
   :caption: RELEASE NOTES

   RELEASE
