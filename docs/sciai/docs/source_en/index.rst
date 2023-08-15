Introduction to SciAI
=====================

Based on MindSpore, SciAI is a model library with 60 built-in most frequently used and cited AI4Sci(AI for Science) models, ranks No.1 in the world in terms of coverage.

MindSpore SciAI provides the developers and users with a high-level API, allowing an immediate deployment. Accuracy and performance are up to industry SOTA.

Specifically, the models provided by SciAI cover from physics-informed (PINNs, DeepRitz, PFNN, etc.) to neural operators (FNO, DeepONet) and cover a wide variety of scientific computation field, including fluid dynamics, electromagnetism, sound, heat, solid and biology etc.

With these features, SciAI provides developers and users with an efficient and convenient AI4SCI computation platform, and is committed to creating a Hugging Face in the field of scientific computing.

.. raw:: html

   <img src="https://mindspore-website.obs.cn-north-4.myhuaweicloud.com/website-images/master/docs/sciai/docs/source_en/images/architecture.jpg" width="600px" alt="" >

.. toctree::
   :maxdepth: 1
   :caption: Installation

   installation

.. toctree::
   :maxdepth: 1
   :caption: Model Library Instruction

   launch_with_scripts
   launch_with_api

.. toctree::
   :maxdepth: 1
   :caption: Tutorial

   build_model_with_sciai

.. toctree::
   :maxdepth: 1
   :caption: API Reference

   sciai.architecture
   sciai.common
   sciai.context
   sciai.operators
   sciai.utils
