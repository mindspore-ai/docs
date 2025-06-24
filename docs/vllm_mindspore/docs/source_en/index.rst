vLLM MindSpore
=========================================

Overview
-----------------------------------------------------
vLLM MindSpore (`vllm-mindspore`) is a plugin brewed by the `MindSpore community <https://www.mindspore.cn/en>`_ , which aims to integrate MindSpore LLM inference capabilities into `vLLM <https://github.com/vllm-project/vllm>`_ . With vLLM MindSpore, technical strengths of Mindspore and vLLM will be organically combined to provide a full-stack open-source, high-performance, easy-to-use LLM inference solution.

vLLM, an opensource and community-driven project initiated by Sky Computing Lab, UC Berkeley, has been widely used in academic research and industry applications. On the basis of Continuous Batching scheduling mechanism and PagedAttention Key-Value cache management, vLLM provides a rich set of inference service features, including speculative inference, Prefix Caching, Multi-LoRA, etc. vLLM also supports a wide range of open-source large models, including Transformer-based models (e.g., LLaMa), Mixture-of-Expert models (e.g., DeepSeek), Embedding models (e.g., E5-Mistral), and multi-modal models (e.g., LLaVA). Because vLLM chooses to use PyTorch to build large models and manage storage resources, it cannot deploy large models built upon MindSpore.

vLLM MindSpore plugin aims to integrate Mindspore large models into vLLM and to enable deploying MindSpore-based LLM inference services. It follows the following design principles:

- Interface compatibility: support the native APIs and service deployment interfaces of vLLM to avoid adding new configuration files or interfaces, reducing user learning costs and ensuring ease of use.
- Minimal invasive modifications: minimize invasive modifications to the vLLM code to ensure system maintainability and evolvability.
- Component decoupling: minimize and standardize the coupling between MindSpore large model components and vLLM service components to facilitate the integration of various MindSpore large model suites.

On the basis of the above design principles, vLLM MindSpore adopts the system architecture shown in the figure below, and implements the docking between vLLM and Mindspore in categories of components:

- Service components: vLLM MindSpore maps PyTorch API calls in service components including LLMEngine and Scheduler to MindSpore capabilities, inheriting support for service functions like Continuous Batching and PagedAttention.
- Model components: vLLM MindSpore registers or replaces model components including models, network layers, and custom operators, and integrates MindSpore Transformers, MindSpore One, and other MindSpore large model suites, as well as custom large models, into vLLM.

.. raw:: html

   <table style="width: 100%">
      <tr>
         <td style="text-align: center; width: 100%; border: none">
            <img src="arch.png">
         </td>
      </tr>
   </table>

vLLM MindSpore uses the plugin mechanism recommended by the vLLM community to realize capability registration. In the future, we expect to promote vLLM community to support integration of inference capabilities of third-party AI frameworks, including PaddlePaddle and JAX by following principles described in `[RPC] Multi-framework support for vllm <https://gitee.com/mindspore/vllm-mindspore/issues/IBTNRG>`_ .

Code：<https://gitee.com/mindspore/vllm-mindspore>

Prerequisites
-----------------------------------------------------

- Hardware：Atlas 800I A2 Inference series, or Atlas 800T A2 Training series, with necessary drivers installed and access to the Internet
- Operating System: openEuler or Ubuntu Linux
- Software：

  * Python >= 3.9, < 3.12
  * CANN >= 8.0.0.beta1
  * MindSpore (matched with the vllm-mindspore version)
  * vLLM (matched with the vllm-mindspore version)

Getting Started
-----------------------------------------------------
Please refer to `Quick Start <./getting_started/quick_start/quick_start.html>`_ and `Installation <./getting_started/installation/installation.html>`_ for more details.

Contributing
-----------------------------------------------------
Please read `CONTRIBUTING <./developer_guide/contributing.html>`_  for details on setting up development environments, testing functions, and submitting PR.

We welcome and value any form of contribution and cooperation. Please use `Issue <https://gitee.com/mindspore/vllm-mindspore/issues>`_ to inform us of any bugs you encounter, or to submit your feature requests, improvement suggestions, and technical solutions.

Branch
-----------------------------------------------------
The vllm-mindspore repository contains the main branch, development branch, and version branches:

- **main**: the main branch, compatible with Mindspore master branch and vLLM v0.7.3 version, is continuously monitored for quality through Ascend-MindSpore CI.
- **develop**: the development branch for adapting vLLM features, which is forked from the main branch when a new vLLM version is released. Once the adapted features is stable, it will be merged into the main branch. The current development branch is adapting vLLM v0.8.3 version.
- **rX.Y.Z**: version branches used for archiving version release, which is forked from the main branch after the adaptation of a certain vLLM version is completed.

The following are the version branches:

.. list-table::
   :header-rows: 1

   *  -  Branch
      -  Status
      -  Notes
   *  -  master
      -  Maintained
      -  Compatible with vLLM v0.7.3, and CI commitment for MindSpore master branch
   *  -  develop
      -  Maintained
      -  Compatible with vLLM v0.8.3
   *  -  r0.1
      -  Unmaintained
      -  Only doc fixed is allowed
   *  -  r0.2
      -  Maintained
      -  Compatible with vLLM v0.7.3, and CI commitment for MindSpore 2.6.0

SIG
-----------------------------------------------------
- Welcome to join vLLM MindSpore SIG to participate in the co-construction of open-source projects and industrial cooperation: https://www.mindspore.cn/community/SIG
- SIG meetings, every other Friday or Saturday evening, 20:00 - 21:00 (UTC+8,  `Convert to your timezone <https://dateful.com/convert/gmt8?t=15>`_ )

License
-----------------------------------------------------
Apache License 2.0, as found in the `LICENSE <https://gitee.com/mindspore/vllm-mindspore/blob/master/LICENSE>`_ file.


.. toctree::
   :glob:
   :maxdepth: 2
   :caption: Quick Start
   :hidden:

   getting_started/quick_start/quick_start
   getting_started/installation/installation
   getting_started/tutorials/qwen2.5_7b_singleNPU/qwen2.5_7b_singleNPU
   getting_started/tutorials/qwen2.5_32b_multiNPU/qwen2.5_32b_multiNPU
   getting_started/tutorials/deepseek_parallel/deepseek_r1_671b_w8a8_dp4_tp4_ep4

.. toctree::
   :glob:
   :maxdepth: 1
   :caption: User Guide
   :hidden:

   user_guide/supported_models/models_list/models_list
   user_guide/supported_features/features_list/features_list
   user_guide/supported_features/quantization/quantization
   user_guide/supported_features/profiling/profiling
   user_guide/supported_features/benchmark/benchmark
   user_guide/environment_variables/environment_variables

.. toctree::
   :glob:
   :maxdepth: 1
   :caption: Developer Guide
   :hidden:

   developer_guide/operations/npu_ops
   developer_guide/contributing

.. toctree::
   :glob:
   :maxdepth: 1
   :caption: FAQ
   :hidden:

   faqs/faqs

.. toctree::
   :glob:
   :maxdepth: 1
   :caption: RELEASE NOTES
   :hidden:

   RELEASE