vLLM MindSpore 文档
=========================================

vLLM MindSpore 简介
-----------------------------------------------------
vLLM MindSpore插件（`vllm-mindspore`）是一个由 `MindSpore社区 <https://www.mindspore.cn/>`_ 孵化的vLLM后端插件。其将基于MindSpore构建的大模型推理能力接入 `vLLM <https://github.com/vllm-project/vllm>`_ ，从而有机整合MindSpore和vLLM的技术优势，提供全栈开源、高性能、易用的大模型推理解决方案。

vLLM是由加州大学伯克利分校Sky Computing Lab创建的社区开源项目，已广泛用于学术研究和工业应用。vLLM以Continuous Batching调度机制和PagedAttention Key-Value缓存管理为基础，提供了丰富的推理服务功能，包括投机推理、Prefix Caching、Multi-LoRA等。同时，vLLM已支持种类丰富的开源大模型，包括Transformer类（如LLaMa）、混合专家类（如DeepSeek）、Embedding类（如E5-Mistral）、多模态类（如LLaVA）等。由于vLLM选用PyTorch构建大模型和管理计算存储资源，此前无法使用其部署基于MindSpore大模型的推理服务。

vLLM MindSpore插件以将MindSpore大模型接入vLLM，并实现服务化部署为功能目标。其遵循以下设计原则：

- 接口兼容：支持vLLM原生的API和服务部署接口，避免新增配置文件或接口，降低用户学习成本和确保易用性。
- 最小化侵入式修改：尽可能避免侵入式修改vLLM代码，以保障系统的可维护性和可演进性。
- 组件解耦：最小化和规范化MindSpore大模型组件和vLLM服务组件的耦合面，以利于多种MindSpore大模型套件接入。

基于上述设计原则，vLLM MindSpore采用如下图所示的系统架构，分组件类别实现vLLM与MindSpore的对接：

- 服务化组件：通过将LLM Engine、Scheduler等服务化组件中的PyTorch API调用映射至MindSpore能力调用，继承支持包括Continuous Batching、PagedAttention在内的服务化功能。
- 大模型组件：通过注册或替换模型、网络层、自定义算子等组件，将MindSpore Transformers、MindSpore One等MindSpore大模型套件和自定义大模型接入vLLM。

.. raw:: html

   <table style="width: 100%">
      <tr>
         <td style="text-align: center; width: 100%; border: none">
            <img src="arch.cn.png">
         </td>
      </tr>
   </table>

vLLM MindSpore采用vLLM社区推荐的插件机制，实现能力注册。未来期望遵循 `RPC Multi-framework support for vllm <https://gitee.com/mindspore/vllm-mindspore/issues/IBTNRG>`_ 所述原则。

代码仓地址：<https://gitee.com/mindspore/vllm-mindspore>

环境准备
-----------------------------------------------------

- 硬件：Atlas 800I A2推理服务器，或Atlas 800T A2推理服务器，已安装必要的驱动程序，并可连接至互联网
- 操作系统：openEuler或Ubuntu Linux
- 软件：

  * Python >= 3.9, < 3.12
  * CANN >= 8.0.0.beta1
  * MindSpore (与vLLM MindSpore版本配套)
  * vLLM (与vLLM MindSpore版本配套)

快速体验
-----------------------------------------------------
请查看 `快速开始 <./getting_started/quick_start/quick_start.html>`_ 和 `安装指南 <./getting_started/installation/installation.html>`_ 了解更多。

贡献
-----------------------------------------------------
请参考  `CONTRIBUTING <./developer_guide/contributing.html>`_  文档了解更多关于开发环境搭建、功能测试以及 PR 提交规范的信息。

我们欢迎并重视任何形式的贡献与合作，请通过 `Issue <https://gitee.com/mindspore/vllm-mindspore/issues>`_ 来告知我们您遇到的任何Bug，或提交您的特性需求、改进建议、技术方案。

分支策略
-----------------------------------------------------
vLLM MindSpore代码仓包含主干分支、开发分支、版本分支：

- **main**: 主干分支，与MindSpore master分支和vLLM v0.7.3版本配套，并通过昇腾+昇思CI持续进行质量看护；
- **develop**: 开发分支，在vLLM部分新版本发布时从主干分支拉出，用于开发适配vLLM的新功能特性。待特性适配稳定后合入主干分支。当前开发分支正在适配vLLM v0.8.3版本；
- **rX.Y.Z**: 版本分支，在完成vLLM某版本适配后，从主干分支拉出，用于正式版本发布归档。

下面是维护中的版本分支：

.. list-table::
   :header-rows: 1

   *  -  分支
      -  状态
      -  备注
   *  -  master
      -  Maintained
      -  基于vLLM v0.7.3版本和MindSpore master分支CI看护
   *  -  develop
      -  Maintained
      -  基于vLLM v0.8.3版本
   *  -  r0.1
      -  Unmaintained
      -  仅允许文档修复
   *  -  r0.2
      -  Maintained
      -  基于vLLM v0.7.3版本和MindSpore 2.6.0版本CI看护

SIG组织
-----------------------------------------------------
- 欢迎加入LLM Infercence Serving，参与开源项目共建和产业合作：https://www.mindspore.cn/community/SIG
- SIG例会，双周周五或周六晚上，20:00 - 21:00 (UTC+8,  `查看您的时区 <https://dateful.com/convert/gmt8?t=15>`_ )

许可证
-----------------------------------------------------
Apache 许可证 2.0，如  `LICENSE <https://gitee.com/mindspore/vllm-mindspore/blob/master/LICENSE>`_  文件中所示。


.. toctree::
   :glob:
   :maxdepth: 2
   :caption: 快速开始
   :hidden:

   getting_started/quick_start/quick_start
   getting_started/installation/installation
   getting_started/tutorials/qwen2.5_7b_singleNPU/qwen2.5_7b_singleNPU
   getting_started/tutorials/qwen2.5_32b_multiNPU/qwen2.5_32b_multiNPU
   getting_started/tutorials/deepseek_parallel/deepseek_r1_671b_w8a8_dp4_tp4_ep4

.. toctree::
   :glob:
   :maxdepth: 1
   :caption: 使用指南
   :hidden:

   user_guide/supported_models/models_list/models_list
   user_guide/supported_features/features_list/features_list
   user_guide/supported_features/operations/npu_ops
   user_guide/supported_features/quantization/quantization
   user_guide/supported_features/profiling/profiling
   user_guide/supported_features/benchmark/benchmark
   user_guide/environment_variables/environment_variables

.. toctree::
   :glob:
   :maxdepth: 1
   :caption: 开发者指南
   :hidden:

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