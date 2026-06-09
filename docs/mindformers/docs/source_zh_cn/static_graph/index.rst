静态图实现（Deprecated）
=========================================

.. warning::

   本章节为 **静态图（GRAPH_MODE）实现**，已标记为 **废弃（Deprecated）**。其内容沿用原 1.9.0 版本资料，去除了「模型库」与「安装」两章（已分别并入顶层「模型支持库」与「安装指南」）。新特性请优先查阅 `r2.0.0 动态图实现 <../index.html>`_ 相关文档。

.. toctree::
   :glob:
   :maxdepth: 1
   :caption: 介绍 / 整体架构
   :hidden:

   introduction/overview

.. toctree::
   :glob:
   :maxdepth: 1
   :caption: 大模型全流程指南
   :hidden:

   guide/llm_training
   guide/pre_training
   guide/supervised_fine_tuning
   guide/inference
   guide/deployment
   guide/evaluation

.. toctree::
   :glob:
   :maxdepth: 1
   :caption: 功能特性
   :hidden:

   feature/overview
   feature/start_tasks
   feature/ckpt
   feature/safetensors
   feature/configuration
   feature/load_huggingface_config
   feature/logging
   feature/tokenizer
   feature/dataset
   feature/training_hyperparameters
   feature/monitor
   feature/resume_training
   feature/checkpoint_saving_and_loading
   feature/resume_training2.0
   feature/parallel_training
   feature/high_availability
   feature/memory_optimization
   feature/skip_data_and_ckpt_health_monitor
   feature/pma_fused_checkpoint
   feature/other_training_features
   feature/quantization

.. toctree::
   :glob:
   :maxdepth: 1
   :caption: 高阶开发
   :hidden:

   advanced_development/overview
   advanced_development/precision_optimization
   advanced_development/performance_optimization
   advanced_development/dev_migration
   advanced_development/yaml_config_inference
   advanced_development/inference_precision_comparison
   advanced_development/accuracy_comparison
   advanced_development/training_template_instruction
   advanced_development/weight_transfer

.. toctree::
   :glob:
   :maxdepth: 1
   :caption: 优秀实践
   :hidden:

   example/docker-installation
   example/distilled/distilled
   example/convert_ckpt_to_megatron/convert_ckpt_to_megatron
   example/model_test/model_test
   example/finetune_with_glm4/finetune_with_glm4

.. toctree::
   :glob:
   :maxdepth: 1
   :caption: 环境变量
   :hidden:

   env_variables
