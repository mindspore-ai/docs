.. MindSpore documentation master file, created by
   sphinx-quickstart on Thu Mar 24 11:00:00 2022.
   You can adapt this file completely to your liking, but it should at least
   contain the root `toctree` directive.

MindSpore 文档
=========================

.. toctree::
   :glob:
   :maxdepth: 1
   :caption: 设计

   design/technical_white_paper
   design/all_scenarios_architecture
   design/gradient
   design/dynamic_graph_and_static_graph
   design/distributed_training_design
   design/distributed_advanced
   design/heterogeneous_training
   design/mindir
   design/data_engine
   design/dataset_offload
   design/graph_kernel_fusion
   design/jit_fallback
   可视化调试调优↗ <https://www.mindspore.cn/mindinsight/docs/zh-CN/r1.7/training_visual_design.html>
   安全可信↗ <https://www.mindspore.cn/mindarmour/docs/zh-CN/r1.7/design.html>
   design/glossary
   
.. toctree::
   :glob:
   :maxdepth: 1
   :caption: 参考

   note/benchmark
   note/network_list
   note/operator_list
   note/env_var_list
   note/syntax_list
   note/api_mapping

.. toctree::
   :glob:
   :maxdepth: 1
   :caption: API

   api_python/mindspore
   api_python/mindspore.common.initializer
   api_python/mindspore.communication
   api_python/mindspore.context
   api_python/mindspore.dataset
   api_python/mindspore.dataset.audio
   api_python/mindspore.dataset.config
   api_python/mindspore.dataset.text
   api_python/mindspore.dataset.transforms
   api_python/mindspore.dataset.vision
   api_python/mindspore.mindrecord
   api_python/mindspore.nn
   api_python/mindspore.nn.probability
   api_python/mindspore.nn.transformer
   api_python/mindspore.numpy
   api_python/mindspore.ops
   api_python/mindspore.ops.functional
   api_python/mindspore.parallel
   api_python/mindspore.parallel.nn
   api_python/mindspore.profiler
   api_python/mindspore.scipy
   api_python/mindspore.train
   api_python/mindspore.boost
   C++ API↗ <https://www.mindspore.cn/lite/api/zh-CN/r1.7/api_cpp/mindspore.html>

.. toctree::
   :glob:
   :maxdepth: 1
   :caption: 迁移指南

   migration_guide/overview
   migration_guide/preparation
   migration_guide/script_analysis
   migration_guide/script_development
   migration_guide/neural_network_debug
   migration_guide/accuracy_optimization
   migration_guide/performance_optimization
   migration_guide/inference
   migration_guide/sample_code
   migration_guide/faq
   
.. toctree::
   :glob:
   :maxdepth: 1
   :caption: FAQ

   faq/installation
   faq/data_processing
   faq/implement_problem
   faq/network_compilation
   faq/operators_compile
   faq/usage_migrate_3rd
   faq/performance_tuning
   faq/precision_tuning
   faq/distributed_configure
   faq/inference
   faq/feature_advice