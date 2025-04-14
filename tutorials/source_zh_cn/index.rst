.. MindSpore documentation master file, created by
   sphinx-quickstart on Thu Mar 24 11:00:00 2021.
   You can adapt this file completely to your liking, but it should at least
   contain the root `toctree` directive.

MindSpore教程
=====================

.. toctree::
   :glob:
   :maxdepth: 1
   :caption: 快速上手
   :hidden:

   beginner/introduction
   beginner/quick_start
   beginner/tensor
   beginner/dataset
   beginner/model
   beginner/autograd
   beginner/train
   beginner/save_load
   beginner/accelerate_with_static_graph
   beginner/mixed_precision

.. toctree::
   :glob:
   :maxdepth: 1
   :caption: 数据处理
   :hidden:

   dataset/sampler
   dataset/record
   dataset/eager
   dataset/python_objects
   dataset/augment
   dataset/cache
   dataset/optimize

.. toctree::
   :glob:
   :maxdepth: 1
   :caption: 编译
   :hidden:

   compile/static_graph
   compile/operators
   compile/statements
   compile/python_builtin_functions
   compile/static_graph_expert_programming

.. toctree::
   :glob:
   :maxdepth: 1
   :caption: 并行
   :hidden:

   parallel/overview
   parallel/startup_method
   parallel/data_parallel
   parallel/operator_parallel
   parallel/optimizer_parallel
   parallel/pipeline_parallel
   parallel/optimize_technique
   parallel/distributed_case

.. toctree::
   :glob:
   :maxdepth: 1
   :caption: 调试调优
   :hidden:

   debug/pynative
   debug/dump
   debug/sdc
   debug/profiler
   debug/error_analysis
   debug/dryrun

.. toctree::
   :glob:
   :maxdepth: 1
   :caption: 自定义编程
   :hidden:

   custom_program/op_custom
   custom_program/fusion_pass
   custom_program/hook_program

.. toctree::
   :glob:
   :maxdepth: 1
   :caption: 推理
   :hidden:

   model_infer/ms_infer/llm_inference_overview
   model_infer/ms_infer/weight_prepare
   model_infer/ms_infer/model_dev
   model_infer/ms_infer/parallel
   model_infer/ms_infer/weight_split
   model_infer/ms_infer/quantization
   model_infer/lite_infer/overview

.. toctree::
   :glob:
   :maxdepth: 1
   :caption: 高可用
   :hidden:

   train_availability/fault_recover
   train_availability/graceful_exit

.. toctree::
   :glob:
   :maxdepth: 1
   :caption: 香橙派
   :hidden:

   orange_pi/overview
   orange_pi/environment_setup
   orange_pi/model_infer
   orange_pi/dev_start

.. toctree::
   :glob:
   :maxdepth: 1
   :caption: 模型案例
   :hidden:

   model_migration/model_migration
   cv
   nlp
   generative

.. raw:: html

   <div class="container">
      <div class="row">
         <div class="col-md-6">
               <div class="doc-article-list">
                  <div class="doc-article-item">
                     <a href="./beginner/introduction.html" class="article-link">
                        <div>
                           <div class="doc-article-head">
                              <span class="doc-head-content">快速上手</span>
                           </div>
                           <div class="doc-article-desc">
                              贯穿MindSpore深度学习的基本流程，包括数据处理、模型加载与保存、图模式加速等实践案例。
                           </div>
                        </div>
                     </a>
                  </div>
               </div>
         </div>
         <div class="col-md-6">
               <div class="doc-article-list">
                  <div class="doc-article-item">
                     <a href="./dataset/sampler.html" class="article-link">
                        <div>
                           <div class="doc-article-head">
                              <span class="doc-head-content">数据处理</span>
                           </div>
                           <div class="doc-article-desc">
                              提供数据处理相关增强、缓存、pipeline等功能案例。
                           </div>
                        </div>
                     </a>
                  </div>
               </div>
         </div>
      </div>
      <div class="row">
         <div class="col-md-6">
               <div class="doc-article-list">
                  <div class="doc-article-item">
                     <a href="./compile/static_graph.html" class="article-link">
                        <div>
                           <div class="doc-article-head">
                              <span class="doc-head-content">编译</span>
                           </div>
                           <div class="doc-article-desc">
                              提供MindSpore编译语法支持案例以及图模式编程案例。
                           </div>
                        </div>
                     </a>
                  </div>
               </div>
         </div>
         <div class="col-md-6">
               <div class="doc-article-list">
                  <div class="doc-article-item">
                     <a href="./parallel/overview.html" class="article-link">
                        <div>
                           <div class="doc-article-head">
                              <span class="doc-head-content">并行</span>
                           </div>
                           <div class="doc-article-desc">
                              提供数据并行、算子级并行、优化器并行等实践案例和优化策略。
                           </div>
                        </div>
                     </a>
                  </div>
               </div>
         </div>
      </div>
      <div class="row">
         <div class="col-md-6">
               <div class="doc-article-list">
                  <div class="doc-article-item">
                     <a href="./debug/pynative.html" class="article-link">
                        <div>
                           <div class="doc-article-head">
                              <span class="doc-head-content">调试调优</span>
                           </div>
                           <div class="doc-article-desc">
                              提供dump、profiler、dryrun等功能调试案例。
                           </div>
                        </div>
                     </a>
                  </div>
               </div>
         </div>
         <div class="col-md-6">
               <div class="doc-article-list">
                  <div class="doc-article-item">
                     <a href="./custom_program/op_custom.html" class="article-link">
                        <div>
                           <div class="doc-article-head">
                              <span class="doc-head-content">自定义编程</span>
                           </div>
                           <div class="doc-article-desc">
                              提供自定义算子、自定义融合实践案例。
                           </div>
                        </div>
                     </a>
                  </div>
               </div>
         </div>
      </div>
      <div class="row">
         <div class="col-md-6">
               <div class="doc-article-list">
                  <div class="doc-article-item">
                     <a href="./model_infer/ms_infer/llm_inference_overview.html" class="article-link">
                        <div>
                           <div class="doc-article-head">
                              <span class="doc-head-content">推理</span>
                           </div>
                           <div class="doc-article-desc">
                              介绍MindSpore推理端到端流程，包括模型构建、权重切分等功能。
                           </div>
                        </div>
                     </a>
                  </div>
               </div>
         </div>
         <div class="col-md-6">
               <div class="doc-article-list">
                  <div class="doc-article-item">
                     <a href="./train_availability/fault_recover.html" class="article-link">
                        <div>
                           <div class="doc-article-head">
                              <span class="doc-head-content">高可用</span>
                           </div>
                           <div class="doc-article-desc">
                              介绍训练高可用特性，包括故障恢复、进程优雅退出等功能。
                           </div>
                        </div>
                     </a>
                  </div>
               </div>
         </div>
      </div>
      <div class="row">
         <div class="col-md-6">
               <div class="doc-article-list">
                  <div class="doc-article-item">
                     <a href="./orange_pi/overview.html" class="article-link">
                        <div>
                           <div class="doc-article-head">
                              <span class="doc-head-content">香橙派</span>
                           </div>
                           <div class="doc-article-desc">
                              提供香橙派环境搭建、开发、推理等功能案例。
                           </div>
                        </div>
                     </a>
                  </div>
               </div>
         </div>
         <div class="col-md-6">
               <div class="doc-article-list">
                  <div class="doc-article-item">
                     <a href="./model_migration/model_migration.html" class="article-link">
                        <div>
                           <div class="doc-article-head">
                              <span class="doc-head-content">模型案例</span>
                           </div>
                           <div class="doc-article-desc">
                              提供模型迁移指导以及各类CV、NLP、生成式模型构建案例。
                           </div>
                        </div>
                     </a>
                  </div>
               </div>
         </div>
      </div>
   </div>