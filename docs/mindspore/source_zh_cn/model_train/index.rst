模型构建与训练
=========================

.. toctree::
   :glob:
   :maxdepth: 1
   :hidden:
   :caption: 编程形态

   program_form/overview
   program_form/pynative
   program_form/static_graph

.. toctree::
   :glob:
   :maxdepth: 1
   :hidden:
   :caption: 数据加载与处理

   dataset/overview
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
   :hidden:
   :caption: 模型构建

   model_building/overview
   model_building/tensor_and_parameter
   model_building/functional_and_cell

.. toctree::
   :glob:
   :maxdepth: 1
   :hidden:
   :caption: 训练流程

   train_process/overview
   train_process/model
   train_process/train_optimize
   train_process/derivation
   train_process/algorithm_optimize

.. toctree::
   :glob:
   :maxdepth: 1
   :hidden:
   :caption: 分布式并行

   parallel/overview
   parallel/startup_method
   parallel/data_parallel
   parallel/semi_auto_parallel
   parallel/auto_parallel
   parallel/manual_parallel
   parallel/parameter_server_training
   parallel/model_save_load
   parallel/recover
   parallel/optimize_technique
   parallel/others
   parallel/distributed_case

.. toctree::
   :glob:
   :maxdepth: 1
   :hidden:
   :caption: 自定义高阶编程

   custom_program/overview
   custom_program/op_custom
   custom_program/initializer
   custom_program/loss
   custom_program/optimizer
   custom_program/fusion_pass
   custom_program/network_custom
   custom_program/hook_program

.. toctree::
   :glob:
   :maxdepth: 1
   :hidden:
   :caption: 训练高可用

   train_availability/fault_recover
   train_availability/graceful_exit
   train_availability/mindio_ttp
   train_availability/storage_sys

.. toctree::
   :glob:
   :maxdepth: 1
   :hidden:
   :caption: 模型调试

   debug/overview
   debug/dump
   debug/rdr
   debug/sdc
   debug/error_analysis
   debug/pynative
   报错地图↗ <https://www.hiascend.com/forum/thread-0229108045633055169-1-1.html>

.. toctree::
   :glob:
   :maxdepth: 1
   :hidden:
   :caption: 模型调优

   optimize/overview
   optimize/graph_fusion_engine
   optimize/mem_reuse
   optimize/aoe
   optimize/profiler

.. toctree::
   :glob:
   :maxdepth: 1
   :hidden:
   :caption: 模型库

   models/official_models

构建
---------

.. raw:: html

   <div class="container">
      <div class="row">
         <div class="col-md-6">
               <div class="doc-article-list">
                  <div class="doc-article-item">
                     <a href="./program_form/overview.html" class="article-link">
                        <div>
                           <div class="doc-article-head">
                              <span class="doc-head-content">编程形态</span>
                           </div>
                           <div class="doc-article-desc">
                              提供动态图、静态图、动静统一的编程形态，使开发者可以兼顾开发效率和执行性能。
                           </div>
                        </div>
                     </a>
                  </div>
               </div>
         </div>
         <div class="col-md-6">
               <div class="doc-article-list">
                  <div class="doc-article-item">
                     <a href="./dataset/overview.html" class="article-link">
                        <div>
                           <div class="doc-article-head">
                              <span class="doc-head-content">数据加载与处理</span>
                           </div>
                           <div class="doc-article-desc">
                              数据处理Pipeline和数据处理轻量化两种数据处理模式。
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
                     <a href="./model_building/overview.html" class="article-link">
                        <div>
                           <div class="doc-article-head">
                              <span class="doc-head-content">模型构建</span>
                           </div>
                           <div class="doc-article-desc">
                              利用函数式和对象式融合编程的思想，高效地构建和管理复杂的神经网络架构。
                           </div>
                        </div>
                     </a>
                  </div>
               </div>
         </div>
         <div class="col-md-6">
               <div class="doc-article-list">
                  <div class="doc-article-item">
                     <a href="./models/official_models.html" class="article-link">
                        <div>
                           <div class="doc-article-head">
                              <span class="doc-head-content">模型库</span>
                           </div>
                           <div class="doc-article-desc">
                              基于各类套件的模型实现参考案例。
                           </div>
                        </div>
                     </a>
                  </div>
               </div>
         </div>
      </div>
   </div>

训练
---------

.. raw:: html

   <div class="container">
         <div class="row">
            <div class="col-md-6">
               <div class="doc-article-list">
                  <div class="doc-article-item">
                     <a href="./train_process/overview.html" class="article-link">
                        <div>
                           <div class="doc-article-head">
                              <span class="doc-head-content">训练流程</span>
                           </div>
                           <div class="doc-article-desc">
                              搭建一个包括数据集预处理、模型创建、定义损失函数和优化器、训练及保存模型的完整训练流程。
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
                              <span class="doc-head-content">分布式并行</span>
                           </div>
                           <div class="doc-article-desc">
                              通过分布式并行，降低对内存、计算性能等硬件的需求，为处理大规模数据和复杂模型提供了强大的计算能力和性能优势。
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
                        <a href="./custom_program/overview.html" class="article-link">
                           <div>
                              <div class="doc-article-head">
                                 <span class="doc-head-content">自定义高阶编程</span>
                              </div>
                              <div class="doc-article-desc">
                                 当框架提供的高级方法不能满足某些场景，或对性能有较高要求时，可以采用自定义的方法添加或修改某些流程，以满足开发或调试需求。
                              </div>
                           </div>
                        </a>
                     </div>
                  </div>
            </div>
         </div>
   </div>

调试调优
----------

.. raw:: html

   <div class="container">
         <div class="row">
            <div class="col-md-6">
               <div class="doc-article-list">
                  <div class="doc-article-item">
                     <a href="./train_availability/fault_recover.html" class="article-link">
                        <div>
                           <div class="doc-article-head">
                              <span class="doc-head-content">训练高可用</span>
                           </div>
                           <div class="doc-article-desc">
                              训练过程中的故障恢复等高可用相关功能。
                           </div>
                        </div>
                     </a>
                  </div>
               </div>
            </div>
            <div class="col-md-6">
               <div class="doc-article-list">
                  <div class="doc-article-item">
                     <a href="./debug/overview.html" class="article-link">
                        <div>
                           <div class="doc-article-head">
                              <span class="doc-head-content">模型调试</span>
                           </div>
                           <div class="doc-article-desc">
                              模型调试方法和案例，如Dump、特征值检测等。
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
                        <a href="./optimize/overview.html" class="article-link">
                           <div>
                              <div class="doc-article-head">
                                 <span class="doc-head-content">模型调优</span>
                              </div>
                              <div class="doc-article-desc">
                                 模型调优方法和案例，如图算融合、内存复用等。
                              </div>
                           </div>
                        </a>
                     </div>
                  </div>
            </div>
         </div>
   </div>
