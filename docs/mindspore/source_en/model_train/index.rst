Model Building and Training
=================================

.. toctree::
   :glob:
   :maxdepth: 1
   :hidden:
   :caption: Programming Forms

   program_form/overview
   program_form/pynative
   program_form/static_graph

.. toctree::
   :glob:
   :maxdepth: 1
   :hidden:
   :caption: Data Loading and Processing

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
   :caption: Model Building

   model_building/overview
   model_building/tensor_and_parameter
   model_building/functional_and_cell

.. toctree::
   :glob:
   :maxdepth: 1
   :hidden:
   :caption: Training Process

   train_process/overview
   train_process/model
   train_process/train_optimize
   train_process/derivation
   train_process/algorithm_optimize

.. toctree::
   :glob:
   :maxdepth: 1
   :hidden:
   :caption: Distributed Parallelism

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
   :caption: Custom High-order Programming

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
   :caption: Training High Availability

   train_availability/fault_recover
   train_availability/graceful_exit
   train_availability/storage_sys

.. toctree::
   :glob:
   :maxdepth: 1
   :hidden:
   :caption: Model Debugging

   debug/overview
   debug/dump
   debug/rdr
   debug/sdc
   debug/error_analysis
   debug/pynative
   ErrorMap↗ <https://www.hiascend.com/forum/thread-0229108045633055169-1-1.html>

.. toctree::
   :glob:
   :maxdepth: 1
   :hidden:
   :caption: Model Tuning

   optimize/overview
   optimize/graph_fusion_engine
   optimize/mem_reuse
   optimize/aoe

.. toctree::
   :glob:
   :maxdepth: 1
   :hidden:
   :caption: Models Library

   models/official_models

Building
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
                              <span class="doc-head-content">Programming Forms</span>
                           </div>
                           <div class="doc-article-desc">
                              Provide dynamic diagrams, static diagrams, dynamic and static unified programming form, so that developers can take into account the development efficiency and execution performance.
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
                              <span class="doc-head-content">Data Loading and Processing</span>
                           </div>
                           <div class="doc-article-desc">
                              Two data processing models, Data Processing Pipeline and Data Processing Lightweight.
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
                              <span class="doc-head-content">Model Building</span>
                           </div>
                           <div class="doc-article-desc">
                              Efficiently build and manage complex neural network architectures by utilizing the ideas of functional and object fusion programming.
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
                              <span class="doc-head-content">Model Library</span>
                           </div>
                           <div class="doc-article-desc">
                              Reference cases of model implementations based on various types of kits.
                           </div>
                        </div>
                     </a>
                  </div>
               </div>
         </div>
      </div>
   </div>

Training
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
                              <span class="doc-head-content">Training Processes</span>
                           </div>
                           <div class="doc-article-desc">
                              Build a complete training process that includes dataset preprocessing, model creation, defining loss functions and optimizers, training and saving the model.
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
                              <span class="doc-head-content">Distributed Parallelism</span>
                           </div>
                           <div class="doc-article-desc">
                              Distributed parallelism reduces the need for hardware such as memory and computational performance, providing powerful computational capabilities and performance advantages for processing large-scale data and complex models.
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
                                 <span class="doc-head-content">Customized Higher-Order Programming</span>
                              </div>
                              <div class="doc-article-desc">
                                 When the advanced methods provided by the framework can not meet certain scenarios, or have high performance requirements, you can use a customized method to add or modify certain processes to meet development or debugging needs.
                              </div>
                           </div>
                        </a>
                     </div>
                  </div>
            </div>
         </div>
   </div>

Debugging and Optimization
-----------------------------

.. raw:: html

   <div class="container">
         <div class="row">
            <div class="col-md-6">
               <div class="doc-article-list">
                  <div class="doc-article-item">
                     <a href="./train_availability/fault_recover.html" class="article-link">
                        <div>
                           <div class="doc-article-head">
                              <span class="doc-head-content">Training High Availability</span>
                           </div>
                           <div class="doc-article-desc">
                              Failure recovery during training and other high availability functions.
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
                              <span class="doc-head-content">Model Debugging</span>
                           </div>
                           <div class="doc-article-desc">
                              Model debugging methods and examples such as Dump, feature value detection, etc.
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
                                 <span class="doc-head-content">Model Tuning</span>
                              </div>
                              <div class="doc-article-desc">
                                 Model tuning methods and cases, such as graph-kernel fusion, memory reuse, etc.
                              </div>
                           </div>
                        </a>
                     </div>
                  </div>
            </div>
         </div>
   </div>
