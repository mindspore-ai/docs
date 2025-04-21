.. MindSpore documentation master file, created by
   sphinx-quickstart on Thu Mar 24 11:00:00 2021.
   You can adapt this file completely to your liking, but it should at least
   contain the root `toctree` directive.

MindSpore Tutorial
=========================

.. toctree::
   :glob:
   :maxdepth: 1
   :caption: Quick Start
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
   :caption: Data Processing
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
   :caption: Compilation
   :hidden:

   compile/static_graph
   compile/operators
   compile/statements
   compile/python_builtin_functions
   compile/static_graph_expert_programming

.. toctree::
   :glob:
   :maxdepth: 1
   :caption: Parallel
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
   :caption: Debugging and Tuning
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
   :caption: Custom programming
   :hidden:

   custom_program/op_custom
   custom_program/fusion_pass
   custom_program/hook_program

.. toctree::
   :glob:
   :maxdepth: 1
   :caption: Infer
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
   :caption: High Availability
   :hidden:

   train_availability/fault_recover
   train_availability/graceful_exit

.. toctree::
   :glob:
   :maxdepth: 1
   :caption: Orange Pi
   :hidden:

   orange_pi/overview
   orange_pi/environment_setup
   orange_pi/model_infer
   orange_pi/dev_start

.. toctree::
   :glob:
   :maxdepth: 1
   :caption: Model Cases
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
                              <span class="doc-head-content">Quick Start</span>
                           </div>
                           <div class="doc-article-desc">
                              Basic processes of MindSpore deep learning, including data processing, model loading and saving, and graph mode acceleration.
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
                              <span class="doc-head-content">Data Processing</span>
                           </div>
                           <div class="doc-article-desc">
                              Provide cases of data processing enhancement, cache, and pipeline functions.
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
                              <span class="doc-head-content">Compilation</span>
                           </div>
                           <div class="doc-article-desc">
                              Provide MindSpore compilation syntax support cases and graph mode programming cases.
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
                              <span class="doc-head-content">Parallel</span>
                           </div>
                           <div class="doc-article-desc">
                              Provide practice cases and optimization policies, such as data parallelism, operator-level parallelism, and optimizer parallelism.
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
                              <span class="doc-head-content">Debugging and Tuning</span>
                           </div>
                           <div class="doc-article-desc">
                              Provide debugging cases for dump, profiler, and dryrun functions.
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
                              <span class="doc-head-content">Custom programming</span>
                           </div>
                           <div class="doc-article-desc">
                              Provide custom operator and customized convergence practice cases.
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
                              <span class="doc-head-content">Infer</span>
                           </div>
                           <div class="doc-article-desc">
                              Describe the device-to-device process of MindSpore inference, including model building and weight segmentation.
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
                              <span class="doc-head-content">High Availability</span>
                           </div>
                           <div class="doc-article-desc">
                              Describe the training HA feature, including fault recovery and graceful process exit.
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
                              <span class="doc-head-content">Orange Pi</span>
                           </div>
                           <div class="doc-article-desc">
                              Provide cases of setting up, developing, and reasoning the Orange Pie environment.
                           </div>
                        </div>
                     </a>
                  </div>
               </div>
         </div>
         <div class="col-md-6">
               <div class="doc-article-list">
                  <div class="doc-article-item">
                     <a href="./model_migration/cv.html" class="article-link">
                        <div>
                           <div class="doc-article-head">
                              <span class="doc-head-content">Model Cases</span>
                           </div>
                           <div class="doc-article-desc">
                              Provide model migration guidance and various CV, NLP, and generative model building cases.
                           </div>
                        </div>
                     </a>
                  </div>
               </div>
         </div>
      </div>
   </div>