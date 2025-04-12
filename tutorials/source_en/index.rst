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
   train_availability/disaster_recover
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
   :caption: Practical Cases
   :hidden:

   cv
   nlp
   generative

.. raw:: html

   <div class="container">
	   <div class="row">
         <div class="col-md-6">
               <div class="doc-article-list">
                  <div class="doc-article-item">
                     <a href="./beginner/quick_start.html" class="article-link">
                        <div>
                           <div class="doc-article-head">
                              <span class="doc-head-content">Quick Start</span>
                           </div>
                           <div class="doc-article-desc">
                              Run through the basic process of MindSpore deep learning, using the LeNet5 network model as an example of a common task in deep learning.
                           </div>
                        </div>
                     </a>
                  </div>
					</div>
			</div>
			<div class="col-md-6">
               <div class="doc-article-list">
                  <div class="doc-article-item">
                     <a href="./cv/resnet50.html" class="article-link">
                        <div>
                           <div class="doc-article-head">
                              <span class="doc-head-content">Computer Vision Practice Cases</span>
                           </div>
                           <div class="doc-article-desc">
                              Provides typical network implementations in the field of computer vision, as well as a complete training, evaluation, and prediction process.
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
                     <a href="./nlp/sentiment_analysis.html" class="article-link">
                        <div>
                           <div class="doc-article-head">
                              <span class="doc-head-content">Examples of Natural Language Processing in Practice</span>
                           </div>
                           <div class="doc-article-desc">
                              Provides typical network implementations in the field of natural language processing, as well as a complete training, evaluation, and prediction process.
                           </div>
                        </div>
                     </a>
                  </div>
					</div>
			</div>
         <div class="col-md-6">
               <div class="doc-article-list">
                  <div class="doc-article-item">
                     <a href="./generative/gan.html" class="article-link">
                        <div>
                           <div class="doc-article-head">
                              <span class="doc-head-content">Generative Practice Examples</span>
                           </div>
                           <div class="doc-article-desc">
                              Provides typical network implementations in the generative field, as well as a complete training, evaluation, and prediction process.
                           </div>
                        </div>
                     </a>
                  </div>
					</div>
			</div>
	   </div>
	</div>
