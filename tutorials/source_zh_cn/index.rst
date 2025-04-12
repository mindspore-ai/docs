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
   train_availability/disaster_recover
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
   :caption: 实践案例
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
                     <a href="./beginner/quick_start.html" class="article-link">
                        <div>
                           <div class="doc-article-head">
                              <span class="doc-head-content">快速入门</span>
                           </div>
                           <div class="doc-article-desc">
                              贯穿MindSpore深度学习的基本流程，以LeNet5网络模型为例子，实现深度学习中的常见任务。
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
                              <span class="doc-head-content">计算机视觉实践案例</span>
                           </div>
                           <div class="doc-article-desc">
                              提供计算机视觉领域的典型网络实现，以及完整的训练、评估、预测流程。
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
                              <span class="doc-head-content">自然语言处理实践案例</span>
                           </div>
                           <div class="doc-article-desc">
                              提供自然语言处理领域的典型网络实现，以及完整的训练、评估、预测流程。
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
                              <span class="doc-head-content">生成式实践案例</span>
                           </div>
                           <div class="doc-article-desc">
                              提供生成式领域的典型网络实现，以及完整的训练、评估、预测流程。
                           </div>
                        </div>
                     </a>
                  </div>
					</div>
			</div>
	   </div>
	</div>
