.. MindSpore documentation master file, created by
   sphinx-quickstart on Thu Mar 24 11:00:00 2020.
   You can adapt this file completely to your liking, but it should at least
   contain the root `toctree` directive.

MindSpore Programming Guide
=================================

.. toctree::
   :glob:
   :maxdepth: 1
   :caption: Overview
   :hidden:

   architecture
   api_structure

.. toctree::
   :glob:
   :maxdepth: 1
   :caption: Design
   :hidden:

   design/technical_white_paper
   design/gradient
   design/distributed_training_design
   design/mindir
   Design of Visualization↗ <https://www.mindspore.cn/mindinsight/docs/en/master/training_visual_design.html>
   design/glossary

.. toctree::
   :glob:
   :maxdepth: 1
   :caption: Quickstart
   :hidden:

   Implementing Simple Linear Function Fitting↗ <https://www.mindspore.cn/tutorials/en/master/linear_regression.html>
   Implementing an Image Classification Application↗ <https://www.mindspore.cn/tutorials/en/master/quick_start.html>

.. toctree::
   :glob:
   :maxdepth: 1
   :caption: Basic Concepts
   :hidden:

   dtype
   tensor
   parameter_introduction
   operators
   cell
   dataset_introduction

.. toctree::
   :glob:
   :maxdepth: 1
   :caption: Data Pipeline
   :hidden:

   dataset_sample
   dataset
   pipeline
   dataset_advanced
   dataset_usage

.. toctree::
   :glob:
   :maxdepth: 1
   :caption: Build the Network
   :hidden:

   build_net
   initializer
   parameter
   control_flow
   indefinite_parameter
   loss
   grad_operation
   constexpr
   hypermap
   optim
   train_and_eval

.. toctree::
   :glob:
   :maxdepth: 1
   :caption: Model Running
   :hidden:

   context
   run
   save_and_load_models
   model

.. toctree::
   :glob:
   :maxdepth: 1
   :caption: Inference
   :hidden:

   multi_platform_inference
   online_inference
   offline_inference

.. toctree::
   :glob:
   :maxdepth: 1
   :caption: Distributed Training
   :hidden:

   distributed_training
   distributed_advanced
   distributed_example

.. toctree::
   :glob:
   :maxdepth: 1
   :caption: PyNative
   :hidden:

   debug_in_pynative_mode

.. toctree::
   :glob:
   :maxdepth: 1
   :caption: Numpy
   :hidden:

   numpy

.. toctree::
   :glob:
   :maxdepth: 1
   :caption: Function Debugging
   :hidden:

   read_ir_files
   Debugging in PyNative Mode↗ <https://www.mindspore.cn/docs/programming_guide/en/master/debug_in_pynative_mode.html>
   dump_in_graph_mode
   custom_debugging_info
   incremental_operator_build

.. toctree::
   :glob:
   :maxdepth: 1
   :caption: Performance Optimization
   :hidden:

   enable_mixed_precision
   enable_auto_tune
   enable_dataset_autotune
   enable_dataset_offload
   apply_gradient_accumulation
   Debugging performance with Profiler↗ <https://www.mindspore.cn/mindinsight/docs/en/master/performance_profiling.html>

.. toctree::
   :glob:
   :maxdepth: 1
   :caption: Advanced Features
   :hidden:

   second_order_optimizer
   graph_kernel_fusion

.. toctree::
   :glob:
   :maxdepth: 1
   :caption: Application
   :hidden:

   cv
   nlp
   hpc
   use_on_the_cloud

.. raw:: html

    <div class="container">
			<div class="row">
				<div class="col-md-12">
					<div>


						<div class="doc-condition">
							<span class="doc-filter">Filter</span>
							<button class="doc-delete doc-btn" id="all">Clear All Conditions</button>
						</div>

						<div class="doc-label-content">
							<div class="doc-label-choice">
								<div class="row">
								<div class="col-sm-2">
									<div class="doc-hardware">Hardware</div>
								</div>
								<div class="col-sm-10 col-sm-pull-1">
									<button class="doc-filter-btn doc-btn" id="hardware-ascend">Ascend</button>
									<button class="doc-filter-btn doc-btn" id="hardware-gpu">GPU</button>
									<button class="doc-filter-btn doc-btn" id="hardware-cpu">CPU</button>
									<button class="doc-filter-btn doc-btn" id="hardware-device">Device</button>
								</div>
							</div>
							</div>

							<div class="doc-label-choice">
								<div class="row">
									<div class="col-sm-2 ">
										<div class="doc-stage">Classification</div>
									</div>
									<div class="col-sm-10 col-sm-pull-1 doc-stage-detail">
										<button class="doc-filter-btn doc-btn" id="stage-Beginner">Beginner</button>
                              <button class="doc-filter-btn doc-btn" id="stage-Whole-Process">Whole Process</button>
										<button class="doc-filter-btn doc-btn" id="stage-Data-Preparation">Data Preparation</button>
										<button class="doc-filter-btn doc-btn" id="stage-Model-Development">Model Development</button>
                              <button class="doc-filter-btn doc-btn" id="stage-Model-Running">Model Running</button>
                              <button class="doc-filter-btn doc-btn" id="stage-Model-Optimization">Model Optimization</button>
										<button class="doc-filter-btn doc-btn" id="stage-Model-Export">Model Export</button>
                              <button class="doc-filter-btn doc-btn" id="stage-Inference">stage Inference</button>
										<button class="doc-filter-btn doc-btn" id="stage-Model-Evaluation">Model Evaluation</button>
										<button class="doc-filter-btn doc-btn" id="stage-Model-Loading">Model Loading</button>
										<button class="doc-filter-btn doc-btn" id="stage-Distributed-Parallel">Distributed Parallel</button>
										<button class="doc-filter-btn doc-btn" id="stage-Function-Extension">Function Extension</button>
										<button class="doc-filter-btn doc-btn" id="stage-Design">Design</button>
									</div>

								</div>
							</div>
							<div class="doc-label-choice">
								<div class="row">
									<div class="col-sm-2 ">
										<div class="doc-experience">Experience</div>
									</div>
									<div class="col-sm-10 col-sm-pull-1 doc-stage-detail">
										<button class="doc-filter-btn doc-btn" id="experience-online">Online</button>
										<button class="doc-filter-btn doc-btn" id="experience-local">Local</button>
									</div>

								</div>

							</div>

						</div>

                  <font size="2">Note: Clicking on the title with "↗" will leave the Programming Guide page.</font>

						<hr>

						<div class="doc-footer">
							<nav aria-label="Page navigation">
								<ul class="pagination" id="pageNav">

								</ul>
							</nav>
						</div>

                  <div class="doc-article-list">
						<div class="doc-article-item all hardware-ascend hardware-gpu hardware-cpu stage-Beginner experience-local experience-online hidden">
                     <a href="https://www.mindspore.cn/docs/programming_guide/en/master/api_structure.html" class="article-link">
                        <div>
                              <div class="doc-article-head">
                                 <span class="doc-head-content">MindSpore API Overview</span>
                              </div>
                              <div class="doc-article-desc">
                              MindSpore is a deep learning framework in all scenarios, aiming to achieve easy development, efficient execution, and all-scenario coverage.
                              </div>
                        </div>
                     </a>
                  </div>
                  <div class="doc-article-item all hardware-gpu stage-Model-Optimization experience-local hidden">
                     <a href="https://www.mindspore.cn/docs/programming_guide/en/master/apply_gradient_accumulation.html" class="article-link">
                        <div>
                              <div class="doc-article-head">
                                 <span class="doc-head-content">Gradient Accumulation Algorithm</span>
                              </div>
                              <div class="doc-article-desc">
                              This tutorial describes the gradient accumulation training methods to solve the problem that some large-scale networks cannot train large batch_size due to insufficient memory.
                              </div>
                        </div>
                     </a>
                  </div>
                  <div class="doc-article-item all hardware-ascend hardware-gpu hardware-cpu stage-Model-Development hidden">
                     <a href="https://www.mindspore.cn/docs/programming_guide/en/master/model_use_guide.html" class="article-link">
                        <div>
                              <div class="doc-article-head">
                                 <span class="doc-head-content">Basic Use of Model</span>
                              </div>
                              <div class="doc-article-desc">
                              This document describes how to use high-level API models for model training and evaluation.
                              </div>
                        </div>
                     </a>
                  </div>
                  <div class="doc-article-item all hardware-ascend hardware-gpu stage-Distributed-Parallel stage-Whole-Process experience-local hidden">
                     <a href="https://www.mindspore.cn/docs/programming_guide/en/master/apply_host_device_training.html" class="article-link">
                        <div>
                              <div class="doc-article-head">
                                 <span class="doc-head-content">Host&Device Heterogeneous</span>
                              </div>
                              <div class="doc-article-desc">
                              In deep learning, to efficiently train a huge model, one solution is to employ homogeneous accelerators (*e.g.*, Ascend 910 AI Accelerator and GPU) for distributed training.
                              </div>
                        </div>
                     </a>
                  </div>
                  <div class="doc-article-item all hardware-ascend hardware-gpu hardware-device stage-Distributed-Parallel stage-Whole-Process experience-local hidden">
                     <a href="https://www.mindspore.cn/docs/programming_guide/en/master/apply_parameter_server_training.html" class="article-link">
                        <div>
                              <div class="doc-article-head">
                                 <span class="doc-head-content">Parameter Server Mode</span>
                              </div>
                              <div class="doc-article-desc">
                              A parameter server is a widely used architecture in distributed training. Compared with the synchronous AllReduce training method, a parameter server has better flexibility, scalability, and node failover capabilities.
                              </div>
                        </div>
                     </a>
                  </div>
                  <div class="doc-article-item all hardware-ascend stage-Distributed-Parallel stage-Whole-Process experience-local hidden">
                     <a href="https://www.mindspore.cn/docs/programming_guide/en/master/apply_pipeline_parallel.html" class="article-link">
                        <div>
                              <div class="doc-article-head">
                                 <span class="doc-head-content">Pipeline Parallelism</span>
                              </div>
                              <div class="doc-article-desc">
                              MindSpore can automatically convert a standalone model to the pipeline parallel mode based on user configurations.
                              </div>
                        </div>
                     </a>
                  </div>
                  <div class="doc-article-item all hardware-ascend hardware-device experience-local hidden">
                     <a href="https://www.mindspore.cn/docs/programming_guide/en/master/apply_post_training_quantization.html" class="article-link">
                        <div>
                              <div class="doc-article-head">
                                 <span class="doc-head-content">Applying Post Training Quantization</span>
                              </div>
                              <div class="doc-article-desc">
                              Post training quantization refers to perform weights quantization or full quantization on a pre-trained model. It can reduce model size while also speed up the inference.
                              </div>
                        </div>
                     </a>
                  </div>
                  <div class="doc-article-item all experience-local hidden">
                     <a href="https://www.mindspore.cn/docs/programming_guide/en/master/apply_recompute.html" class="article-link">
                        <div>
                              <div class="doc-article-head">
                                 <span class="doc-head-content">Recomputation</span>
                              </div>
                              <div class="doc-article-desc">
                              The automatic differential of MindSpore is in reverse-mode, which derives the backward pass according to the forward pass.
                              </div>
                        </div>
                     </a>
                  </div>
                  <div class="doc-article-item all hardware-ascend hardware-gpu hardware-cpu stage-Beginner experience-local hidden">
                     <a href="https://www.mindspore.cn/docs/programming_guide/en/master/architecture.html" class="article-link">
                        <div>
                              <div class="doc-article-head">
                                 <span class="doc-head-content">Overall Architecture</span>
                              </div>
                              <div class="doc-article-desc">
                              MindSpore is a deep learning framework in all scenarios, aiming to achieve easy development, efficient execution, and all-scenario coverage.
                              </div>
                        </div>
                     </a>
                  </div>
                  <div class="doc-article-item all hardware-ascend hardware-gpu hardware-cpu stage-Data-Preparation experience-local hidden">
                     <a href="https://www.mindspore.cn/docs/programming_guide/en/master/augmentation.html" class="article-link">
                        <div>
                              <div class="doc-article-head">
                                 <span class="doc-head-content">Image Data Processing and Enhancement</span>
                              </div>
                              <div class="doc-article-desc">
                              In a computer vision task, if the data volume is small or the scenario of the samples are simple, the training effect will be affected. In this case, you may preprocess images by performing data augmentation, so as to improve generalization of the model.
                              </div>
                        </div>
                     </a>
                  </div>
                  <div class="doc-article-item all hardware-ascend hardware-gpu hardware-cpu stage-Data-Preparation experience-local hidden">
                     <a href="https://www.mindspore.cn/docs/programming_guide/en/master/auto_augmentation.html" class="article-link">
                        <div>
                              <div class="doc-article-head">
                                 <span class="doc-head-content">Auto Augmentation</span>
                              </div>
                              <div class="doc-article-desc">
                              MindSpore not only allows you to customize data augmentation, but also provides an auto augmentation method to automatically perform data augmentation on images based on specific policies.
                              </div>
                        </div>
                     </a>
                  </div>
                  <div class="doc-article-item all hardware-ascend hardware-gpu stage-Distributed-Parallel experience-local hidden">
                     <a href="https://www.mindspore.cn/docs/programming_guide/en/master/auto_parallel.html" class="article-link">
                        <div>
                              <div class="doc-article-head">
                                 <span class="doc-head-content">Parallel Distributed Training Interfaces</span>
                              </div>
                              <div class="doc-article-desc">
                              Parallel distributed training can reduce the requirements on hardware such as memory and computing performance and is an important optimization method for training.
                              </div>
                        </div>
                     </a>
                  </div>
                  <div class="doc-article-item all hardware-ascend hardware-gpu hardware-cpu stage-Model-Development experience-local hidden">
                     <a href="https://www.mindspore.cn/docs/programming_guide/en/master/build_net.html" class="article-link">
                        <div>
                              <div class="doc-article-head">
                                 <span class="doc-head-content">Constructing Single Operator Network and Multi-layer Network</span>
                              </div>
                              <div class="doc-article-desc">
                              The Cell class of MindSpore is the base class for constructing all networks, which is also the base unit of networks.
                              </div>
                        </div>
                     </a>
                  </div>
                  <div class="doc-article-item all hardware-ascend hardware-gpu hardware-cpu stage-Data-Preparation experience-local hidden">
                     <a href="https://www.mindspore.cn/docs/programming_guide/en/master/cache.html" class="article-link">
                        <div>
                              <div class="doc-article-head">
                                 <span class="doc-head-content">Single-Node Tensor Cache</span>
                              </div>
                              <div class="doc-article-desc">
                              If you need to repeatedly access remote datasets or load datasets from disks, you can use the single-node cache operator to cache datasets in the local memory to accelerate dataset loading.
                              </div>
                        </div>
                     </a>
                  </div>
                  <div class="doc-article-item all hardware-ascend hardware-gpu hardware-cpu stage-Model-Development experience-local hidden">
                     <a href="https://www.mindspore.cn/docs/programming_guide/en/master/callback.html" class="article-link">
                        <div>
                              <div class="doc-article-head">
                                 <span class="doc-head-content">Callback Mechanism</span>
                              </div>
                              <div class="doc-article-desc">
                              The callback function is implemented as a class in MindSpore. The callback mechanism is similar to a monitoring mode, which helps you observe parameter changes and network internal status during network training.
                              </div>
                        </div>
                     </a>
                  </div>
                  <div class="doc-article-item all hardware-ascend hardware-gpu hardware-cpu stage-Beginner experience-local hidden">
                     <a href="https://www.mindspore.cn/docs/programming_guide/en/master/cell.html" class="article-link">
                        <div>
                              <div class="doc-article-head">
                                 <span class="doc-head-content">Cell</span>
                              </div>
                              <div class="doc-article-desc">
                              The <em>Cell</em> class of MindSpore is the base class for building all networks and the basic unit of a network.
                              </div>
                        </div>
                     </a>
                  </div>
                  <div class="doc-article-item all hardware-ascend hardware-gpu hardware-cpu stage-Model-Development experience-local hidden">
                     <a href="https://www.mindspore.cn/docs/programming_guide/en/master/constexpr.html" class="article-link">
                        <div>
                              <div class="doc-article-head">
                                 <span class="doc-head-content">Construct Constants In the Network</span>
                              </div>
                              <div class="doc-article-desc">
                              A @constexpr python decorator is provided in <em>mindspore.ops.constexpr</em> , which can be used to decorate a function.
                              </div>
                        </div>
                     </a>
                  </div>
                  <div class="doc-article-item all hardware-ascend hardware-gpu hardware-cpu stage-Model-Running experience-local hidden">
                     <a href="https://www.mindspore.cn/docs/programming_guide/en/master/context.html" class="article-link">
                        <div>
                              <div class="doc-article-head">
                                 <span class="doc-head-content">Configuring Running Information</span>
                              </div>
                              <div class="doc-article-desc">
                              Before initializing the network, configure the context parameter to control the policy executed by the program.
                              </div>
                        </div>
                     </a>
                  </div>
                  <div class="doc-article-item all hardware-ascend hardware-gpu hardware-cpu stage-Model-Development experience-local hidden">
                     <a href="https://www.mindspore.cn/docs/programming_guide/en/master/control_flow.html" class="article-link">
                        <div>
                              <div class="doc-article-head">
                                 <span class="doc-head-content">Using the Process Control Statement</span>
                              </div>
                              <div class="doc-article-desc">
                              The MindSpore process control statement is similar to the native Python syntax, especially in <em>PYNATIVE_MODE</em> mode.
                              </div>
                        </div>
                     </a>
                  </div>
                  <div class="doc-article-item all hardware-ascend hardware-gpu hardware-cpu stage-Data-Preparation experience-local experience-online hidden">
                     <a href="https://www.mindspore.cn/docs/programming_guide/en/master/convert_dataset.html" class="article-link">
                        <div>
                              <div class="doc-article-head">
                                 <span class="doc-head-content">Converting Dataset to MindRecord</span>
                              </div>
                              <div class="doc-article-desc">
                              Users can convert non-standard datasets and common datasets into the MindSpore data format, MindRecord, so that they can be easily loaded to MindSpore for training.
                              </div>
                        </div>
                     </a>
                  </div>
                  <div class="doc-article-item all hardware-ascend stage-Model-Optimization experience-local hidden">
                     <a href="https://www.mindspore.cn/docs/programming_guide/en/master/custom_debugging_info.html" class="article-link">
                        <div>
                              <div class="doc-article-head">
                                 <span class="doc-head-content">Custom Debugging Information</span>
                              </div>
                              <div class="doc-article-desc">
                              This section describes how to use the customized capabilities provided by MindSpore, such as <em>callback</em>, <em>metrics</em>, <em>Print</em> operators and log printing, to help you quickly debug the training network.
                              </div>
                        </div>
                     </a>
                  </div>
                  <div class="doc-article-item all hardware-ascend stage-Model-Development experience-local hidden">
                     <a href="https://www.mindspore.cn/docs/programming_guide/en/master/custom_operator_ascend.html" class="article-link">
                        <div>
                              <div class="doc-article-head">
                                 <span class="doc-head-content">Custom Operators (Ascend)</span>
                              </div>
                              <div class="doc-article-desc">
                              When built-in operators cannot meet requirements during network development, you can call the Python API of MindSpore to quickly extend custom operators of the Ascend AI processor.
                              </div>
                        </div>
                     </a>
                  </div>
                  <div class="doc-article-item all hardware-cpu stage-Model-Development experience-local hidden">
                     <a href="https://www.mindspore.cn/docs/programming_guide/en/master/custom_operator_cpu.html" class="article-link">
                        <div>
                              <div class="doc-article-head">
                                 <span class="doc-head-content">Custom Operators (CPU)</span>
                              </div>
                              <div class="doc-article-desc">
                              When the built-in operators are not enough for developing the network, you can extend your custom CPU operators fast and conveniently using MindSpore’s Python API and C++ API.
                              </div>
                        </div>
                     </a>
                  </div>
                  <div class="doc-article-item all hardware-gpu stage-Model-Development experience-local hidden">
                     <a href="https://www.mindspore.cn/docs/programming_guide/en/master/custom_operator_gpu.html" class="article-link">
                        <div>
                              <div class="doc-article-head">
                                 <span class="doc-head-content">Custom Operators (GPU)</span>
                              </div>
                              <div class="doc-article-desc">
                              Operator is the basic element of constructing neural network. When built-in operators cannot meet requirements during network development, you can utilize MindSpore to quickly extend custom operators of the Graphics Processing Unit.
                              </div>
                        </div>
                     </a>
                  </div>
                  <div class="doc-article-item all hardware-ascend hardware-gpu hardware-cpu experience-local hidden">
                     <a href="https://www.mindspore.cn/docs/programming_guide/en/master/cv_mobilenetv2_fine_tune.html" class="article-link">
                        <div>
                              <div class="doc-article-head">
                                 <span class="doc-head-content">Using MobileNetV2 to Implement Fine-Tuning</span>
                              </div>
                              <div class="doc-article-desc">
                              In a computer vision task, training a network from scratch is time-consuming and requires a large amount of computing power. Therefore, most tasks perform fine-tuning on pre-trained models.
                              </div>
                        </div>
                     </a>
                  </div>
                  <div class="doc-article-item all hardware-ascend hardware-gpu stage-Whole-Process experience-local hidden">
                     <a href="https://www.mindspore.cn/docs/programming_guide/en/master/cv_resnet50.html" class="article-link">
                        <div>
                              <div class="doc-article-head">
                                 <span class="doc-head-content">Image Classification Using ResNet-50 Network</span>
                              </div>
                              <div class="doc-article-desc">
                              Deep neural network can extract image features layer by layer and retain local invariance. It is widely used in visual tasks such as classification, detection, segmentation, retrieval, recognition, promotion, and reconstruction.
                              </div>
                        </div>
                     </a>
                  </div>
                  <div class="doc-article-item all hardware-ascend hardware-gpu stage-Function-Extension stage-Whole-Process experience-local hidden">
                     <a href="https://www.mindspore.cn/docs/programming_guide/en/master/cv_resnet50_second_order_optimizer.html" class="article-link">
                        <div>
                              <div class="doc-article-head">
                                 <span class="doc-head-content">ResNet-50 Second-Order Optimization Practice</span>
                              </div>
                              <div class="doc-article-desc">
                              Common optimization algorithms are classified into the first-order and the second-order optimization algorithms.
                              </div>
                        </div>
                     </a>
                  </div>
                  <div class="doc-article-item all hardware-ascend hardware-gpu hardware-cpu stage-Data-Preparation experience-local hidden">
                     <a href="https://www.mindspore.cn/docs/programming_guide/en/master/dataset_conversion.html" class="article-link">
                        <div>
                              <div class="doc-article-head">
                                 <span class="doc-head-content">MindSpore Data Format Conversion</span>
                              </div>
                              <div class="doc-article-desc">
                              You can convert non-standard datasets and common datasets into the MindSpore data format (that is, MindRecord) to easily load the datasets to MindSpore for training.
                              </div>
                        </div>
                     </a>
                  </div>
                  <div class="doc-article-item all hardware-ascend hardware-gpu hardware-cpu stage-Beginner experience-local hidden">
                     <a href="https://www.mindspore.cn/docs/programming_guide/en/master/dataset_introduction.html" class="article-link">
                        <div>
                              <div class="doc-article-head">
                                 <span class="doc-head-content">Dataset</span>
                              </div>
                              <div class="doc-article-desc">
                              Data is the foundation of deep learning, and high-quality data input will play a positive role in the entire deep neural network.
                              </div>
                        </div>
                     </a>
                  </div>
                  <div class="doc-article-item all hardware-ascend hardware-gpu hardware-cpu stage-Data-Preparation experience-local hidden">
                     <a href="https://www.mindspore.cn/docs/programming_guide/en/master/dataset_loading.html" class="article-link">
                        <div>
                              <div class="doc-article-head">
                                 <span class="doc-head-content">Loading Dataset Overview</span>
                              </div>
                              <div class="doc-article-desc">
                              MindSpore can load common image datasets. You can directly use the classes in <em>mindspore.dataset</em> to load datasets.
                              </div>
                        </div>
                     </a>
                  </div>
                  <div class="doc-article-item all hardware-ascend hardware-gpu hardware-cpu stage-Data-Preparation experience-local hidden">
                     <a href="https://www.mindspore.cn/docs/programming_guide/en/master/dataset_usage.html" class="article-link">
                        <div>
                              <div class="doc-article-head">
                                 <span class="doc-head-content">Data Iteration</span>
                              </div>
                              <div class="doc-article-desc">
                              Original dataset is read into the memory through dataset loading interface, and then data is transformed through data enhancement operation.
                              </div>
                        </div>
                     </a>
                  </div>
                  <div class="doc-article-item all hardware-ascend hardware-gpu hardware-cpu stage-Model-Running experience-local hidden">
                     <a href="https://www.mindspore.cn/docs/programming_guide/en/master/debug_in_pynative_mode.html" class="article-link">
                        <div>
                              <div class="doc-article-head">
                                 <span class="doc-head-content">Debugging in PyNative Mode</span>
                              </div>
                              <div class="doc-article-desc">
                              MindSpore supports the following running modes which are optimized for debugging or running.
                              </div>
                        </div>
                     </a>
                  </div>
                  <div class="doc-article-item all hardware-ascend experience-local hidden">
                     <a href="https://www.mindspore.cn/docs/programming_guide/en/master/distributed_inference.html" class="article-link">
                        <div>
                              <div class="doc-article-head">
                                 <span class="doc-head-content">Distributed Inference</span>
                              </div>
                              <div class="doc-article-desc">
                              Distributed inference means use multiple devices for prediction. If data parallel or integrated save is used in training, the method of distributed inference is same with the above description. It is noted that each device should load one same checkpoint file.
                              </div>
                        </div>
                     </a>
                  </div>
                  <div class="doc-article-item all hardware-ascend stage-Distributed-Parallel stage-Whole-Process experience-local hidden">
                     <a href="https://www.mindspore.cn/docs/programming_guide/en/master/distributed_training_ascend.html" class="article-link">
                        <div>
                              <div class="doc-article-head">
                                 <span class="doc-head-content">Parallel Distributed Training Example (Ascend)</span>
                              </div>
                              <div class="doc-article-desc">
                              This tutorial describes how to train the ResNet-50 network in data parallel and automatic parallel modes on MindSpore based on the Ascend 910 AI processor.
                              </div>
                        </div>
                     </a>
                  </div>
                  <div class="doc-article-item all hardware-gpu stage-Distributed-Parallel stage-Whole-Process experience-local hidden">
                     <a href="https://www.mindspore.cn/docs/programming_guide/en/master/distributed_training_gpu.html" class="article-link">
                        <div>
                              <div class="doc-article-head">
                                 <span class="doc-head-content">Distributed Parallel Training Example (GPU)</span>
                              </div>
                              <div class="doc-article-desc">
                              This tutorial describes how to train the ResNet-50 network using MindSpore data parallelism and automatic parallelism on the GPU hardware platform.
                              </div>
                        </div>
                     </a>
                  </div>
                  <div class="doc-article-item all hardware-ascend hardware-gpu hardware-cpu stage-Beginner experience-local hidden">
                     <a href="https://www.mindspore.cn/docs/programming_guide/en/master/dtype.html" class="article-link">
                        <div>
                              <div class="doc-article-head">
                                 <span class="doc-head-content">DataType</span>
                              </div>
                              <div class="doc-article-desc">
                              MindSpore tensors support different data types, which correspond to the data types of NumPy.
                              </div>
                        </div>
                     </a>
                  </div>
                  <div class="doc-article-item all hardware-ascend hardware-gpu hardware-cpu stage-Model-Optimization experience-local hidden">
                     <a href="https://www.mindspore.cn/docs/programming_guide/en/master/dump_in_graph_mode.html" class="article-link">
                        <div>
                              <div class="doc-article-head">
                                 <span class="doc-head-content">Using Dump in the Graph Mode</span>
                              </div>
                              <div class="doc-article-desc">
                              The input and output of the operator can be saved for debugging through the data dump when the training result deviates from the expectation.
                              </div>
                        </div>
                     </a>
                  </div>
                  <div class="doc-article-item all hardware-ascend hardware-gpu hardware-cpu stage-Data-Preparation experience-local hidden">
                     <a href="https://www.mindspore.cn/docs/programming_guide/en/master/eager.html" class="article-link">
                        <div>
                              <div class="doc-article-head">
                                 <span class="doc-head-content">Lightweight Data Processing</span>
                              </div>
                              <div class="doc-article-desc">
                              When resources permit, in order to purse higher performance, data pipeline mode is generally used for data augmentation.
                              </div>
                        </div>
                     </a>
                  </div>
                  <div class="doc-article-item all hardware-ascend hardware-gpu hardware-cpu stage-Data-Preparation experience-local hidden">
                     <a href="https://www.mindspore.cn/docs/programming_guide/en/master/enable_auto_augmentation.html" class="article-link">
                        <div>
                              <div class="doc-article-head">
                                 <span class="doc-head-content">Application of Auto Augmentation</span>
                              </div>
                              <div class="doc-article-desc">
                              Auto Augmentation finds a suitable image augmentation scheme for a specific dataset by searching through a series of image augmentation sub-policies. The <em>c_transforms</em> module of MindSpore provides various C++ operators that are used in Auto Augmentation.
                              </div>
                        </div>
                     </a>
                  </div>
                  <div class="doc-article-item all hardware-ascend stage-Model-Optimization experience-local hidden">
                     <a href="https://www.mindspore.cn/docs/programming_guide/en/master/enable_auto_tune.html" class="article-link">
                        <div>
                              <div class="doc-article-head">
                                 <span class="doc-head-content">AutoTune</span>
                              </div>
                              <div class="doc-article-desc">
                              AutoTune is a tool that uses hardware resources and automatically tune the performance of TBE operators. This document mainly introduces how to use the AutoTune tool to Online tune.
                              </div>
                        </div>
                     </a>
                  </div>
                  <div class="doc-article-item all hardware-ascend hardware-gpu stage-Data-Preparation experience-local hidden">
                     <a href="https://www.mindspore.cn/docs/programming_guide/en/master/enable_dataset_autotune.html" class="article-link">
                        <div>
                              <div class="doc-article-head">
                                 <span class="doc-head-content">Dataset AutoTune for Dataset Pipeline</span>
                              </div>
                              <div class="doc-article-desc">
                              MindSpore provides a tool named Dataset AutoTune for optimizing dataset.
                              </div>
                        </div>
                     </a>
                  </div>
                  <div class="doc-article-item all hardware-ascend hardware-gpu hardware-cpu stage-Data-Preparation experience-local hidden">
                     <a href="https://www.mindspore.cn/docs/programming_guide/en/master/enable_dataset_offload.html" class="article-link">
                        <div>
                              <div class="doc-article-head">
                                 <span class="doc-head-content">Enabling Offload for Dataset</span>
                              </div>
                              <div class="doc-article-desc">
                              The offload feature may speed up data processing by moving dataset operations from dataset pipeline to computation graph, allowing these operations to be run by the hardware accelerator.
                              </div>
                        </div>
                     </a>
                  </div>
                  <div class="doc-article-item all hardware-ascend hardware-gpu hardware-cpu stage-Data-Preparation experience-local hidden">
                     <a href="https://www.mindspore.cn/docs/programming_guide/en/master/enable_cache.html" class="article-link">
                        <div>
                              <div class="doc-article-head">
                                 <span class="doc-head-content">Application of Single-Node Tensor Cache</span>
                              </div>
                              <div class="doc-article-desc">
                              If you need to repeatedly access remote datasets or read datasets from disks, you can use the single-node cache operator to cache datasets in the local memory to accelerate dataset reading.
                              </div>
                        </div>
                     </a>
                  </div>
                  <div class="doc-article-item all hardware-ascend hardware-gpu stage-Model-Optimization experience-local hidden">
                     <a href="https://www.mindspore.cn/docs/programming_guide/en/master/enable_graph_kernel_fusion.html" class="article-link">
                        <div>
                              <div class="doc-article-head">
                                 <span class="doc-head-content">Enabling Graph Kernel Fusion</span>
                              </div>
                              <div class="doc-article-desc">
                              The graph kernel fusion is used to optimize network performance by cooperating with MindSpore AKG which is a operator compiler based on polyhedral technology.
                              </div>
                        </div>
                     </a>
                  </div>
                  <div class="doc-article-item all hardware-ascend hardware-gpu stage-Model-Optimization experience-local hidden">
                     <a href="https://www.mindspore.cn/docs/programming_guide/en/master/enable_mixed_precision.html" class="article-link">
                        <div>
                              <div class="doc-article-head">
                                 <span class="doc-head-content">Enabling Mixed Precision</span>
                              </div>
                              <div class="doc-article-desc">
                              The mixed precision training method accelerates the deep learning neural network training process by using both the single-precision and half-precision data formats, and maintains the network precision achieved by the single-precision training at the same time.
                              </div>
                        </div>
                     </a>
                  </div>
                  <div class="doc-article-item all hardware-ascend hardware-gpu hardware-cpu stage-Model-Running experience-local hidden">
                     <a href="https://www.mindspore.cn/docs/programming_guide/en/master/evaluate_the_model_during_training.html" class="article-link">
                        <div>
                              <div class="doc-article-head">
                                 <span class="doc-head-content">Evaluating the Model during Training</span>
                              </div>
                              <div class="doc-article-desc">
                              This section uses the method of <em>Evaluating the Model during Training</em> and takes the LeNet network as an example.
                              </div>
                        </div>
                     </a>
                  </div>
                  <div class="doc-article-item all hardware-ascend hardware-gpu hardware-cpu stage-Model-Development experience-local hidden">
                     <a href="https://www.mindspore.cn/docs/programming_guide/en/master/grad_operation.html" class="article-link">
                        <div>
                              <div class="doc-article-head">
                                 <span class="doc-head-content">Gradient Operation</span>
                              </div>
                              <div class="doc-article-desc">
                              Grad Operation is used to generate the gradient of the input function.
                              </div>
                        </div>
                     </a>
                  </div>
                  <div class="doc-article-item all hardware-gpu stage-Whole-Process experience-local hidden">
                     <a href="https://www.mindspore.cn/docs/programming_guide/en/master/hpc_gomo.html" class="article-link">
                        <div>
                              <div class="doc-article-head">
                                 <span class="doc-head-content">Implementing Regional Ocean Model GOMO</span>
                              </div>
                              <div class="doc-article-desc">
                              Generalized Operator Modelling of the Ocean (GOMO) is a 3D regional ocean model based on OpenArray.
                              </div>
                        </div>
                     </a>
                  </div>
                  <div class="doc-article-item all hardware-ascend hardware-gpu hardware-cpu stage-Model-Development experience-local hidden">
                     <a href="https://www.mindspore.cn/docs/programming_guide/en/master/hypermap.html" class="article-link">
                        <div>
                              <div class="doc-article-head">
                                 <span class="doc-head-content">Operation Overloading</span>
                              </div>
                              <div class="doc-article-desc">
                              <em>mindspore.ops.composite</em> provide some operator combinations related to graph transformation such as <em>MultitypeFuncGraph</em> and <em>HyperMap</em>.
                              </div>
                        </div>
                     </a>
                  </div>
                  <div class="doc-article-item all hardware-ascend stage-Model-Optimization experience-local hidden">
                     <a href="https://www.mindspore.cn/docs/programming_guide/en/master/incremental_operator_build.html" class="article-link">
                        <div>
                              <div class="doc-article-head">
                                 <span class="doc-head-content">Incremental Operator Build</span>
                              </div>
                              <div class="doc-article-desc">
                              When a network model is executed, MindSpore builds the used operators. To improve the performance of secondary model execution, an incremental operator build mechanism is provided.
                              </div>
                        </div>
                     </a>
                  </div>
                  <div class="doc-article-item all hardware-ascend hardware-gpu hardware-cpu stage-Model-Development experience-local hidden">
                     <a href="https://www.mindspore.cn/docs/programming_guide/en/master/indefinite_parameter.html" class="article-link">
                        <div>
                              <div class="doc-article-head">
                                 <span class="doc-head-content">Parameter Passing</span>
                              </div>
                              <div class="doc-article-desc">
                              This article describes the use of variable parameters in network construction, indicates that a variable number of parameters can be used to construct a network.
                              </div>
                        </div>
                     </a>
                  </div>
                  <div class="doc-article-item all hardware-ascend hardware-gpu hardware-cpu stage-Model-Development experience-local hidden">
                     <a href="https://www.mindspore.cn/docs/programming_guide/en/master/initializer.html" class="article-link">
                        <div>
                              <div class="doc-article-head">
                                 <span class="doc-head-content">Initializer</span>
                              </div>
                              <div class="doc-article-desc">
                              The Initializer class is the basic data structure used for initialization in MindSpore.
                              </div>
                        </div>
                     </a>
                  </div>
                  <div class="doc-article-item all hardware-ascend hardware-gpu hardware-cpu stage-Data-Preparation experience-local experience-online hidden">
                     <a href="https://www.mindspore.cn/docs/programming_guide/en/master/load_dataset_image.html" class="article-link">
                        <div>
                              <div class="doc-article-head">
                                 <span class="doc-head-content">Loading Image Dataset</span>
                              </div>
                              <div class="doc-article-desc">
                              In computer vision training tasks, it is often difficult to read the entire dataset directly into memory due to memory capacity. The <em>mindspore.dataset</em> module provided by MindSpore enables users to customize their data fetching strategy from disk.
                              </div>
                        </div>
                     </a>
                  </div>
                  <div class="doc-article-item all hardware-ascend hardware-gpu hardware-cpu stage-Data-Preparation experience-local hidden">
                     <a href="https://www.mindspore.cn/docs/programming_guide/en/master/load_dataset_networks.html" class="article-link">
                        <div>
                              <div class="doc-article-head">
                                 <span class="doc-head-content">Dataset in Model Zoo Networks</span>
                              </div>
                              <div class="doc-article-desc">
                              This article lists the dataset in model zoo networks.
                              </div>
                        </div>
                     </a>
                  </div>
                  <div class="doc-article-item all hardware-ascend hardware-gpu hardware-cpu stage-Data-Preparation experience-local experience-online hidden">
                     <a href="https://www.mindspore.cn/docs/programming_guide/en/master/load_dataset_text.html" class="article-link">
                        <div>
                              <div class="doc-article-head">
                                 <span class="doc-head-content">Loading Text Dataset</span>
                              </div>
                              <div class="doc-article-desc">
                              The <em>mindspore.dataset</em> module provided by MindSpore enables users to customize their data fetching strategy from disk.
                              </div>
                        </div>
                     </a>
                  </div>
                  <div class="doc-article-item all hardware-ascend hardware-gpu hardware-cpu stage-Model-Loading experience-local hidden">
                     <a href="https://www.mindspore.cn/docs/programming_guide/en/master/load_model_for_inference_and_transfer.html" class="article-link">
                        <div>
                              <div class="doc-article-head">
                                 <span class="doc-head-content">Loading a Model for Inference and Transfer Learning</span>
                              </div>
                              <div class="doc-article-desc">
                              CheckPoints which are saved locally during model training, they are used for inference and transfer training.
                              </div>
                        </div>
                     </a>
                  </div>
                  <div class="doc-article-item all hardware-ascend hardware-gpu hardware-cpu stage-Model-Development experience-local hidden">
                     <a href="https://www.mindspore.cn/docs/programming_guide/en/master/loss.html" class="article-link">
                        <div>
                              <div class="doc-article-head">
                                 <span class="doc-head-content">Loss Function</span>
                              </div>
                              <div class="doc-article-desc">
                              Loss function, also known as object function, is used for measuring the difference between predicted and true value.
                              </div>
                        </div>
                     </a>
                  </div>
                  <div class="doc-article-item all hardware-ascend hardware-gpu hardware-cpu experience-local hidden">
                     <a href="https://www.mindspore.cn/docs/programming_guide/en/master/multi_platform_inference.html" class="article-link">
                        <div>
                              <div class="doc-article-head">
                                 <span class="doc-head-content">Inference Model Overview</span>
                              </div>
                              <div class="doc-article-desc">
                              MindSpore can execute inference tasks on different hardware platforms based on trained models.
                              </div>
                        </div>
                     </a>
                  </div>
                  <div class="doc-article-item all hardware-ascend experience-local hidden">
                     <a href="https://www.mindspore.cn/docs/programming_guide/en/master/multi_platform_inference_ascend_310_air.html" class="article-link">
                        <div>
                              <div class="doc-article-head">
                                 <span class="doc-head-content">Inference on the Ascend 310 AI Processor</span>
                              </div>
                              <div class="doc-article-desc">
                              Ascend 310 is a highly efficient and integrated AI processor oriented to edge scenarios.
                              </div>
                        </div>
                     </a>
                  </div>
                  <div class="doc-article-item all hardware-ascend experience-local hidden">
                     <a href="https://www.mindspore.cn/docs/programming_guide/en/master/multi_platform_inference_ascend_310_mindir.html" class="article-link">
                        <div>
                              <div class="doc-article-head">
                                 <span class="doc-head-content">Inference Using the MindIR Model on Ascend 310 AI Processors</span>
                              </div>
                              <div class="doc-article-desc">
                              Ascend 310 is a highly efficient and integrated AI processor oriented to edge scenarios. This tutorial describes how to use MindSpore to perform inference on the Ascend 310 based on the MindIR model file.
                              </div>
                        </div>
                     </a>
                  </div>
                  <div class="doc-article-item all hardware-ascend experience-local hidden">
                     <a href="https://www.mindspore.cn/docs/programming_guide/en/master/multi_platform_inference_ascend_910.html" class="article-link">
                        <div>
                              <div class="doc-article-head">
                                 <span class="doc-head-content">Inference on the Ascend 910 AI processor</span>
                              </div>
                              <div class="doc-article-desc">
                              Users can create C++ applications and call MindSpore C++ interface to inference MindIR models.
                              </div>
                        </div>
                     </a>
                  </div>
                  <div class="doc-article-item all hardware-cpu experience-local hidden">
                     <a href="https://www.mindspore.cn/docs/programming_guide/en/master/multi_platform_inference_cpu.html" class="article-link">
                        <div>
                              <div class="doc-article-head">
                                 <span class="doc-head-content">Inference on a CPU</span>
                              </div>
                              <div class="doc-article-desc">
                              This article describes how to perform inference on a CPU.
                              </div>
                        </div>
                     </a>
                  </div>
                  <div class="doc-article-item all hardware-gpu experience-local hidden">
                     <a href="https://www.mindspore.cn/docs/programming_guide/en/master/multi_platform_inference_gpu.html" class="article-link">
                        <div>
                              <div class="doc-article-head">
                                 <span class="doc-head-content">Inference on a GPU</span>
                              </div>
                              <div class="doc-article-desc">
                              This article describes how to perform inference on a GPU.
                              </div>
                        </div>
                     </a>
                  </div>
                  <div class="doc-article-item all hardware-ascend stage-Whole-Process experience-local hidden">
                     <a href="https://www.mindspore.cn/docs/programming_guide/en/master/nlp_bert_poetry.html" class="article-link">
                        <div>
                              <div class="doc-article-head">
                                 <span class="doc-head-content">Using the BERT Network to Implement Intelligent Poem Writing</span>
                              </div>
                              <div class="doc-article-desc">
                              Poetry is an indispensable part of the five-millennium-old Chinese culture. Today, let's see how the science-backed MindSpore trains a model to show its sense of arts!
                              </div>
                        </div>
                     </a>
                  </div>
                  <div class="doc-article-item all hardware-cpu stage-Whole-Process experience-local hidden">
                     <a href="https://www.mindspore.cn/docs/programming_guide/en/master/nlp_sentimentnet.html" class="article-link">
                        <div>
                              <div class="doc-article-head">
                                 <span class="doc-head-content">Realizing Sentiment Classification With SentimentNet</span>
                              </div>
                              <div class="doc-article-desc">
                              Sentiment classification is a subset of text classification in NLP, and is one of the most basic applications of NLP.
                              </div>
                        </div>
                     </a>
                  </div>
                  <div class="doc-article-item all hardware-ascend stage-Whole-Process experience-local hidden">
                     <a href="https://www.mindspore.cn/docs/programming_guide/en/master/nlp_tprr.html" class="article-link">
                        <div>
                              <div class="doc-article-head">
                                 <span class="doc-head-content">Multi-hop Knowledge Reasoning Question-answering Model TPRR</span>
                              </div>
                              <div class="doc-article-desc">
                              TPRR(Thinking Path Re-Ranker) is an open-domain knowledge based multi-hop question-answering model proposed by Huawei, which is used to realize multi-hop knowledge reasoning question-answering.
                              </div>
                        </div>
                     </a>
                  </div>
                  <div class="doc-article-item all hardware-ascend hardware-gpu hardware-cpu stage-Model-Development experience-local experience-online hidden">
                     <a href="https://www.mindspore.cn/docs/programming_guide/en/master/numpy.html" class="article-link">
                        <div>
                              <div class="doc-article-head">
                                 <span class="doc-head-content">NumPy Interfaces in MindSpore</span>
                              </div>
                              <div class="doc-article-desc">
                              MindSpore Numpy package contains a set of Numpy-like interfaces, which allows developers to build models on MindSpore with similar syntax of Numpy.
                              </div>
                        </div>
                     </a>
                  </div>
                  <div class="doc-article-item all hardware-ascend experience-local hidden">
                     <a href="https://www.mindspore.cn/docs/programming_guide/en/master/online_inference.html" class="article-link">
                        <div>
                              <div class="doc-article-head">
                                 <span class="doc-head-content">Online Inference with Checkpoint</span>
                              </div>
                              <div class="doc-article-desc">
                              When the pre-trained models are saved in local, the steps of performing inference on validation dataset are described in this article.
                              </div>
                        </div>
                     </a>
                  </div>
                  <div class="doc-article-item all hardware-ascend hardware-gpu hardware-cpu stage-Model-Running experience-local hidden">
                     <a href="https://www.mindspore.cn/docs/programming_guide/en/master/on_device.html" class="article-link">
                        <div>
                              <div class="doc-article-head">
                                 <span class="doc-head-content">On-Device Execution</span>
                              </div>
                              <div class="doc-article-desc">
                              The backends supported by MindSpore include Ascend, GPU, and CPU. The device in the "On-Device" refers to the Ascend AI processor.
                              </div>
                        </div>
                     </a>
                  </div>
                  <div class="doc-article-item all hardware-ascend hardware-gpu hardware-cpu stage-Beginner experience-local hidden">
                     <a href="https://www.mindspore.cn/docs/programming_guide/en/master/operators_classification.html" class="article-link">
                        <div>
                              <div class="doc-article-head">
                                 <span class="doc-head-content">Operators Classification</span>
                              </div>
                              <div class="doc-article-desc">
                              Operators can be classified into some functional modules: tensor operations, network operations, array operations, image operations, encoding operations, debugging operations, and quantization operations.
                              </div>
                        </div>
                     </a>
                  </div>
                  <div class="doc-article-item all experience-local hidden">
                     <a href="https://www.mindspore.cn/docs/programming_guide/en/master/operators_usage.html" class="article-link">
                        <div>
                              <div class="doc-article-head">
                                 <span class="doc-head-content">Operators Usage</span>
                              </div>
                              <div class="doc-article-desc">
                              APIs related to operators include operations, functional, and composite. Operators related to these three APIs can be directly obtained using ops.
                              </div>
                        </div>
                     </a>
                  </div>
                  <div class="doc-article-item all hardware-ascend hardware-gpu hardware-cpu stage-Model-Development experience-local hidden">
                     <a href="https://www.mindspore.cn/docs/programming_guide/en/master/optim.html" class="article-link">
                        <div>
                              <div class="doc-article-head">
                                 <span class="doc-head-content">Optimization Algorithms</span>
                              </div>
                              <div class="doc-article-desc">
                              <em>mindspore.nn.optim</em> is a module in the MindSpore framework for implementing various optimization algorithms, including common optimizers and learning rates.
                              </div>
                        </div>
                     </a>
                  </div>
                  <div class="doc-article-item all hardware-ascend hardware-gpu hardware-cpu stage-Data-Preparation experience-local experience-online hidden">
                     <a href="https://www.mindspore.cn/docs/programming_guide/en/master/optimize_data_processing.html" class="article-link">
                        <div>
                              <div class="doc-article-head">
                                 <span class="doc-head-content">Optimizing the Data Processing</span>
                              </div>
                              <div class="doc-article-desc">
                              Data is the most important factor of deep learning. High-quality data input is beneficial to the entire deep neural network.
                              </div>
                        </div>
                     </a>
                  </div>
                  <div class="doc-article-item all hardware-ascend hardware-gpu hardware-cpu stage-Model-Development experience-local hidden">
                     <a href="https://www.mindspore.cn/docs/programming_guide/en/master/parameter.html" class="article-link">
                        <div>
                              <div class="doc-article-head">
                                 <span class="doc-head-content">Network Parameters</span>
                              </div>
                              <div class="doc-article-desc">
                              <em>Parameter</em> is a variable tensor, indicating the parameters that need to be updated during network training.
                              </div>
                        </div>
                     </a>
                  </div>
                  <div class="doc-article-item all hardware-ascend hardware-gpu hardware-cpu stage-Beginner experience-local hidden">
                     <a href="https://www.mindspore.cn/docs/programming_guide/en/master/parameter_introduction.html" class="article-link">
                        <div>
                              <div class="doc-article-head">
                                 <span class="doc-head-content">Parameter</span>
                              </div>
                              <div class="doc-article-desc">
                              <em>Parameter</em> is a variable tensor, indicating the parameters that need to be updated during network training. Typically, it consists of <em>weight</em> and <em>bias</em>.
                              </div>
                        </div>
                     </a>
                  </div>
                  <div class="doc-article-item all hardware-ascend hardware-gpu hardware-cpu stage-Data-Preparation experience-local hidden">
                     <a href="https://www.mindspore.cn/docs/programming_guide/en/master/pipeline_common.html" class="article-link">
                        <div>
                              <div class="doc-article-head">
                                 <span class="doc-head-content">General Data Processing</span>
                              </div>
                              <div class="doc-article-desc">
                              Data is the basis of deep learning. Good data input can play a positive role in the entire deep neural network training.
                              </div>
                        </div>
                     </a>
                  </div>
                  <div class="doc-article-item all hardware-ascend hardware-gpu hardware-cpu stage-Model-Optimization experience-local hidden">
                     <a href="https://www.mindspore.cn/docs/programming_guide/en/master/read_ir_files.html" class="article-link">
                        <div>
                              <div class="doc-article-head">
                                 <span class="doc-head-content">Reading IR</span>
                              </div>
                              <div class="doc-article-desc">
                              When a model compiled using MindSpore runs in the graph mode <em>context.set_context(mode=context.GRAPH_MODE)</em> and <em>context.set_context(save_graphs=True)</em> is set in the configuration, some intermediate files will be generated during graph compliation. These intermediate files are called IR files.
                              </div>
                        </div>
                     </a>
                  </div>
                  <div class="doc-article-item all hardware-ascend hardware-gpu stage-Model-Running experience-local hidden">
                     <a href="https://www.mindspore.cn/docs/programming_guide/en/master/run.html" class="article-link">
                        <div>
                              <div class="doc-article-head">
                                 <span class="doc-head-content">Running Mode</span>
                              </div>
                              <div class="doc-article-desc">
                              There are three execution modes: single operator, common function, and network training model.
                              </div>
                        </div>
                     </a>
                  </div>
                  <div class="doc-article-item all hardware-ascend hardware-gpu hardware-cpu stage-Data-Preparation experience-local hidden">
                     <a href="https://www.mindspore.cn/docs/programming_guide/en/master/sampler.html" class="article-link">
                        <div>
                              <div class="doc-article-head">
                                 <span class="doc-head-content">Data Sampling</span>
                              </div>
                              <div class="doc-article-desc">
                              MindSpore provides multiple samplers to help you sample datasets for various purposes to meet training requirements and solve problems such as oversized datasets and uneven distribution of sample categories.
                              </div>
                        </div>
                     </a>
                  </div>
                  <div class="doc-article-item all hardware-ascend hardware-gpu stage-Distributed-Parallel stage-Model-Export stage-Model-Loading experience-local hidden">
                     <a href="https://www.mindspore.cn/docs/programming_guide/en/master/save_load_model_hybrid_parallel.html" class="article-link">
                        <div>
                              <div class="doc-article-head">
                                 <span class="doc-head-content">Saving and Loading Models in Hybrid Parallel Mode</span>
                              </div>
                              <div class="doc-article-desc">
                              In the MindSpore model parallel scenario, each instance process stores only the parameter data on the current node. The parameter data of a model parallel Cell on each node is a slice of the complete parameter data.
                              </div>
                        </div>
                     </a>
                  </div>
                  <div class="doc-article-item all hardware-ascend hardware-gpu hardware-cpu stage-Model-Export experience-local hidden">
                     <a href="https://www.mindspore.cn/docs/programming_guide/en/master/save_model.html" class="article-link">
                        <div>
                              <div class="doc-article-head">
                                 <span class="doc-head-content">Saving Models</span>
                              </div>
                              <div class="doc-article-desc">
                              During model training, you can add CheckPoints to save model parameters for inference and retraining after interruption.
                              </div>
                        </div>
                     </a>
                  </div>
                  <div class="doc-article-item all hardware-ascend hardware-gpu hardware-cpu stage-Model-Evaluation experience-local hidden">
                     <a href="https://www.mindspore.cn/docs/programming_guide/en/master/self_define_metric.html" class="article-link">
                        <div>
                              <div class="doc-article-head">
                                 <span class="doc-head-content">Customize Metrics to Verify Model Evaluation Accuracy</span>
                              </div>
                              <div class="doc-article-desc">
                              After training, it is necessary to use Metrics for model evaluation. Different tasks usually needs different Metrics.
                              </div>
                        </div>
                     </a>
                  </div>
                  <div class="doc-article-item all hardware-ascend hardware-gpu hardware-cpu stage-Beginner experience-local hidden">
                     <a href="https://www.mindspore.cn/docs/programming_guide/en/master/tensor.html" class="article-link">
                        <div>
                              <div class="doc-article-head">
                                 <span class="doc-head-content">Tensor</span>
                              </div>
                              <div class="doc-article-desc">
                              Tensor is a basic data structure in the MindSpore network computing.
                              </div>
                        </div>
                     </a>
                  </div>
                  <div class="doc-article-item all hardware-ascend hardware-gpu hardware-cpu stage-Data-Preparation experience-local hidden">
                     <a href="https://www.mindspore.cn/docs/programming_guide/en/master/tokenizer.html" class="article-link">
                        <div>
                              <div class="doc-article-head">
                                 <span class="doc-head-content">Text Data Processing and Enhancement</span>
                              </div>
                              <div class="doc-article-desc">
                              Tokenization is a process of re-combining continuous character sequences into word sequences according to certain specifications. Reasonable tokenization is helpful for semantic comprehension.
                              </div>
                        </div>
                     </a>
                  </div>
                  <div class="doc-article-item all hardware-ascend hardware-gpu hardware-cpu stage-Model-Development stage-Model-Running stage-Model-Evaluation experience-local hidden">
                     <a href="https://www.mindspore.cn/docs/programming_guide/en/master/use_on_the_cloud.html" class="article-link">
                        <div>
                              <div class="doc-article-head">
                                 <span class="doc-head-content">Using MindSpore on the Cloud</span>
                              </div>
                              <div class="doc-article-desc">
                              ModelArts is a one-stop AI development platform provided by HUAWEI CLOUD. It integrates the Ascend AI Processor resource pool. Developers can experience MindSpore on this platform.
                              </div>
                        </div>
                     </a>
                  </div>
                  <div class="doc-article-item all hardware-ascend hardware-gpu stage-Design stage-Distributed-Parallel experience-local hidden">
                     <a href="https://www.mindspore.cn/docs/programming_guide/en/master/design/distributed_training_design.html" class="article-link">
                        <div>
                              <div class="doc-article-head">
                                 <span class="doc-head-content">Distributed Training Design</span>
                              </div>
                              <div class="doc-article-desc">
                               Parallel distributed training has become a development trend to resolve the performance bottleneck of ultra-large scale networks. MindSpore supports the mainstream distributed training paradigm and develops an automatic hybrid parallel solution.
                              </div>
                        </div>
                     </a>
                  </div>
                  <div class="doc-article-item all hardware-ascend hardware-gpu hardware-cpu stage-Design experience-local hidden">
                     <a href="https://www.mindspore.cn/docs/programming_guide/en/master/design/glossary.html" class="article-link">
                        <div>
                              <div class="doc-article-head">
                                 <span class="doc-head-content">Glossary</span>
                              </div>
                              <div class="doc-article-desc">
                              Here is the glossary in MindSpore.
                              </div>
                        </div>
                     </a>
                  </div>
                  <div class="doc-article-item all hardware-ascend hardware-gpu hardware-cpu stage-Design stage-Model-Development experience-local hidden">
                     <a href="https://www.mindspore.cn/docs/programming_guide/en/master/design/gradient.html" class="article-link">
                        <div>
                              <div class="doc-article-head">
                                 <span class="doc-head-content">MindSpore Automatic Differentiation</span>
                              </div>
                              <div class="doc-article-desc">
                             Automatic differentiation (AD) is one of the key techniques.
                              </div>
                        </div>
                     </a>
                  </div>
                  <div class="doc-article-item all hardware-ascend hardware-gpu stage-Design stage-Model-Development experience-local hidden">
                     <a href="https://www.mindspore.cn/docs/programming_guide/en/master/design/mindir.html" class="article-link">
                        <div>
                              <div class="doc-article-head">
                                 <span class="doc-head-content">MindSpore IR (MindIR)</span>
                              </div>
                              <div class="doc-article-desc">
                              An intermediate representation (IR) is a representation of a program between the source and target languages, which facilitates program analysis and optimization for the compiler.
                              </div>
                        </div>
                     </a>
                  </div>
                  <div class="doc-article-item all hardware-ascend hardware-gpu hardware-cpu stage-Design experience-local hidden">
                     <a href="https://www.mindspore.cn/docs/programming_guide/en/master/design/technical_white_paper.html" class="article-link">
                        <div>
                              <div class="doc-article-head">
                                 <span class="doc-head-content">Technical White Paper</span>
                              </div>
                              <div class="doc-article-desc">
                              Deep learning research and application have experienced explosive development in recent decades, triggering the third wave of artificial intelligence and achieving great success in image recognition, speech recognition and synthesis, unmanned driving, and machine vision.
                              </div>
                        </div>
                     </a>
                  </div>
						</div>

					</div>

				</div>
			</div>
		</div>
