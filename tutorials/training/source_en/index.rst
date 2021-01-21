.. MindSpore documentation master file, created by
   sphinx-quickstart on Thu Mar 24 09:00:00 2020.
   You can adapt this file completely to your liking, but it should at least
   contain the root `toctree` directive.

Train with MindSpore
==========================

.. toctree::
   :glob:
   :maxdepth: 1
   :caption: Quick Start
   :hidden:

   quick_start/quick_start
   quick_start/linear_regression
   quick_start/quick_video

.. toctree::
   :glob:
   :maxdepth: 1
   :caption: Basic Use
   :hidden:

   use/data_preparation
   use/defining_the_network
   use/save_model
   use/load_model_for_inference_and_transfer
   use/publish_model

.. toctree::
   :glob:
   :maxdepth: 1
   :caption: Process Data
   :hidden:

   advanced_use/convert_dataset
   advanced_use/optimize_data_processing

.. toctree::
   :glob:
   :maxdepth: 1
   :caption: Build Networks
   :hidden:

   advanced_use/custom_operator
   advanced_use/migrate_script
   advanced_use/apply_deep_probability_programming
   advanced_use/achieve_high_order_differentiation

.. toctree::
   :glob:
   :maxdepth: 1
   :caption: Model Optimization
   :hidden:

   advanced_use/debug_in_pynative_mode
   advanced_use/custom_debugging_info
   advanced_use/visualization_tutorials
   advanced_use/enable_auto_augmentation
   advanced_use/evaluate_the_model_during_training

.. toctree::
   :glob:
   :maxdepth: 1
   :caption: Performance Optimization
   :hidden:

   advanced_use/distributed_training_tutorials
   advanced_use/enable_mixed_precision
   advanced_use/enable_graph_kernel_fusion
   advanced_use/apply_gradient_accumulation
   advanced_use/enable_cache

.. toctree::
   :glob:
   :maxdepth: 1
   :caption: Model Compression
   :hidden:

   advanced_use/apply_quantization_aware_training

.. toctree::
   :glob:
   :maxdepth: 1
   :caption: Model Security and Privacy
   :hidden:

   advanced_use/improve_model_security_nad
   advanced_use/protect_user_privacy_with_differential_privacy
   advanced_use/test_model_security_fuzzing
   advanced_use/test_model_security_membership_inference

.. toctree::
   :glob:
   :maxdepth: 1
   :caption: Application
   :hidden:

   advanced_use/cv
   advanced_use/nlp
   advanced_use/hpc
   advanced_use/use_on_the_cloud

.. raw:: html

    <div class="container">
			<div class="row">
				<div class="col-md-12">
					<div>
						
						
						<div class="doc-condition">
							<span class="doc-filter">Filter condition</span>
							<button class="doc-delete doc-btn" id="all">Clear All Conditions</button>
						</div>
					
						<div class="doc-label-content">
							<div class="doc-label-choice">
								<div class="row">
									<div class="col-sm-2">
										<div class="doc-os">Operating System</div>
									</div>
									<div class="col-sm-10 col-sm-pull-1">
										<button class="doc-filter-btn doc-btn"  id="os-Windows">Windows</button>
										<button class="doc-filter-btn doc-btn" id="os-Linux" >Linux</button>
									</div>
								</div>
			
							</div>
							<div class="doc-label-choice">
								<div class="row">
								<div class="col-sm-2">
									<div class="doc-hardware">Hardware</div>
								</div>
								<div class="col-sm-10 col-sm-pull-1">
									<button class="doc-filter-btn doc-btn" id="hardware-Ascend">Ascend</button>
									<button class="doc-filter-btn doc-btn" id="hardware-GPU">GPU</button>
									<button class="doc-filter-btn doc-btn" id="hardware-CPU">CPU</button>
								</div>
							</div>
							</div>
							
							<div class="doc-label-choice">
								<div class="row">
									<div class="col-sm-2">
										<div class="doc-user">User</div>
									</div>
									<div class="col-sm-10 col-sm-pull-1">
										<button class="doc-filter-btn doc-btn" id="user-Beginner">Beginner</button>
										<button class="doc-filter-btn doc-btn" id="user-Intermediate">Intermediate</button>
										<button class="doc-filter-btn doc-btn" id="user-Expert">Expert</button>
										<button class="doc-filter-btn doc-btn" id="user-Enterprise">Enterprise</button>
									</div>
								</div>
							</div>
							
							<div class="doc-label-choice">
								<div class="row">
									<div class="col-sm-2 ">
										<div class="doc-stage">Stage</div>
									</div>							
									<div class="col-sm-10 col-sm-pull-1 doc-stage-detail">
										<button class="doc-filter-btn doc-btn" id="stage-Whole-Process">Whole Process</button>
										<button class="doc-filter-btn doc-btn" id="stage-Data-Preparation">Data preparation</button>
										<button class="doc-filter-btn doc-btn" id="stage-Model-Development">Model Development</button>
										<button class="doc-filter-btn doc-btn" id="stage-Model-Training">Model Training</button>
										<button class="doc-filter-btn doc-btn" id="stage-Model-Optimization">Model Optimization</button>
										<button class="doc-filter-btn doc-btn" id="stage-Model-Export">Model Export</button>
										<button class="doc-filter-btn doc-btn" id="stage-Model-Loading">Model Loading</button>
										<button class="doc-filter-btn doc-btn" id="stage-Inference">Inference Application</button>
										<button class="doc-filter-btn doc-btn" id="stage-Model-Publishing">Model Publishing</button>
										<button class="doc-filter-btn doc-btn" id="stage-Model-Evaluation">Model Evaluation</button>
									</div>
									
								</div>
								
							</div>
							
						</div>
						<hr>
						<div class="doc-article-list">
							<div class="doc-article-item all os-Linux os-Windows hardware-Ascend hardware-GPU hardware-CPU stage-Whole-Process user-Beginner user-Intermediate user-Expert hidden">
								<a href="https://www.mindspore.cn/tutorial/training/en/r1.1/quick_start/quick_start.html" class="article-link">
									<div>
										<div class="doc-article-head">
											<span class="doc-head-content">Implementing an Image Classification Application</span>
										</div>
							
										<div class="doc-article-desc">
											This tutorial uses a simple image classification example to demonstrate the basic functions of MindSpore.
										</div>
									</div>
								</a>
							</div>
							<div class="doc-article-item all os-Linux os-Windows hardware-Ascend hardware-GPU hardware-CPU stage-Whole-Process user-Beginner user-Intermediate user-Expert hidden">
								<a href="https://www.mindspore.cn/tutorial/training/en/r1.1/quick_start/linear_regression.html" class="article-link">
									<div>
										<div class="doc-article-head">
											<span class="doc-head-content">Implementing Simple Linear Function Fitting</span>
										</div>
										<div class="doc-article-desc">
											Regression algorithms usually use a series of properties to predict a value, and the predicted values are consecutive. This example describes the linear regression algorithms and how to use MindSpore to perform linear regression AI training.
										</div>
									</div>
								</a>
							</div>
							
							<div class="doc-article-item all os-Linux hardware-Ascend hardware-GPU hardware-CPU stage-Data-Preparation user-Beginner user-Intermediate user-Expert hidden">
								<a href="https://www.mindspore.cn/tutorial/training/en/r1.1/use/load_dataset_image.html" class="article-link">
									<div>
										<div class="doc-article-head">
											<span class="doc-head-content">Loading Image Dataset</span>
										</div>
										<div class="doc-article-desc">
											This tutorial uses the MNIST dataset as an example to demonstrate how to load and process image data using MindSpore.
												
										</div>
									</div>
								</a>
							</div>
							<div class="doc-article-item all os-Linux hardware-Ascend hardware-GPU hardware-CPU stage-Data-Preparation user-Beginner user-Intermediate user-Expert hidden">
								<a href="https://www.mindspore.cn/tutorial/training/en/r1.1/use/load_dataset_text.html" class="article-link">
									<div>
										<div class="doc-article-head">
											<span class="doc-head-content">Loading Text Dataset</span>
										</div>
										<div class="doc-article-desc">
											This tutorial briefly demonstrates how to load and process text data using MindSpore.
										</div>
									</div>
								</a>
							</div>
							
							<div class="doc-article-item all os-Linux hardware-Ascend hardware-GPU hardware-CPU stage-Model-Export user-Beginner user-Intermediate user-Expert hidden">
								<a href="https://www.mindspore.cn/tutorial/training/en/r1.1/use/save_model.html" class="article-link">
									<div>
										<div class="doc-article-head">
											<span class="doc-head-content">Saving Models</span>
										</div>
										<div class="doc-article-desc">
											This tutorial describes how to save MindSpore CheckPoint files, and how to export MindIR, AIR, ONNX files.
										</div>
									</div>
								</a>
							</div>
							<div class="doc-article-item all os-Linux hardware-Ascend hardware-GPU hardware-CPU stage-Model-Loading user-Beginner user-Intermediate user-Expert hidden">
								<a href="https://www.mindspore.cn/tutorial/training/en/r1.1/use/load_model_for_inference_and_transfer.html" class="article-link">
									<div>
										<div class="doc-article-head">
											<span class="doc-head-content">Loading a Model for Inference and Transfer Learning</span>
										</div>
										<div class="doc-article-desc">
											This tutorial uses examples to describe how to load models from local and MindSpore Hub.
										</div>
									</div>
								</a>
							</div>
							<div class="doc-article-item all os-Linux hardware-Ascend hardware-GPU stage-Model-Publishing  user-Intermediate user-Expert hidden">
								<a href="https://www.mindspore.cn/tutorial/training/en/r1.1/use/publish_model.html" class="article-link">
									<div>
										<div class="doc-article-head">
											<span class="doc-head-content">Publishing Models</span>
										</div>
										<div class="doc-article-desc">
											This tutorial uses GoogleNet as an example to describe how to submit models for model developers who are interested in publishing models into MindSpore Hub.
										</div>
									</div>
								</a>
							</div>
							<div class="doc-article-item all os-Linux hardware-Ascend hardware-GPU hardware-CPU stage-Data-Preparation user-Intermediate user-Expert hidden">
								<a href="https://www.mindspore.cn/tutorial/training/en/r1.1/advanced_use/convert_dataset.html" class="article-link">
									<div>
										<div class="doc-article-head">
											<span class="doc-head-content">Converting Dataset to MindRecord</span>
										</div>
										<div class="doc-article-desc">
											Users can convert non-standard datasets and common datasets into the MindSpore data format, MindRecord, so that they can be easily loaded to MindSpore for training. 
											In addition, the performance of MindSpore in some scenarios is optimized, which delivers better user experience when you use datasets in the MindSpore data format.
										</div>
									</div>
								</a>
							</div>
							<div class="doc-article-item all os-Linux hardware-Ascend hardware-GPU hardware-CPU stage-Data-Preparation user-Intermediate user-Expert hidden">
								<a href="https://www.mindspore.cn/tutorial/training/en/r1.1/advanced_use/optimize_data_processing.html" class="article-link">
									<div>
										<div class="doc-article-head">
											<span class="doc-head-content">Optimize the Data Processing</span>
										</div>
										<div class="doc-article-desc">
											MindSpore provides data processing and data augmentation functions for users. In the pipeline process, if each step can be properly used, the data performance will be greatly improved. 
											This section describes how to optimize performance during data loading, data processing, and data augmentation based on the CIFAR-10 dataset.
										</div>
									</div>
								</a>
							</div>
							<div class="doc-article-item all os-Linux hardware-Ascend stage-Model-Development user-Expert hidden">
								<a href="https://www.mindspore.cn/tutorial/training/en/r1.1/advanced_use/custom_operator_ascend.html" class="article-link">
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
					
							<div class="doc-article-item all os-Linux hardware-Ascend stage-Model-Development user-Beginner hidden">
								<a href="https://www.mindspore.cn/tutorial/training/en/r1.1/advanced_use/migrate_3rd_scripts_mindconverter.html" class="article-link">
									<div>
										<div class="doc-article-head">
											<span class="doc-head-content">Migrating from Third Party Frameworks with Tools</span>
										</div>
										<div class="doc-article-desc">
											MindConverter is a migration tool to transform the model scripts from PyTorch to Mindspore. Users can migrate their PyTorch models to Mindspore rapidly with minor changes according to the conversion report.
										</div>
									</div>
								</a>
							</div>
							<div class="doc-article-item all os-Linux hardware-Ascend hardware-GPU hardware-CPU stage-Whole-Process user-Beginner user-Intermediate user-Expert hidden">
								<a href="https://www.mindspore.cn/tutorial/training/en/r1.1/advanced_use/migrate_3rd_scripts.html" class="article-link">
									<div>
										<div class="doc-article-head">
											<span class="doc-head-content">Migrating Training Scripts from Third Party Frameworks</span>
										</div>
										<div class="doc-article-desc">
											This tutorial describes how to migrate existing TensorFlow and PyTorch networks to MindSpore, including key steps and operation recommendations which help you quickly migrate your network.
										</div>
									</div>
								</a>
							</div>

							<div class="doc-article-item all hardware-Ascend hardware-GPU stage-Whole-Process user-Beginner user-Intermediate user-Expert hidden">
								<a href="https://www.mindspore.cn/tutorial/training/en/r1.1/advanced_use/apply_deep_probability_programming.html" class="article-link">
									<div>
										<div class="doc-article-head">
											<span class="doc-head-content">Deep Probabilistic Programming</span>
										</div>
										<div class="doc-article-desc">
											MindSpore Deep Probabilistic Programming (MDP) combines deep learning and Bayesian learning. By setting a network weight to distribution and introducing latent space distribution, MDP can sample the distribution and forward propagation, which introduces uncertainty and enhances the robustness and explainability of a model.
											This chapter will introduce in detail the application of deep probabilistic programming on MindSpore.
										</div>
									</div>
								</a>
							</div>

							<div class="doc-article-item all os-Linux hardware-Ascend hardware-GPU hardware-CPU stage-Model-Development user-Beginner user-Intermediate user-Expert hidden">
								<a href="https://www.mindspore.cn/tutorial/training/en/r1.1/advanced_use/debug_in_pynative_mode.html" class="article-link">
									<div>
										<div class="doc-article-head">
											<span class="doc-head-content">Debugging in PyNative Mode</span>
										</div>
										<div class="doc-article-desc">
											In PyNative mode, single operators, common functions, network inference, and separated gradient calculation can be executed. This tutorial describes the usage and precautions in detail.
										</div>
									</div>
								</a>
							</div>
							<div class="doc-article-item all os-Linux hardware-Ascend hardware-GPU hardware-CPU stage-Model-Optimization user-Intermediate user-Expert hidden">
								<a href="https://www.mindspore.cn/tutorial/training/en/r1.1/advanced_use/custom_debugging_info.html" class="article-link">
									<div>
										<div class="doc-article-head">
											<span class="doc-head-content">Customized Debugging Information</span>
										</div>
										<div class="doc-article-desc">
											This tutorial describes how to use the customized capabilities provided by MindSpore, such as callback, metrics, and log printing, to help you quickly debug the training network.
										</div>
									</div>
								</a>
							</div>
							<div class="doc-article-item all os-Linux hardware-Ascend hardware-GPU hardware-CPU stage-Model-Optimization user-Intermediate user-Expert hidden">
								<a href="https://www.mindspore.cn/tutorial/training/en/r1.1/advanced_use/summary_record.html" class="article-link">
									<div>
										<div class="doc-article-head">
											<span class="doc-head-content">Collecting Summary Data</span>
										</div>
										<div class="doc-article-desc">
											Scalars, images, computational graphs, and model hyperparameters during training are recorded in files and can be viewed on the web page.
										</div>
									</div>
								</a>
							</div>
							<div class="doc-article-item all os-Linux hardware-Ascend hardware-GPU hardware-CPU stage-Model-Optimization user-Intermediate user-Expert hidden">
								<a href="https://www.mindspore.cn/tutorial/training/en/r1.1/advanced_use/dashboard.html" class="article-link">
									<div>
										<div class="doc-article-head">
											<span class="doc-head-content">Viewing Dashboard</span>
										</div>
										<div class="doc-article-desc">
											Training dashboard is an important part of mindinsight’s visualization component, and its tags include scalar visualization, parameter distribution visualization, computational graph visualization, data graph visualization, image visualization and tensor visualization.
										</div>
									</div>
								</a>
							</div>
							<div class="doc-article-item all os-Linux hardware-Ascend hardware-GPU hardware-CPU stage-Model-Optimization user-Intermediate user-Expert hidden">
								<a href="https://www.mindspore.cn/tutorial/training/en/r1.1/advanced_use/lineage_and_scalars_comparision.html" class="article-link">
									<div>
										<div class="doc-article-head">
											<span class="doc-head-content">Viewing Lineage and Scalars Comparison</span>
										</div>
										<div class="doc-article-desc">
											Model lineage, data lineage and comparison Kanban in mindinsight are the same as training dashboard. In the visualization of training data, different scalar trend charts are observed by comparison dashboard to find problems, 
											and then the lineage function is used to locate the problem causes, so as to give users the ability of efficient tuning in data enhancement and deep neural network.
										</div>
									</div>
								</a>
							</div>
							<div class="doc-article-item all os-Linux hardware-Ascend hardware-GPU hardware-CPU stage-Model-Optimization user-Intermediate user-Expert hidden">
								<a href="https://www.mindspore.cn/tutorial/training/en/r1.1/advanced_use/hyper_parameters_auto_tuning.html" class="article-link">
									<div>
										<div class="doc-article-head">
											<span class="doc-head-content">Use Mindoptimizer to Tune Hyperparameters</span>
										</div>
										<div class="doc-article-desc">
											Because different hyperparameters impact the performance of model, hyperparameters are highly important in training tasks. 
											Traditional methods require manual analysis of hyperparameters, manual debugging, and configuration, which consumes time and effort. MindInsight parameter tuning command can be used for automatic parameter tuning.
										</div>
									</div>
								</a>
							</div>
							<div class="doc-article-item all os-Linux hardware-Ascend stage-Model-Optimization user-Intermediate user-Expert hidden">
								<a href="https://www.mindspore.cn/tutorial/training/en/r1.1/advanced_use/performance_profiling.html" class="article-link">
									<div>
										<div class="doc-article-head">
											<span class="doc-head-content">Performance Profiling (Ascend)</span>
										</div>
										<div class="doc-article-desc">
											Performance data like operators’ execution time is recorded in files and can be viewed on the web page, this can help the user optimize the performance of neural networks. 
											This tutorial is applicable for the hardware platform of Ascend AI processors.
										</div>
									</div>
								</a>
							</div>
							<div class="doc-article-item all os-Linux hardware-GPU stage-Model-Optimization user-Intermediate user-Expert hidden">
								<a href="https://www.mindspore.cn/tutorial/training/en/r1.1/advanced_use/performance_profiling_gpu.html" class="article-link">
									<div>
										<div class="doc-article-head">
											<span class="doc-head-content">Performance Profiling (GPU)</span>
										</div>
										<div class="doc-article-desc">
											Performance data like operators’ execution time is recorded in files and can be viewed on the web page, this can help the user optimize the performance of neural networks. 
											This tutorial is applicable for the hardware platform of GPU.
										</div>
									</div>
								</a>
							</div>
							<div class="doc-article-item all os-Linux hardware-Ascend hardware-GPU  stage-Model-Optimization user-Intermediate user-Expert hidden">
								<a href="https://www.mindspore.cn/tutorial/training/en/r1.1/advanced_use/debugger.html" class="article-link">
									<div>
										<div class="doc-article-head">
											<span class="doc-head-content">Use Debugger</span>
										</div>
										<div class="doc-article-desc">
											MindSpore Debugger is a debugging tool for training in `Graph Mode`. It can be applied to visualize and analyze the intermediate computation results of the computational graph.
										</div>
									</div>
								</a>
							</div>
							<div class="doc-article-item all os-Linux hardware-Ascend hardware-GPU stage-Model-Optimization user-Beginner user-Intermediate user-Expert hidden">
								<a href="https://www.mindspore.cn/tutorial/training/en/r1.1/advanced_use/model_explaination.html" class="article-link">
									<div>
										<div class="doc-article-head">
											<span class="doc-head-content">Explain Models</span>
										</div>
										<div class="doc-article-desc">
											Currently, most deep learning models are black-box models with good performance but poor explainability. 
											The model explanation module aims to provide users with explanation of the model decision basis, help users better understand the model, trust the model, and improve the model when an error occurs in the model. This tutorial introduces how to use MindSpore to explain models.
										</div>
									</div>
								</a>
							</div>
							<div class="doc-article-item all os-Linux hardware-Ascend hardware-GPU hardware-CPU stage-Model-Optimization user-Intermediate user-Expert hidden">
								<a href="https://www.mindspore.cn/tutorial/training/en/r1.1/advanced_use/mindinsight_commands.html" class="article-link">
									<div>
										<div class="doc-article-head">
											<span class="doc-head-content">MindInsight Commands</span>
										</div>
										<div class="doc-article-desc">
											This tutorial introduces the MindInsight commands, including starting a service, stopping a service, exporting Summary, etc.
										</div>
									</div>
								</a>
							</div>
							<div class="doc-article-item all os-Linux hardware-Ascend hardware-GPU hardware-CPU stage-Data-Preparation user-Intermediate user-Expert hidden">
								<a href="https://www.mindspore.cn/tutorial/training/en/r1.1/advanced_use/enable_auto_augmentation.html" class="article-link">
									<div>
										<div class="doc-article-head">
											<span class="doc-head-content">Data Augmentation</span>
										</div>
										<div class="doc-article-desc">
											Auto Augmentation finds a suitable image augmentation scheme for a specific dataset by searching through a series of image augmentation sub-policies. 
											This tutorial introduces how to apply auto augmentation to the ImageNet dataset.
										</div>
									</div>
								</a>
							</div>
							<div class="doc-article-item all os-Linux hardware-Ascend hardware-GPU hardware-CPU stage-Model-Export stage-Model-Training user-Beginner user-Intermediate user-Expert hidden">
								<a href="https://www.mindspore.cn/tutorial/training/en/r1.1/advanced_use/evaluate_the_model_during_training.html" class="article-link">
									<div>
										<div class="doc-article-head">
											<span class="doc-head-content">Evaluating the Model during Training</span>
										</div>
										<div class="doc-article-desc">
											This tutorial applies a neural network, LeNet5, to the MINIST dataset and introduces how to evaluate the model during training, save models in corresponding epochs, and select the optimal model from the saved ones.
										</div>
									</div>
								</a>
							</div>
							<div class="doc-article-item all os-Linux hardware-Ascend stage-Model-Training user-Intermediate user-Expert hidden">
								<a href="https://www.mindspore.cn/tutorial/training/en/r1.1/advanced_use/distributed_training_ascend.html" class="article-link">
									<div>
										<div class="doc-article-head">
											<span class="doc-head-content">Parallel Distributed Training (Ascend)</span>
										</div>
										<div class="doc-article-desc">
											This tutorial describes how to train the ResNet-50 network in data parallel and automatic parallel modes on MindSpore based on the Ascend 910 AI processor.
										</div>
									</div>
								</a>
							</div>
							<div class="doc-article-item all os-Linux hardware-GPU stage-Model-Training user-Intermediate user-Expert hidden">
								<a href="https://www.mindspore.cn/tutorial/training/en/r1.1/advanced_use/distributed_training_gpu.html" class="article-link">
									<div>
										<div class="doc-article-head">
											<span class="doc-head-content">Parallel Distributed Training (GPU)</span>
										</div>
										<div class="doc-article-desc">
											This tutorial describes how to train the ResNet-50 network in data parallel and automatic parallel modes on MindSpore based on GPU.
										</div>
									</div>
								</a>
							</div>
							<div class="doc-article-item all os-Linux hardware-Ascend hardware-CPU stage-Model-Training user-Intermediate user-Expert hidden">
								<a href="https://www.mindspore.cn/tutorial/training/en/r1.1/advanced_use/apply_host_device_training.html" class="article-link">
									<div>
										<div class="doc-article-head">
											<span class="doc-head-content">Adopt Host&Device for Hybrid Training</span>
										</div>
										<div class="doc-article-desc">
											In Deep Learning, researchers have difficulty on training large models, because the requirement of model parameters may sometimes exceed the memory limitation of the equipment. 
											For efficiently training large models, one possible option is to adopt a hybrid training method that combines Host and Device. This hybrid method takes advantage of both sides: large memory of Host and great computation capability of Device. This tutorial takes Wide&Deep as an example to introduce how to combine Host and Ascend 910 AI processor for hybrid training in MindSpore.
										</div>
									</div>
								</a>
							</div>
							<div class="doc-article-item all os-Linux hardware-Ascend hardware-GPU stage-Model-Training user-Intermediate user-Expert hidden">
								<a href="https://www.mindspore.cn/tutorial/training/en/r1.1/advanced_use/apply_parameter_server_training.html" class="article-link">
									<div>
										<div class="doc-article-head">
											<span class="doc-head-content">Training with Parameter Server</span>
										</div>
										<div class="doc-article-desc">
											A parameter server is a widely used architecture in distributed training. Compared with the synchronous AllReduce training method, a parameter server has better flexibility, scalability, and node failover capabilities. 
											This tutorial describes how to use parameter server to train LeNet on Ascend 910.
										</div>
									</div>
								</a>
							</div>
							<div class="doc-article-item all os-Linux hardware-Ascend hardware-GPU stage-Model-Training user-Intermediate user-Expert hidden">
								<a href="https://www.mindspore.cn/tutorial/training/en/r1.1/advanced_use/save_load_model_hybrid_parallel.html" class="article-link">
									<div>
										<div class="doc-article-head">
											<span class="doc-head-content">Saving and Loading Models in Hybrid Parallel Mode</span>
										</div>
										<div class="doc-article-desc">
											In the hybrid parallel scenario, the dividing strategy is implemented by users. MindSpore saves the slice strategy of model, which is the same on each node, and the data corresponding to each node is stored respectively. 
											Users need to integrate, save, and load the checkpoint files by themselves. This tutorial describes how to integrate, save, and load checkpoint files in the hybrid parallel scenario.
										</div>
									</div>
								</a>
							</div>
							<div class="doc-article-item all os-Linux hardware-Ascend hardware-GPU stage-Model-Training user-Intermediate user-Expert hidden">
								<a href="https://www.mindspore.cn/tutorial/training/en/r1.1/advanced_use/enable_mixed_precision.html" class="article-link">
									<div>
										<div class="doc-article-head">
											<span class="doc-head-content">Enabling Mixed Precision</span>
										</div>
										<div class="doc-article-desc">
											The mixed precision training method accelerates the deep learning neural network training process by using both the single-precision and half-precision data formats, and maintains the network precision achieved by the single-precision training at the same time. 
											Mixed precision training can accelerate the computation process, reduce memory usage, and enable a larger model or batch size to be trained on specific hardware.
										</div>
									</div>
								</a>
							</div>
							<div class="doc-article-item all os-Linux hardware-Ascend hardware-GPU stage-Model-Optimization user-Intermediate user-Expert hidden">
								<a href="https://www.mindspore.cn/tutorial/training/en/r1.1/advanced_use/enable_graph_kernel_fusion.html" class="article-link">
									<div>
										<div class="doc-article-head">
											<span class="doc-head-content">Enabling Graph Kernel Fusion</span>
										</div>
										<div class="doc-article-desc">
											The graph kernel fusion is used to analyze and optimize the computational graph logic of the existing network, as well as split, reconstruct, and fuse the original computing logic to reduce the overhead of operator execution gaps and improve the computing resource utilization of devices, thereby optimizing the overall execution time of the network.
										</div>
									</div>
								</a>
							</div>
							<div class="doc-article-item all os-Linux hardware-GPU stage-Model-Optimization user-Intermediate user-Expert hidden">
								<a href="https://www.mindspore.cn/tutorial/training/en/r1.1/advanced_use/apply_gradient_accumulation.html" class="article-link">
									<div>
										<div class="doc-article-head">
											<span class="doc-head-content">Applying Gradient Accumulation Algorithm</span>
										</div>
										<div class="doc-article-desc">
											This tutorial describes the gradient accumulation training method to solve the problem that some large-scale networks cannot train large batch_size due to insufficient memory.
										</div>
									</div>
								</a>
							</div>
							<div class="doc-article-item all os-Linux hardware-Ascend hardware-GPU stage-Model-Optimization user-Expert hidden">
								<a href="https://www.mindspore.cn/tutorial/training/en/r1.1/advanced_use/apply_quantization_aware_training.html" class="article-link">
									<div>
										<div class="doc-article-head">
											<span class="doc-head-content">Applying Quantization Aware Training</span>
										</div>
										<div class="doc-article-desc">
											MindSpore quantization aware training is to replace high-accuracy data with low-accuracy data to simplify the model training process. In this process, the accuracy loss is inevitable. 
											Therefore, a fake quantization node is used to simulate the accuracy loss, and backward propagation learning is used to reduce the accuracy loss.
										</div>
									</div>
								</a>
							</div>
							<div class="doc-article-item all os-Linux hardware-Ascend hardware-GPU hardware-CPU stage-Model-Training stage-Model-Optimization user-Enterprise user-Expert hidden">
								<a href="https://www.mindspore.cn/tutorial/training/en/r1.1/advanced_use/improve_model_security_nad.html" class="article-link">
									<div>
										<div class="doc-article-head">
											<span class="doc-head-content">Applying NAD algorithm to Improve the Security of a Model</span>
										</div>
										<div class="doc-article-desc">
											This tutorial introduces security measures provided by MindArmour, leading you to a quick start of MindArmour. MindArmour is able to provide security for your AI models.
										</div>
									</div>
								</a>
							</div>
							<div class="doc-article-item all os-Linux hardware-Ascend stage-Model-Training stage-Model-Optimization user-Enterprise user-Expert hidden">
								<a href="https://www.mindspore.cn/tutorial/training/en/r1.1/advanced_use/protect_user_privacy_with_differential_privacy.html" class="article-link">
									<div>
										<div class="doc-article-head">
											<span class="doc-head-content">Protecting User Privacy with Differential Privacy Mechanism</span>
										</div>
										<div class="doc-article-desc">
											Machine learning algorithms usually update model parameters and learn data features based on a large amount of data. Ideally, these models can learn the common features of a class of entities and achieve good generalization, such as “smoking patients are more likely to get lung cancer” rather than models with individual features, such as “Zhang San is a smoker who gets lung cancer.” 
											However, machine learning algorithms do not distinguish between general and individual features. The published machine learning models, especially the deep neural networks, may unintentionally memorize and expose the features of individual entities in training data. This can be exploited by malicious attackers to reveal Zhang San’s privacy information from the published model. 
											Therefore, it is necessary to use differential privacy to protect machine learning models from privacy leakage.
										</div>
									</div>
								</a>
							</div>
							<div class="doc-article-item all os-Linux hardware-Ascend hardware-GPU hardware-CPU stage-Model-Evaluation user-Enterprise user-Expert hidden">
								<a href="https://www.mindspore.cn/tutorial/training/en/r1.1/advanced_use/test_model_security_fuzzing.html" class="article-link">
									<div>
										<div class="doc-article-head">
											<span class="doc-head-content">Applying the fuzz_testing Module to Test Security of a Model</span>
										</div>
										<div class="doc-article-desc">
											The fuzz_testing module in MindArmour treats neuron coverage as the test criterion. We use neuron coverage to guide input mutation, so that the input can activate more neurons and the distribution of neuron values is wider, so as to explore different types of model output results and wrong behaviors.
										</div>
									</div>
								</a>
							</div>
							<div class="doc-article-item all os-Linux hardware-Ascend hardware-GPU hardware-CPU stage-Model-Evaluation user-Enterprise user-Expert hidden">
								<a href="https://www.mindspore.cn/tutorial/training/en/r1.1/advanced_use/test_model_security_membership_inference.html" class="article-link">
									<div>
										<div class="doc-article-head">
											<span class="doc-head-content">Using Membership Inference to Test Model Security</span>
										</div>
										<div class="doc-article-desc">
											In machine learning and deep learning, if an attacker has some access permissions (black box, gray box, or white box) of a model to obtain some or all information about the model output, structure, or parameters, they can determine whether a sample belongs to a training set of a model. In this case, we can use membership inference to evaluate the privacy data security of machine learning and deep learning models.
											If more than 60% samples can be correctly inferred using membership inference, the model has privacy data leakage risks.
										</div>
									</div>
								</a>
							</div>
							<div class="doc-article-item all os-Linux hardware-Ascend hardware-GPU stage-Whole-Process user-Beginner user-Intermediate user-Expert hidden">
								<a href="https://www.mindspore.cn/tutorial/training/en/r1.1/advanced_use/cv_resnet50.html" class="article-link">
									<div>
										<div class="doc-article-head">
											<span class="doc-head-content">Image Classification Using ResNet-50 Network</span>
										</div>
										<div class="doc-article-desc">
											Computer vision is one of the most widely researched and mature technology fields of deep learning, and is widely applied to scenarios such as mobile phone photographing, intelligent security protection, and automated driving. 
											This tutorial describes how to apply MindSpore to computer vision scenarios based on image classification tasks.
										</div>
									</div>
								</a>
							</div>
							<div class="doc-article-item all os-Linux hardware-Ascend hardware-GPU stage-Model-Development stage-Model-Optimization user-Expert hidden">
								<a href="https://www.mindspore.cn/tutorial/training/en/r1.1/advanced_use/cv_resnet50_second_order_optimizer.html" class="article-link">
									<div>
										<div class="doc-article-head">
											<span class="doc-head-content">ResNet-50 Second-Order Optimization Practice</span>
										</div>
										<div class="doc-article-desc">
											This tutorial describes how to use the second-order optimizer THOR provided by MindSpore to train the ResNet-50 v1.5 network and ImageNet dataset on Ascend 910 and GPU.
										</div>
									</div>
								</a>
							</div>
							<div class="doc-article-item all os-Windows os-Linux hardware-CPU hardware-Ascend hardware-GPU stage-Model-Development user-Intermediate user-Expert hidden">
								<a href="https://www.mindspore.cn/tutorial/training/en/r1.1/advanced_use/cv_mobilenetv2_fine_tune.html" class="article-link">
									<div>
										<div class="doc-article-head">
											<span class="doc-head-content">Using MobileNetV2 to Implement Fine-Tuning</span>
										</div>
										<div class="doc-article-desc">
											This tutorial describes how to perform fine-tuning training and validation in the MindSpore frameworks of different systems and processors.
										</div>
									</div>
								</a>
							</div>
							<div class="doc-article-item all os-Linux hardware-GPU hardware-CPU stage-Whole-Process user-Beginner user-Intermediate user-Expert hidden">
								<a href="https://www.mindspore.cn/tutorial/training/en/r1.1/advanced_use/nlp_sentimentnet.html" class="article-link">
									<div>
										<div class="doc-article-head">
											<span class="doc-head-content">Realizing Sentiment Classification with SentimentNet</span>
										</div>
										<div class="doc-article-desc">
											Sentiment classification is a subset of text classification in NLP, and is one of the most basic applications of NLP. It is a process of analyzing and inferencing affective states and subjective information, that is, analyzing whether a person’s sentiment is positive or negative.
										</div>
									</div>
								</a>
							</div>
							<div class="doc-article-item all os-Linux hardware-Ascend stage-Model-Training stage-Inference user-Expert hidden">
								<a href="https://www.mindspore.cn/tutorial/training/en/r1.1/advanced_use/nlp_bert_poetry.html" class="article-link">
									<div>
										<div class="doc-article-head">
											<span class="doc-head-content">Using the BERT Network to Implement Intelligent Poem Writing</span>
										</div>
										<div class="doc-article-desc">
											This tutorial introduces how to use MindSpore to train an intelligent poem writing model and deploy the prediction service.
										</div>
									</div>
								</a>
							</div>
			
						</div>
						<div class="doc-footer">
							<nav aria-label="Page navigation">
								<ul class="pagination" id="pageNav">
									
								</ul>
							</nav>
						</div>					
					</div>
					
				</div>
			</div>
		</div>
		
