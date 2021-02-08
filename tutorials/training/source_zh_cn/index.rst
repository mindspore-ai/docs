.. MindSpore documentation master file, created by
   sphinx-quickstart on Thu Mar 24 09:00:00 2020.
   You can adapt this file completely to your liking, but it should at least
   contain the root `toctree` directive.

使用MindSpore进行训练
=========================

.. toctree::
   :glob:
   :maxdepth: 1
   :caption: 快速入门
   :hidden:

   quick_start/quick_start
   quick_start/linear_regression
   quick_start/quick_video

.. toctree::
   :glob:
   :maxdepth: 1
   :caption: 基础使用
   :hidden:

   use/data_preparation
   use/defining_the_network
   use/save_model
   use/load_model_for_inference_and_transfer
   use/publish_model

.. toctree::
   :glob:
   :maxdepth: 1
   :caption: 处理数据
   :hidden:

   advanced_use/convert_dataset
   advanced_use/optimize_data_processing


.. toctree::
   :glob:
   :maxdepth: 1
   :caption: 构建网络
   :hidden:

   advanced_use/custom_operator
   advanced_use/migrate_script
   advanced_use/apply_deep_probability_programming
   advanced_use/implement_high_order_differentiation

.. toctree::
   :glob:
   :maxdepth: 1
   :caption: 调试网络
   :hidden:

   advanced_use/debug_in_pynative_mode
   advanced_use/dump_data_from_ir_files
   advanced_use/custom_debugging_info
   advanced_use/visualization_tutorials
   advanced_use/enable_auto_augmentation
   advanced_use/evaluate_the_model_during_training

.. toctree::
   :glob:
   :maxdepth: 1
   :caption: 优化训练性能
   :hidden:

   advanced_use/distributed_training_tutorials
   advanced_use/enable_mixed_precision
   advanced_use/enable_graph_kernel_fusion
   advanced_use/apply_gradient_accumulation
   advanced_use/enable_cache

.. toctree::
   :glob:
   :maxdepth: 1
   :caption: 压缩模型
   :hidden:

   advanced_use/apply_quantization_aware_training

.. toctree::
   :glob:
   :maxdepth: 1
   :caption: 模型安全和隐私
   :hidden:

   advanced_use/improve_model_security_nad
   advanced_use/protect_user_privacy_with_differential_privacy
   advanced_use/test_model_security_fuzzing
   advanced_use/test_model_security_membership_inference

.. toctree::
   :glob:
   :maxdepth: 1
   :caption: 应用实践
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
							<span class="doc-filter">筛选条件</span>
							<button class="doc-delete doc-btn" id="all">清除所有条件</button>
						</div>
					
						<div class="doc-label-content">
							<div class="doc-label-choice">
								<div class="row">
									<div class="col-sm-2">
										<div class="doc-os">操作系统</div>
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
									<div class="doc-hardware">硬件</div>
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
										<div class="doc-user">用户</div>
									</div>
									<div class="col-sm-10 col-sm-pull-1">
										<button class="doc-filter-btn doc-btn" id="user-Beginner">初级</button>
										<button class="doc-filter-btn doc-btn" id="user-Intermediate">中级</button>
										<button class="doc-filter-btn doc-btn" id="user-Expert">高级</button>
										<button class="doc-filter-btn doc-btn" id="user-Enterprise">企业</button>
									</div>
								</div>
							</div>
							
							<div class="doc-label-choice">
								<div class="row">
									<div class="col-sm-2 ">
										<div class="doc-stage">阶段</div>
									</div>							
									<div class="col-sm-10 col-sm-pull-1 doc-stage-detail">
										<button class="doc-filter-btn doc-btn" id="stage-Whole-Process">全流程</button>
										<button class="doc-filter-btn doc-btn" id="stage-Data-Preparation">数据准备</button>
										<button class="doc-filter-btn doc-btn" id="stage-Model-Development">模型开发</button>
										<button class="doc-filter-btn doc-btn" id="stage-Model-Training">模型训练</button>
										<button class="doc-filter-btn doc-btn" id="stage-Model-Optimization">模型调优</button>
										<button class="doc-filter-btn doc-btn" id="stage-Model-Export">模型导出</button>
										<button class="doc-filter-btn doc-btn" id="stage-Model-Loading">模型加载</button>
										<button class="doc-filter-btn doc-btn" id="stage-Inference">推理应用</button>
										<button class="doc-filter-btn doc-btn" id="stage-Model-Publishing">模型发布</button>
										<button class="doc-filter-btn doc-btn" id="stage-Model-Evaluation">模型评测</button>
									</div>
									
								</div>
							</div>							
							<div class="doc-label-choice">
								<div class="row">
									<div class="col-sm-2 ">
										<div class="doc-experience">体验</div>
									</div>							
									<div class="col-sm-10 col-sm-pull-1 doc-stage-detail">
										<button class="doc-filter-btn doc-btn" id="experience-Online-Experience">在线体验</button>
										<button class="doc-filter-btn doc-btn" id="experience-Local-Experience">本地体验</button>
									</div>
									
								</div>
								
							</div>
							
						</div>
						<hr>

						<div class="doc-footer">
							<nav aria-label="Page navigation">
								<ul class="pagination" id="pageNav">
									
								</ul>
							</nav>
						</div>

						<div class="doc-article-list">
							<div class="doc-article-item all os-Linux os-Windows hardware-Ascend hardware-GPU hardware-CPU stage-Whole-Process user-Beginner user-Intermediate user-Expert experience-Online-Experience experience-Local-Experience hidden">
								<a href="https://www.mindspore.cn/tutorial/training/zh-CN/master/quick_start/quick_start.html" class="article-link">
									<div>
										<div class="doc-article-head">
											<span class="doc-head-content">实现一个图片分类应用</span>
										</div>
							
										<div class="doc-article-desc">
											本教程通过实现一个简单的图片分类的功能，带领大家体验MindSpore基础的功能。
										</div>
									</div>
								</a>
							</div>
							<div class="doc-article-item all os-Linux os-Windows hardware-Ascend hardware-GPU hardware-CPU stage-Whole-Process user-Beginner user-Intermediate user-Expert experience-Online-Experience experience-Local-Experience hidden">
								<a href="https://www.mindspore.cn/tutorial/training/zh-CN/master/quick_start/linear_regression.html" class="article-link">
									<div>
										<div class="doc-article-head">
											<span class="doc-head-content">实现简单线性函数拟合</span>
										</div>
										<div class="doc-article-desc">
											回归问题算法通常是利用一系列属性来预测一个值，预测的值是连续的。本教程介绍线性回归算法，并通过MindSpore进行线性回归AI训练体验。
										</div>
									</div>
								</a>
							</div>
							
							<div class="doc-article-item all os-Linux hardware-Ascend hardware-GPU hardware-CPU stage-Data-Preparation user-Beginner user-Intermediate user-Expert experience-Online-Experience experience-Local-Experience hidden">
								<a href="https://www.mindspore.cn/tutorial/training/zh-CN/master/use/load_dataset_image.html" class="article-link">
									<div>
										<div class="doc-article-head">
											<span class="doc-head-content">加载图像数据集</span>
										</div>
										<div class="doc-article-desc">
											本教程将以加载MNIST数据集为例，演示如何使用MindSpore加载和处理图像数据。
												
										</div>
									</div>
								</a>
							</div>
							<div class="doc-article-item all os-Linux hardware-Ascend hardware-GPU hardware-CPU stage-Data-Preparation user-Beginner user-Intermediate user-Expert experience-Online-Experience experience-Local-Experience hidden">
								<a href="https://www.mindspore.cn/tutorial/training/zh-CN/master/use/load_dataset_text.html" class="article-link">
									<div>
										<div class="doc-article-head">
											<span class="doc-head-content">加载文本数据集</span>
										</div>
										<div class="doc-article-desc">
											本教程将简要演示如何使用MindSpore加载和处理文本数据。
										</div>
									</div>
								</a>
							</div>
							<div class="doc-article-item all os-Linux hardware-Ascend hardware-GPU hardware-CPU stage-Model-Development user-Beginner user-Intermediate user-Expert hidden">
								<a href="https://www.mindspore.cn/tutorial/training/zh-CN/master/use/defining_the_network.html" class="article-link">
									<div>
										<div class="doc-article-head">
											<span class="doc-head-content">定义网络</span>
										</div>
										<div class="doc-article-desc">
											由多个层组成的神经网络模型，是训练过程的重要组成部分。本教程介绍了多种定义网络模型的方式。
										</div>
									</div>
								</a>
							</div>
							<div class="doc-article-item all os-Linux hardware-Ascend hardware-GPU hardware-CPU stage-Model-Export user-Beginner user-Intermediate user-Expert experience-Online-Experience experience-Local-Experience hidden">
								<a href="https://www.mindspore.cn/tutorial/training/zh-CN/master/use/save_model.html" class="article-link">
									<div>
										<div class="doc-article-head">
											<span class="doc-head-content">保存模型</span>
										</div>
										<div class="doc-article-desc">
											本教程通过示例来介绍保存CheckPoint格式文件和导出MindIR、AIR和ONNX格式文件的方法。
										</div>
									</div>
								</a>
							</div>
							<div class="doc-article-item all os-Linux hardware-Ascend hardware-GPU hardware-CPU stage-Model-Loading user-Beginner user-Intermediate user-Expert experience-Online-Experience experience-Local-Experience hidden">
								<a href="https://www.mindspore.cn/tutorial/training/zh-CN/master/use/load_model_for_inference_and_transfer.html" class="article-link">
									<div>
										<div class="doc-article-head">
											<span class="doc-head-content">加载模型用于推理或迁移学习</span>
										</div>
										<div class="doc-article-desc">
											本教程通过示例来介绍如何通过本地加载或Hub加载模型，用于推理验证和迁移学习。
										</div>
									</div>
								</a>
							</div>
							<div class="doc-article-item all os-Linux hardware-Ascend hardware-GPU stage-Model-Publishing  user-Intermediate user-Expert hidden">
								<a href="https://www.mindspore.cn/tutorial/training/zh-CN/master/use/publish_model.html" class="article-link">
									<div>
										<div class="doc-article-head">
											<span class="doc-head-content">发布模型</span>
										</div>
										<div class="doc-article-desc">
											本教程以GoogleNet为例，向想要将模型发布到MindSpore Hub的模型开发者介绍了模型上传步骤。
										</div>
									</div>
								</a>
							</div>
							<div class="doc-article-item all os-Linux hardware-Ascend hardware-GPU hardware-CPU stage-Data-Preparation user-Intermediate user-Expert experience-Online-Experience experience-Local-Experience hidden">
								<a href="https://www.mindspore.cn/tutorial/training/zh-CN/master/advanced_use/convert_dataset.html" class="article-link">
									<div>
										<div class="doc-article-head">
											<span class="doc-head-content">转换数据集为MindRecord</span>
										</div>
										<div class="doc-article-desc">
											用户可以将非标准的数据集和常用的数据集转换为MindSpore数据格式，即MindRecord，从而方便地加载到MindSpore中进行训练。同时，MindSpore在部分场景做了性能优化，使用MindSpore数据格式可以获得更好的性能。
										</div>
									</div>
								</a>
							</div>
							<div class="doc-article-item all os-Linux hardware-Ascend hardware-GPU hardware-CPU stage-Data-Preparation user-Intermediate user-Expert experience-Online-Experience experience-Local-Experience hidden">
								<a href="https://www.mindspore.cn/tutorial/training/zh-CN/master/advanced_use/optimize_data_processing.html" class="article-link">
									<div>
										<div class="doc-article-head">
											<span class="doc-head-content">优化数据处理</span>
										</div>
										<div class="doc-article-desc">
											MindSpore为用户提供了数据处理和数据增强的功能，在整个pipeline过程中的每一步骤，如果都能够进行合理的运用，那么数据的性能会得到很大的优化和提升。本教程将基于CIFAR-10数据集来为大家展示如何在数据加载、数据处理和数据增强的过程中进行性能的优化。
										</div>
									</div>
								</a>
							</div>
							<div class="doc-article-item all os-Linux hardware-Ascend stage-Model-Development user-Expert hidden">
								<a href="https://www.mindspore.cn/tutorial/training/zh-CN/master/advanced_use/custom_operator_ascend.html" class="article-link">
									<div>
										<div class="doc-article-head">
											<span class="doc-head-content">自定义算子（Ascend）</span>
										</div>
										<div class="doc-article-desc">
											当开发网络遇到内置算子不足以满足需求时，你可以利用MindSpore的Python API方便快捷地扩展昇腾AI处理器的自定义算子。
										</div>
									</div>
								</a>
							</div>
							<div class="doc-article-item all os-Linux hardware-GPU stage-Model-Development user-Expert hidden">
								<a href="https://www.mindspore.cn/tutorial/training/zh-CN/master/advanced_use/custom_operator_gpu.html" class="article-link">
									<div>
										<div class="doc-article-head">
											<span class="doc-head-content">自定义算子（GPU）</span>
										</div>
										<div class="doc-article-desc">
											当开发网络遇到内置算子不足以满足需求时，你可以利用MindSpore的Python API、C++ API和CUDA方便快捷地扩展GPU端的自定义算子。
										</div>
									</div>
								</a>
							</div>							
							<div class="doc-article-item all os-Linux hardware-CPU stage-Model-Development user-Expert hidden">
								<a href="https://www.mindspore.cn/tutorial/training/zh-CN/master/advanced_use/custom_operator_cpu.html" class="article-link">
									<div>
										<div class="doc-article-head">
											<span class="doc-head-content">自定义算子（CPU）</span>
										</div>
										<div class="doc-article-desc">
											当开发网络遇到内置算子不足以满足需求时，你可以利用MindSpore的Python API和C++ API方便快捷地扩展CPU端的自定义算子。
										</div>
									</div>
								</a>
							</div>
							<div class="doc-article-item all os-Linux hardware-Ascend stage-Model-Development user-Beginner hidden">
								<a href="https://www.mindspore.cn/tutorial/training/zh-CN/master/advanced_use/migrate_3rd_scripts_mindconverter.html" class="article-link">
									<div>
										<div class="doc-article-head">
											<span class="doc-head-content">使用工具迁移第三方框架脚本</span>
										</div>
										<div class="doc-article-desc">
											MindConverter是一款将PyTorch模型脚本转换至MindSpore的脚本迁移工具。结合转换报告的提示信息，用户对转换后脚本进行微小改动，即可快速将PyTorch模型脚本迁移至MindSpore。
										</div>
									</div>
								</a>
							</div>
							<div class="doc-article-item all os-Linux hardware-Ascend hardware-GPU hardware-CPU stage-Whole-Process user-Beginner user-Intermediate user-Expert hidden">
								<a href="https://www.mindspore.cn/tutorial/training/zh-CN/master/advanced_use/migrate_3rd_scripts.html" class="article-link">
									<div>
										<div class="doc-article-head">
											<span class="doc-head-content">迁移第三方框架训练脚本</span>
										</div>
										<div class="doc-article-desc">
											本教程介绍如何将已有的TensorFlow、PyTorch等的网络迁移到MindSpore，包括主要步骤和操作建议，帮助你快速进行网络迁移。
										</div>
									</div>
								</a>
							</div>
							<div class="doc-article-item all os-Linux hardware-Ascend hardware-GPU stage-Whole-Process user-Beginner user-Intermediate user-Expert hidden">
								<a href="https://www.mindspore.cn/tutorial/training/zh-CN/master/advanced_use/apply_deep_probability_programming.html" class="article-link">
									<div>
										<div class="doc-article-head">
											<span class="doc-head-content">深度概率编程</span>
										</div>
										<div class="doc-article-desc">
											MindSpore深度概率编程（MindSpore Deep Probabilistic Programming, MDP）将深度学习和贝叶斯学习结合，通过设置网络权重为分布、引入隐空间分布等，可以对分布进行采样前向传播，由此引入了不确定性，从而增强了模型的鲁棒性和可解释性。本章将详细介绍深度概率编程在MindSpore上的应用。
										</div>
									</div>
								</a>
							</div>
							<div class="doc-article-item all hardware-Ascend hardware-GPU hardware-CPU stage-Whole-Process user-Beginner user-Intermediate user-Expert hidden">
								<a href="https://www.mindspore.cn/tutorial/training/zh-CN/master/advanced_use/implement_high_order_differentiation.html" class="article-link">
									<div>
										<div class="doc-article-head">
											<span class="doc-head-content">实现高阶自动微分</span>
										</div>
										<div class="doc-article-desc">
											本教程主要介绍MindSpore图模式下的高阶导数。
										</div>
									</div>
								</a>
							</div>
							<div class="doc-article-item all os-Linux hardware-Ascend hardware-GPU hardware-CPU stage-Model-Development user-Beginner user-Intermediate user-Expert experience-Online-Experience experience-Local-Experience hidden">
								<a href="https://www.mindspore.cn/tutorial/training/zh-CN/master/advanced_use/debug_in_pynative_mode.html" class="article-link">
									<div>
										<div class="doc-article-head">
											<span class="doc-head-content">使用PyNative模式调试</span>
										</div>
										<div class="doc-article-desc">
											PyNative模式下，支持执行单算子、普通函数和网络，以及单独求梯度的操作，本教程将详细介绍使用方法和注意事项。
										</div>
									</div>
								</a>
							</div>
							<div class="doc-article-item all os-Linux hardware-Ascend hardware-GPU stage-Model-Development user-Beginner user-Intermediate user-Expert hidden">
								<a href="https://www.mindspore.cn/tutorial/training/zh-CN/master/advanced_use/dump_data_from_ir_files.html" class="article-link">
									<div>
										<div class="doc-article-head">
											<span class="doc-head-content">借助IR文件进行调试</span>
										</div>
										<div class="doc-article-desc">
											Graph模式下，生成训练时产生的IR文件，并根据此文件Dump出特定算子供于调试。本教程将详细介绍使用方法和注意事项。
										</div>
									</div>
								</a>
							</div>
							<div class="doc-article-item all os-Linux hardware-Ascend hardware-GPU hardware-CPU stage-Model-Optimization user-Intermediate user-Expert experience-Online-Experience experience-Local-Experience hidden">
								<a href="https://www.mindspore.cn/tutorial/training/zh-CN/master/advanced_use/custom_debugging_info.html" class="article-link">
									<div>
										<div class="doc-article-head">
											<span class="doc-head-content">自定义调试信息</span>
										</div>
										<div class="doc-article-desc">
											本教程介绍如何使用MindSpore提供的Callback、metrics、Print算子、日志打印等自定义能力，帮助用户快速调试训练网络。
										</div>
									</div>
								</a>
							</div>
							<div class="doc-article-item all os-Linux hardware-Ascend hardware-GPU hardware-CPU stage-Model-Optimization user-Intermediate user-Expert experience-Local-Experience hidden">
								<a href="https://www.mindspore.cn/tutorial/training/zh-CN/master/advanced_use/summary_record.html" class="article-link">
									<div>
										<div class="doc-article-head">
											<span class="doc-head-content">收集Summary数据</span>
										</div>
										<div class="doc-article-desc">
											训练过程中的标量、图像、计算图以及模型超参等信息记录到文件中，通过可视化界面供用户查看。
										</div>
									</div>
								</a>
							</div>
							<div class="doc-article-item all os-Linux hardware-Ascend hardware-GPU hardware-CPU stage-Model-Optimization user-Intermediate user-Expert experience-Local-Experience hidden">
								<a href="https://www.mindspore.cn/tutorial/training/zh-CN/master/advanced_use/dashboard.html" class="article-link">
									<div>
										<div class="doc-article-head">
											<span class="doc-head-content">查看训练看板</span>
										</div>
										<div class="doc-article-desc">
											训练看板是MindInsight的可视化组件的重要组成部分，而训练看板的标签包含：标量可视化、参数分布图可视化、计算图可视化、数据图可视化、图像可视化和张量可视化等。
										</div>
									</div>
								</a>
							</div>
							<div class="doc-article-item all os-Linux hardware-Ascend hardware-GPU hardware-CPU stage-Model-Optimization user-Intermediate user-Expert experience-Local-Experience hidden">
								<a href="https://www.mindspore.cn/tutorial/training/zh-CN/master/advanced_use/lineage_and_scalars_comparison.html" class="article-link">
									<div>
										<div class="doc-article-head">
											<span class="doc-head-content">查看溯源和对比看板</span>
										</div>
										<div class="doc-article-desc">
											MindInsight中的模型溯源、数据溯源和对比看板同训练看板一样属于可视化组件中的重要组成部分，在对训练数据的可视化中，通过对比看板观察不同标量趋势图发现问题，再使用溯源功能定位问题原因，给用户在数据增强和深度神经网络中提供高效调优的能力。
										</div>
									</div>
								</a>
							</div>
							<div class="doc-article-item all os-Linux hardware-Ascend hardware-GPU hardware-CPU stage-Model-Optimization user-Intermediate user-Expert hidden">
								<a href="https://www.mindspore.cn/tutorial/training/zh-CN/master/advanced_use/hyper_parameters_auto_tuning.html" class="article-link">
									<div>
										<div class="doc-article-head">
											<span class="doc-head-content">使用mindoptimizer进行超参调优</span>
										</div>
										<div class="doc-article-desc">
											不同的超参数会对模型效果有不小的影响，因此超参在训练任务中的重要性较高。传统的方式都需要人工去调试和配置，这种方式消耗时间和精力。MindInsight调参功能可以用于搜索超参，基于用户给的调参配置信息，可以自动搜索参数并且执行模型训练。
										</div>
									</div>
								</a>
							</div>
							<div class="doc-article-item all os-Linux hardware-Ascend stage-Model-Optimization user-Intermediate user-Expert hidden">
								<a href="https://www.mindspore.cn/tutorial/training/zh-CN/master/advanced_use/performance_profiling.html" class="article-link">
									<div>
										<div class="doc-article-head">
											<span class="doc-head-content">性能调试（Ascend）</span>
										</div>
										<div class="doc-article-desc">
											将训练过程中的算子耗时等信息记录到文件中，通过可视化界面供用户查看分析，帮助用户更高效地调试神经网络性能。本教程适用于Ascend AI处理器硬件平台。
										</div>
									</div>
								</a>
							</div>
							<div class="doc-article-item all os-Linux hardware-GPU stage-Model-Optimization user-Intermediate user-Expert hidden">
								<a href="https://www.mindspore.cn/tutorial/training/zh-CN/master/advanced_use/performance_profiling_gpu.html" class="article-link">
									<div>
										<div class="doc-article-head">
											<span class="doc-head-content">性能调试（GPU）</span>
										</div>
										<div class="doc-article-desc">
											将训练过程中的算子耗时等信息记录到文件中，通过可视化界面供用户查看分析，帮助用户更高效地调试神经网络性能。本教程适用于GPU硬件平台。
										</div>
									</div>
								</a>
							</div>
							<div class="doc-article-item all os-Linux hardware-Ascend hardware-GPU  stage-Model-Optimization user-Intermediate user-Expert hidden">
								<a href="https://www.mindspore.cn/tutorial/training/zh-CN/master/advanced_use/debugger.html" class="article-link">
									<div>
										<div class="doc-article-head">
											<span class="doc-head-content">使用调试器</span>
										</div>
										<div class="doc-article-desc">
											MindSpore调试器是为图模式训练提供的调试工具，可以用来查看并分析计算图节点的中间结果。
										</div>
									</div>
								</a>
							</div>
							<div class="doc-article-item all os-Linux hardware-Ascend hardware-GPU stage-Model-Optimization user-Beginner user-Intermediate user-Expert hidden">
								<a href="https://www.mindspore.cn/tutorial/training/zh-CN/master/advanced_use/model_explaination.html" class="article-link">
									<div>
										<div class="doc-article-head">
											<span class="doc-head-content">解释模型</span>
										</div>
										<div class="doc-article-desc">
											当前深度学习模型多为黑盒模型，性能表现好但可解释性较差。模型解释模块旨在为用户提供对模型决策依据的解释，帮助用户更好地理解模型、信任模型，以及当模型出现错误时有针对性地改进模型效果。本教程介绍如何使用MindSpore进行模型解释。
										</div>
									</div>
								</a>
							</div>
							<div class="doc-article-item all os-Linux hardware-Ascend hardware-GPU hardware-CPU stage-Model-Optimization user-Intermediate user-Expert hidden">
								<a href="https://www.mindspore.cn/tutorial/training/zh-CN/master/advanced_use/mindinsight_commands.html" class="article-link">
									<div>
										<div class="doc-article-head">
											<span class="doc-head-content">MindInsight相关命令</span>
										</div>
										<div class="doc-article-desc">
											本教程介绍MindInsight提供的相关命令，包含启动服务、停止服务、导出Summary等功能。
										</div>
									</div>
								</a>
							</div>
							<div class="doc-article-item all os-Linux hardware-Ascend hardware-GPU hardware-CPU stage-Data-Preparation user-Intermediate user-Expert experience-Online-Experience experience-Local-Experience hidden">
								<a href="https://www.mindspore.cn/tutorial/training/zh-CN/master/advanced_use/enable_auto_augmentation.html" class="article-link">
									<div>
										<div class="doc-article-head">
											<span class="doc-head-content">应用自动数据增强</span>
										</div>
										<div class="doc-article-desc">
											自动数据增强（AutoAugment）是在一系列图像增强子策略的搜索空间中，通过搜索算法找到适合特定数据集的图像增强方案。本教程介绍如何在ImageNet数据集上应用自动数据增强。
										</div>
									</div>
								</a>
							</div>
							<div class="doc-article-item all os-Linux hardware-Ascend hardware-GPU hardware-CPU stage-Model-Export stage-Model-Training user-Beginner user-Intermediate user-Expert experience-Online-Experience experience-Local-Experience hidden">
								<a href="https://www.mindspore.cn/tutorial/training/zh-CN/master/advanced_use/evaluate_the_model_during_training.html" class="article-link">
									<div>
										<div class="doc-article-head">
											<span class="doc-head-content">训练时验证模型</span>
										</div>
										<div class="doc-article-desc">
											本教程使用MNIST数据集通过卷积神经网络LeNet5进行训练，着重介绍了在进行模型训练的同时进行模型的验证，保存对应epoch的模型，并从中挑选出最优模型的方法。
										</div>
									</div>
								</a>
							</div>
							<div class="doc-article-item all os-Linux hardware-Ascend stage-Model-Training user-Intermediate user-Expert hidden">
								<a href="https://www.mindspore.cn/tutorial/training/zh-CN/master/advanced_use/distributed_training_ascend.html" class="article-link">
									<div>
										<div class="doc-article-head">
											<span class="doc-head-content">分布式并行训练 (Ascend) </span>
										</div>
										<div class="doc-article-desc">
											本教程主要介绍如何在Ascend 910 AI处理器硬件平台上，利用MindSpore通过数据并行及自动并行模式训练ResNet-50网络。
										</div>
									</div>
								</a>
							</div>
							<div class="doc-article-item all os-Linux hardware-GPU stage-Model-Training user-Intermediate user-Expert hidden">
								<a href="https://www.mindspore.cn/tutorial/training/zh-CN/master/advanced_use/distributed_training_gpu.html" class="article-link">
									<div>
										<div class="doc-article-head">
											<span class="doc-head-content">分布式并行训练 (GPU) </span>
										</div>
										<div class="doc-article-desc">
											本教程主要介绍如何在GPU硬件平台上，利用MindSpore通过数据并行及自动并行模式训练ResNet-50网络。
										</div>
									</div>
								</a>
							</div>
							<div class="doc-article-item all os-Linux hardware-Ascend hardware-CPU stage-Model-Training user-Intermediate user-Expert hidden">
								<a href="https://www.mindspore.cn/tutorial/training/zh-CN/master/advanced_use/apply_host_device_training.html" class="article-link">
									<div>
										<div class="doc-article-head">
											<span class="doc-head-content">应用Host&Device混合训练 </span>
										</div>
										<div class="doc-article-desc">
											在深度学习中，工作人员时常会遇到超大模型的训练问题，即模型参数所占内存超过了设备内存上限。为高效地训练超大模型，一种可行的方案是使用主机端（Host）和加速器（Device）的混合训练模式。此方案同时发挥了主机端内存大和加速器端计算快的优势，是一种解决超大模型训练较有效的方式。本教程以推荐模型Wide&Deep为例，讲解MindSpore在主机和Ascend 910 AI处理器的混合训练。
										</div>
									</div>
								</a>
							</div>
							<div class="doc-article-item all os-Linux hardware-Ascend hardware-GPU stage-Model-Training user-Intermediate user-Expert hidden">
								<a href="https://www.mindspore.cn/tutorial/training/zh-CN/master/advanced_use/apply_parameter_server_training.html" class="article-link">
									<div>
										<div class="doc-article-head">
											<span class="doc-head-content">使用Parameter Server训练 </span>
										</div>
										<div class="doc-article-desc">
											Parameter Server(参数服务器)是分布式训练中一种广泛使用的架构，相较于同步的AllReduce训练方法，Parameter Server具有更好的灵活性、可扩展性以及节点容灾的能力。本教程以LeNet在Ascend 910上训练上训练为例，介绍如何使用Parameter Server。
										</div>
									</div>
								</a>
							</div>
							<div class="doc-article-item all os-Linux hardware-Ascend hardware-GPU stage-Model-Training user-Intermediate user-Expert hidden">
								<a href="https://www.mindspore.cn/tutorial/training/zh-CN/master/advanced_use/save_load_model_hybrid_parallel.html" class="article-link">
									<div>
										<div class="doc-article-head">
											<span class="doc-head-content">保存和加载模型（HyBrid Parallel模式）</span>
										</div>
										<div class="doc-article-desc">
											对于用户手动设置的并行场景（HyBrid Parallel），切分逻辑由用户自己实现，MindSpore在每个节点上保存相同的模型参数切分策略文件和本节点上的数据，用户需要自己实现CheckPoint文件的合并保存与加载功能。本教程用于指导用户在手动切分场景下，实现CheckPoint的合并保存与加载能力。
										</div>
									</div>
								</a>
							</div>
							<div class="doc-article-item all os-Linux hardware-Ascend hardware-GPU stage-Model-Training user-Intermediate user-Expert experience-Online-Experience experience-Local-Experience hidden">
								<a href="https://www.mindspore.cn/tutorial/training/zh-CN/master/advanced_use/enable_mixed_precision.html" class="article-link">
									<div>
										<div class="doc-article-head">
											<span class="doc-head-content">使能自动混合精度</span>
										</div>
										<div class="doc-article-desc">
											混合精度训练方法是通过混合使用单精度和半精度数据格式来加速深度神经网络训练的过程，同时保持了单精度训练所能达到的网络精度。混合精度训练能够加速计算过程，同时减少内存使用和存取，并使得在特定的硬件上可以训练更大的模型或batch size。
										</div>
									</div>
								</a>
							</div>
							<div class="doc-article-item all os-Linux hardware-Ascend hardware-GPU stage-Model-Optimization user-Intermediate user-Expert hidden">
								<a href="https://www.mindspore.cn/tutorial/training/zh-CN/master/advanced_use/enable_graph_kernel_fusion.html" class="article-link">
									<div>
										<div class="doc-article-head">
											<span class="doc-head-content">使能图算融合</span>
										</div>
										<div class="doc-article-desc">
											图算融合是MindSpore特有的网络性能优化技术。它可以通过自动分析和优化现有网络计算图逻辑，并结合目标硬件能力，对计算图进行计算化简和替代、算子拆分和融合、算子特例化编译等优化，以提升设备计算资源利用率，实现对网络性能的整体优化。
										</div>
									</div>
								</a>
							</div>
							<div class="doc-article-item all os-Linux hardware-GPU stage-Model-Optimization user-Intermediate user-Expert experience-Online-Experience experience-Local-Experience hidden">
								<a href="https://www.mindspore.cn/tutorial/training/zh-CN/master/advanced_use/apply_gradient_accumulation.html" class="article-link">
									<div>
										<div class="doc-article-head">
											<span class="doc-head-content">应用梯度累积算法</span>
										</div>
										<div class="doc-article-desc">
											本教程介绍梯度累积的训练方式，目的是为了解决由于内存不足导致某些大型网络无法训练大Batch_size的问题。
										</div>
									</div>
								</a>
							</div>
							<div class="doc-article-item all os-Linux hardware-Ascend hardware-GPU hardware-CPU stage-Data-Preparation user-Intermediate user-Expert hidden">
								<a href="https://www.mindspore.cn/tutorial/training/zh-CN/master/advanced_use/enable_cache.html" class="article-link">
									<div>
										<div class="doc-article-head">
											<span class="doc-head-content">应用单节点数据缓存</span>
										</div>
										<div class="doc-article-desc">
											本教程将演示如何使用单节点缓存服务来缓存经过数据增强处理的数据。
										</div>
									</div>
								</a>
							</div>
							<div class="doc-article-item all os-Linux hardware-Ascend hardware-GPU stage-Model-Optimization user-Expert hidden">
								<a href="https://www.mindspore.cn/tutorial/training/zh-CN/master/advanced_use/apply_quantization_aware_training.html" class="article-link">
									<div>
										<div class="doc-article-head">
											<span class="doc-head-content">应用感知量化训练</span>
										</div>
										<div class="doc-article-desc">
											MindSpore的感知量化训练是在训练基础上，使用低精度数据替换高精度数据来简化训练模型的过程。这个过程不可避免引入精度的损失，这时使用伪量化节点来模拟引入的精度损失，并通过反向传播学习，来减少精度损失。
										</div>
									</div>
								</a>
							</div>
							<div class="doc-article-item all os-Linux hardware-Ascend hardware-GPU hardware-CPU stage-Model-Training stage-Model-Optimization user-Enterprise user-Expert experience-Online-Experience experience-Local-Experience hidden">
								<a href="https://www.mindspore.cn/tutorial/training/zh-CN/master/advanced_use/improve_model_security_nad.html" class="article-link">
									<div>
										<div class="doc-article-head">
											<span class="doc-head-content">使用NAD算法提升模型安全性</span>
										</div>
										<div class="doc-article-desc">
											本教程介绍MindArmour提供的模型安全防护手段，引导您快速使用MindArmour，为您的AI模型提供一定的安全防护能力。
										</div>
									</div>
								</a>
							</div>
							<div class="doc-article-item all os-Linux hardware-Ascend stage-Model-Training stage-Model-Optimization user-Enterprise user-Expert hidden">
								<a href="https://www.mindspore.cn/tutorial/training/zh-CN/master/advanced_use/protect_user_privacy_with_differential_privacy.html" class="article-link">
									<div>
										<div class="doc-article-head">
											<span class="doc-head-content">应用差分隐私机制保护用户隐私</span>
										</div>
										<div class="doc-article-desc">
											机器学习算法一般是用大量数据并更新模型参数，学习数据特征。在理想情况下，这些算法学习到一些泛化性较好的模型，例如“吸烟患者更容易得肺癌”，而不是特定的个体特征，例如“张三是个吸烟者，患有肺癌”。然而，机器学习算法并不会区分通用特征还是个体特征。当我们用机器学习来完成某个重要的任务，例如肺癌诊断，发布的机器学习模型，可能在无意中透露训练集中的个体特征，恶意攻击者可能从发布的模型获得关于张三的隐私信息，因此使用差分隐私技术来保护机器学习模型是十分必要的。
										</div>
									</div>
								</a>
							</div>
							<div class="doc-article-item all os-Linux hardware-Ascend hardware-GPU hardware-CPU stage-Model-Evaluation user-Enterprise user-Expert hidden">
								<a href="https://www.mindspore.cn/tutorial/training/zh-CN/master/advanced_use/test_model_security_fuzzing.html" class="article-link">
									<div>
										<div class="doc-article-head">
											<span class="doc-head-content">使用fuzz testing模块测试模型安全性</span>
										</div>
										<div class="doc-article-desc">
											MindArmour的fuzz_testing模块以神经元覆盖率作为测试评价准则。我们通过神经元覆盖率来指导输入变异，让输入能够激活更多的神经元，神经元值的分布范围更广，从而探索不同类型的模型输出结果、错误行为。
										</div>
									</div>
								</a>
							</div>
							<div class="doc-article-item all os-Linux hardware-Ascend hardware-GPU hardware-CPU stage-Model-Evaluation user-Enterprise user-Expert hidden">
								<a href="https://www.mindspore.cn/tutorial/training/zh-CN/master/advanced_use/test_model_security_membership_inference.html" class="article-link">
									<div>
										<div class="doc-article-head">
											<span class="doc-head-content">使用成员推理测试模型安全性</span>
										</div>
										<div class="doc-article-desc">
											机器学习/深度学习的成员推理(MembershipInference)，指的是攻击者拥有模型的部分访问权限(黑盒、灰盒或白盒)，能够获取到模型的输出、结构或参数等部分或全部信息，并基于这些信息推断某个样本是否属于模型的训练集。利用成员推理，我们可以评估机器学习/深度学习模型的隐私数据安全。如果在成员推理下能正确识别出60%+的样本，那么我们认为该模型存在隐私数据泄露风险。
										</div>
									</div>
								</a>
							</div>
							<div class="doc-article-item all os-Linux hardware-Ascend hardware-GPU stage-Whole-Process user-Beginner user-Intermediate user-Expert experience-Online-Experience experience-Local-Experience hidden">
								<a href="https://www.mindspore.cn/tutorial/training/zh-CN/master/advanced_use/cv_resnet50.html" class="article-link">
									<div>
										<div class="doc-article-head">
											<span class="doc-head-content">使用ResNet-50网络实现图像分类</span>
										</div>
										<div class="doc-article-desc">
											计算机视觉是当前深度学习研究最广泛、落地最成熟的技术领域，在手机拍照、智能安防、自动驾驶等场景有广泛应用。本教程结合图像分类任务，介绍MindSpore如何应用于计算机视觉场景。
										</div>
									</div>
								</a>
							</div>
							<div class="doc-article-item all os-Linux hardware-Ascend hardware-GPU stage-Model-Development stage-Model-Optimization user-Expert hidden">
								<a href="https://www.mindspore.cn/tutorial/training/zh-CN/master/advanced_use/cv_resnet50_second_order_optimizer.html" class="article-link">
									<div>
										<div class="doc-article-head">
											<span class="doc-head-content">在ResNet-50网络上应用二阶优化实践</span>
										</div>
										<div class="doc-article-desc">
											本篇教程将主要介绍如何在hardware-Ascend 910以及GPU上，使用MindSpore提供的二阶优化器THOR训练ResNet50-v1.5网络和ImageNet数据集。
										</div>
									</div>
								</a>
							</div>
							<div class="doc-article-item all os-Windows os-Linux hardware-CPU hardware-Ascend hardware-GPU stage-Model-Development user-Intermediate user-Expert hidden">
								<a href="https://www.mindspore.cn/tutorial/training/zh-CN/master/advanced_use/cv_mobilenetv2_fine_tune.html" class="article-link">
									<div>
										<div class="doc-article-head">
											<span class="doc-head-content">使用MobileNetV2网络实现微调（Fine Tune）</span>
										</div>
										<div class="doc-article-desc">
											本教程将会介绍如何在不同系统与处理器下的MindSpore框架中做微调的训练与验证。
										</div>
									</div>
								</a>
							</div>
							<div class="doc-article-item all os-Linux hardware-GPU hardware-CPU stage-Whole-Process user-Beginner user-Intermediate user-Expert experience-Online-Experience experience-Local-Experience hidden">
								<a href="https://www.mindspore.cn/tutorial/training/zh-CN/master/advanced_use/nlp_sentimentnet.html" class="article-link">
									<div>
										<div class="doc-article-head">
											<span class="doc-head-content">使用SentimentNet实现情感分类</span>
										</div>
										<div class="doc-article-desc">
											情感分类是自然语言处理中文本分类问题的子集，属于自然语言处理最基础的应用。它是对带有感情色彩的主观性文本进行分析和推理的过程，即分析说话人的态度，是倾向正面还是反面。
										</div>
									</div>
								</a>
							</div>
							<div class="doc-article-item all os-Linux hardware-Ascend stage-Model-Training stage-Inference user-Expert hidden">
								<a href="https://www.mindspore.cn/tutorial/training/zh-CN/master/advanced_use/nlp_bert_poetry.html" class="article-link">
									<div>
										<div class="doc-article-head">
											<span class="doc-head-content">使用BERT网络实现智能写诗</span>
										</div>
										<div class="doc-article-desc">
											本教程介绍通过MindSpore训练出智能写诗模型及部署预测服务。
										</div>
									</div>
								</a>
							</div>
							<div class="doc-article-item all os-Linux hardware-GPU stage-Model-Development user-Expert hidden">
								<a href="https://www.mindspore.cn/tutorial/training/zh-CN/master/advanced_use/hpc_gomo.html" class="article-link">
									<div>
										<div class="doc-article-head">
											<span class="doc-head-content">实现区域海洋模型GOMO</span>
										</div>
										<div class="doc-article-desc">
											本篇教程将主要介绍如何在GPU上，使用MindSpore构建并运行三维海洋模型GOMO。
										</div>
									</div>
								</a>
							</div>
							<div class="doc-article-item all os-Linux hardware-Ascend stage-Whole-Process user-Beginner user-Intermediate user-Expert hidden">
								<a href="https://www.mindspore.cn/tutorial/training/zh-CN/master/advanced_use/use_on_the_cloud.html" class="article-link">
									<div>
										<div class="doc-article-head">
											<span class="doc-head-content">在云上使用MindSpore</span>
										</div>
										<div class="doc-article-desc">
											ModelArts是华为云提供的面向开发者的一站式AI开发平台，集成了昇腾AI处理器资源池，用户可以在该平台下体验MindSpore。本教程以ResNet-50为例，简要介绍如何在ModelArts使用MindSpore完成训练任务。
										</div>
									</div>
								</a>
							</div>
						</div>
				
					</div>
					
				</div>
			</div>
		</div>
		
