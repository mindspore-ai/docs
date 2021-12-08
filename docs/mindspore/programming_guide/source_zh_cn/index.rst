.. MindSpore documentation master file, created by
   sphinx-quickstart on Thu Mar 24 11:00:00 2020.
   You can adapt this file completely to your liking, but it should at least
   contain the root `toctree` directive.

MindSpore编程指南
===================

.. toctree::
   :glob:
   :maxdepth: 1
   :caption: 整体介绍
   :hidden:

   architecture
   api_structure

.. toctree::
   :glob:
   :maxdepth: 1
   :caption: 设计介绍
   :hidden:

   design/technical_white_paper
   design/all_scenarios_architecture
   design/gradient
   design/dynamic_graph_and_static_graph
   design/distributed_training_design
   design/heterogeneous_training
   design/mindir
   design/data_engine
   可视化调试调优↗ <https://www.mindspore.cn/mindinsight/docs/zh-CN/master/training_visual_design.html>
   安全可信↗ <https://www.mindspore.cn/mindarmour/docs/zh-CN/master/design.html>
   design/glossary

.. toctree::
   :glob:
   :maxdepth: 1
   :caption: 快速入门
   :hidden:

   实现简单线性函数拟合↗ <https://www.mindspore.cn/tutorials/zh-CN/master/linear_regression.html> 
   实现一个图片分类应用↗ <https://www.mindspore.cn/tutorials/zh-CN/master/quick_start.html>

.. toctree::
   :glob:
   :maxdepth: 1
   :caption: 基本概念
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
   :caption: 数据加载和处理
   :hidden:

   dataset_sample
   dataset
   pipeline
   dataset_advanced
   dataset_usage

.. toctree::
   :glob:
   :maxdepth: 1
   :caption: 网络构建
   :hidden:

   build_net
   initializer
   parameter
   control_flow
   indefinite_parameter
   constexpr
   loss
   grad_operation
   hypermap
   optim
   train_and_eval

.. toctree::
   :glob:
   :maxdepth: 1
   :caption: 模型运行
   :hidden:

   context
   run
   ms_function
   save_and_load_models
   model

.. toctree::
   :glob:
   :maxdepth: 1
   :caption: 推理
   :hidden:

   multi_platform_inference
   online_inference
   offline_inference

.. toctree::
   :glob:
   :maxdepth: 1
   :caption: 分布式并行
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
   :caption: 功能调试
   :hidden:

   read_ir_files
   使用PyNative模式调试↗ <https://www.mindspore.cn/docs/programming_guide/zh-CN/master/debug_in_pynative_mode.html>
   dump_in_graph_mode
   custom_debugging_info
   incremental_operator_build
   fixing_randomness

.. toctree::
   :glob:
   :maxdepth: 1
   :caption: 精度调优
   :hidden:

   精度问题初步定位指南↗ <https://www.mindspore.cn/mindinsight/docs/zh-CN/master/accuracy_problem_preliminary_location.html>
   精度问题详细定位和调优指南↗ <https://www.mindspore.cn/mindinsight/docs/zh-CN/master/accuracy_optimization.html>

.. toctree::
   :glob:
   :maxdepth: 1
   :caption: 性能优化
   :hidden:

   enable_mixed_precision
   enable_auto_tune
   enable_dataset_autotune
   enable_dataset_offload
   apply_gradient_accumulation
   使用Profiler调试性能↗ <https://www.mindspore.cn/mindinsight/docs/zh-CN/master/performance_profiling.html>

.. toctree::
   :glob:
   :maxdepth: 1
   :caption: 高级特性
   :hidden:

   second_order_optimizer
   graph_kernel_fusion
   apply_quantization_aware_training

.. toctree::
   :glob:
   :maxdepth: 1
   :caption: 应用实践
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
							<span class="doc-filter">筛选条件</span>
							<button class="doc-delete doc-btn" id="all">清除所有条件</button>
						</div>
					
						<div class="doc-label-content">
							<div class="doc-label-choice">
								<div class="row">
								<div class="col-sm-2">
									<div class="doc-hardware">硬件</div>
								</div>
								<div class="col-sm-10 col-sm-pull-1">
									<button class="doc-filter-btn doc-btn" id="hardware-ascend">Ascend</button>
									<button class="doc-filter-btn doc-btn" id="hardware-gpu">GPU</button>
									<button class="doc-filter-btn doc-btn" id="hardware-cpu">CPU</button>
									<button class="doc-filter-btn doc-btn" id="hardware-device">端侧</button>
								</div>
							</div>
							</div>
							
							<div class="doc-label-choice">
								<div class="row">
									<div class="col-sm-2 ">
										<div class="doc-stage">分类</div>
									</div>							
									<div class="col-sm-10 col-sm-pull-1 doc-stage-detail">
										<button class="doc-filter-btn doc-btn" id="stage-Beginner">入门</button>
                              			<button class="doc-filter-btn doc-btn" id="stage-Whole-Process">全流程</button>
										<button class="doc-filter-btn doc-btn" id="stage-Data-Preparation">数据准备</button>
										<button class="doc-filter-btn doc-btn" id="stage-Model-Development">模型开发</button>
                              			<button class="doc-filter-btn doc-btn" id="stage-Model-Running">模型运行</button>
                              			<button class="doc-filter-btn doc-btn" id="stage-Model-Optimization">模型调优</button>
										<button class="doc-filter-btn doc-btn" id="stage-Model-Export">模型导出</button>
                              			<button class="doc-filter-btn doc-btn" id="stage-Inference">推理应用</button>
										<button class="doc-filter-btn doc-btn" id="stage-Model-Evaluation">模型评估</button>
										<button class="doc-filter-btn doc-btn" id="stage-Model-Loading">模型加载</button>
										<button class="doc-filter-btn doc-btn" id="stage-Distributed-Parallel">分布式并行</button>
										<button class="doc-filter-btn doc-btn" id="stage-Function-Extension">扩展功能</button>
										<button class="doc-filter-btn doc-btn" id="stage-Design">设计</button>
									</div>
									
								</div>
							</div>							
							<div class="doc-label-choice">
								<div class="row">
									<div class="col-sm-2 ">
										<div class="doc-experience">体验</div>
									</div>							
									<div class="col-sm-10 col-sm-pull-1 doc-stage-detail">
										<button class="doc-filter-btn doc-btn" id="experience-online">在线体验</button>
										<button class="doc-filter-btn doc-btn" id="experience-local">本地体验</button>
									</div>
									
								</div>
								
							</div>
							
						</div>
						
						<font size="2">注意：点击带“↗”的标题，将会离开编程指南页面。</font>

						<hr>

						<div class="doc-footer">
							<nav aria-label="Page navigation">
								<ul class="pagination" id="pageNav">
									
								</ul>
							</nav>
						</div>

						<div class="doc-article-list">
						   <div class="doc-article-item all hardware-ascend hardware-gpu hardware-cpu stage-Beginner experience-local experience-local experience-online hidden">
								<a href="https://www.mindspore.cn/docs/programming_guide/zh-CN/master/api_structure.html" class="article-link">
									<div>
										<div class="doc-article-head">
											<span class="doc-head-content">MindSpore API概述</span>
										</div>
										<div class="doc-article-desc">
										MindSpore是一个全场景深度学习框架，旨在实现易开发、高效执行、全场景覆盖三大目标，其中易开发表现为API友好、调试难度低，高效执行包括计算效率、数据预处理效率和分布式训练效率，全场景则指框架同时支持云、边缘以及端侧场景。
										</div>
									</div>
								</a>
							</div>
							<div class="doc-article-item all hardware-gpu stage-Model-Optimization experience-local experience-local experience-online hidden">
								<a href="https://www.mindspore.cn/docs/programming_guide/zh-CN/master/apply_gradient_accumulation.html" class="article-link">
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
							<div class="doc-article-item all hardware-ascend hardware-gpu stage-Distributed-Parallel stage-Whole-Process experience-local hidden">
								<a href="https://www.mindspore.cn/docs/programming_guide/zh-CN/master/apply_host_device_training.html" class="article-link">
									<div>
										<div class="doc-article-head">
											<span class="doc-head-content">Host&Device异构</span>
										</div>
										<div class="doc-article-desc">
										在深度学习中，为高效地训练超大模型，可以使用主机端（Host）和加速器（Device）的混合训练模式。此方案同时发挥了主机端内存大和加速器端计算快的优势，是一种解决超大模型训练较有效的方式。
										</div>
									</div>
								</a>
							</div>
							<div class="doc-article-item all hardware-ascend hardware-gpu stage-Distributed-Parallel stage-Whole-Process experience-local hidden">
								<a href="https://www.mindspore.cn/docs/programming_guide/zh-CN/master/apply_parameter_server_training.html" class="article-link">
									<div>
										<div class="doc-article-head">
											<span class="doc-head-content">Parameter Server模式</span>
										</div>
										<div class="doc-article-desc">
										Parameter Server(参数服务器)是分布式训练中一种广泛使用的架构，相较于同步的AllReduce训练方法，Parameter Server具有更好的灵活性、可扩展性以及节点容灾的能力。
										</div>
									</div>
								</a>
							</div>
							<div class="doc-article-item all hardware-ascend stage-Distributed-Parallel stage-Whole-Process experience-local hidden">
								<a href="https://www.mindspore.cn/docs/programming_guide/zh-CN/master/apply_pipeline_parallel.html" class="article-link">
									<div>
										<div class="doc-article-head">
											<span class="doc-head-content">流水线并行</span>
										</div>
										<div class="doc-article-desc">
										流水线并行能够将模型在空间上按 <em>stage</em> 进行切分，每个 <em>stage</em> 只需执行网络的一部分，大大节省了内存开销，同时缩小了通信域，缩短了通信时间。MindSpore能够根据用户的配置，将单机模型自动地转换成流水线并行模式去执行。
										</div>
									</div>
								</a>
							</div>
							<div class="doc-article-item all hardware-ascend hardware-device stage-Inference experience-local hidden">
								<a href="https://www.mindspore.cn/docs/programming_guide/zh-CN/master/apply_post_training_quantization.html" class="article-link">
									<div>
										<div class="doc-article-head">
											<span class="doc-head-content">应用训练后量化</span>
										</div>
										<div class="doc-article-desc">
										训练后量化是指对预训练后的网络进行权重量化或者全量化，以达到减小模型大小和提升推理性能的目的。
										</div>
									</div>
								</a>
							</div>
							<div class="doc-article-item all hardware-ascend hardware-gpu stage-Function-Extension experience-local experience-online hidden">
								<a href="https://www.mindspore.cn/docs/programming_guide/zh-CN/master/apply_quantization_aware_training.html" class="article-link">
									<div>
										<div class="doc-article-head">
											<span class="doc-head-content">应用感知量化训练</span>
										</div>
										<div class="doc-article-desc">
										量化（Quantization）技术是应对移动设备、边缘设备的硬件资源有限的问题衍生出的技术之一。
										</div>
									</div>
								</a>
							</div>
							<div class="doc-article-item all hardware-ascend hardware-gpu hardware-cpu hardware-device stage-Beginner experience-local hidden">
								<a href="https://www.mindspore.cn/docs/programming_guide/zh-CN/master/architecture.html" class="article-link">
									<div>
										<div class="doc-article-head">
											<span class="doc-head-content">MindSpore总体架构</span>
										</div>
										<div class="doc-article-desc">
										昇思MindSpore是一个全场景深度学习框架，旨在实现易开发、高效执行、全场景覆盖三大目标，其中易开发表现为API友好、调试难度低，高效执行包括计算效率、数据预处理效率和分布式训练效率，全场景则指框架同时支持云、边缘以及端侧场景。
										</div>
									</div>
								</a>
							</div>
							<div class="doc-article-item all hardware-ascend hardware-gpu hardware-cpu stage-Data-Preparation experience-local experience-online hidden">
								<a href="https://www.mindspore.cn/docs/programming_guide/zh-CN/master/augmentation.html" class="article-link">
									<div>
										<div class="doc-article-head">
											<span class="doc-head-content">图像处理与增强</span>
										</div>
										<div class="doc-article-desc">
										在计算机视觉任务中，数据量过小或是样本场景单一等问题都会影响模型的训练效果，用户可以通过数据增强操作对图像进行预处理，从而提升模型的泛化性。
										</div>
									</div>
								</a>
							</div>
							<div class="doc-article-item all hardware-ascend hardware-gpu hardware-cpu stage-Data-Preparation experience-local experience-online hidden">
								<a href="https://www.mindspore.cn/docs/programming_guide/zh-CN/master/auto_augmentation.html" class="article-link">
									<div>
										<div class="doc-article-head">
											<span class="doc-head-content">自动数据增强</span>
										</div>
										<div class="doc-article-desc">
										MindSpore除了可以让用户自定义数据增强的使用，还提供了一种自动数据增强方式，可以基于特定策略自动对图像进行数据增强处理。
										</div>
									</div>
								</a>
							</div>
							<div class="doc-article-item all hardware-ascend hardware-gpu stage-Distributed-Parallel experience-local hidden">
								<a href="https://www.mindspore.cn/docs/programming_guide/zh-CN/master/auto_parallel.html" class="article-link">
									<div>
										<div class="doc-article-head">
											<span class="doc-head-content">分布式并行接口说明</span>
										</div>
										<div class="doc-article-desc">
										分布式并行训练，可以降低对内存、计算性能等硬件的需求，是进行训练的重要优化手段。
										</div>
									</div>
								</a>
							</div>
							<div class="doc-article-item all hardware-ascend hardware-gpu hardware-cpu stage-Model-Development experience-local hidden">
								<a href="https://www.mindspore.cn/docs/programming_guide/zh-CN/master/build_net.html" class="article-link">
									<div>
										<div class="doc-article-head">
											<span class="doc-head-content">构建单算子网络和多层网络</span>
										</div>
										<div class="doc-article-desc">
										MindSpore的Cell类是构建所有网络的基类，也是网络的基本单元。MindSpore的ops模块提供了基础算子的实现，nn模块实现了对基础算子的进一步封装，用户可以根据需要，灵活使用不同的算子。
										</div>
									</div>
								</a>
							</div>
							<div class="doc-article-item all hardware-ascend hardware-gpu hardware-cpu stage-Data-Preparation experience-local experience-online hidden">
								<a href="https://www.mindspore.cn/docs/programming_guide/zh-CN/master/cache.html" class="article-link">
									<div>
										<div class="doc-article-head">
											<span class="doc-head-content">单节点数据缓存</span>
										</div>
										<div class="doc-article-desc">
										对于需要重复访问远程的数据集或需要重复从磁盘中读取数据集的情况，可以使用单节点缓存算子将数据集缓存于本地内存中，以加速数据集的读取。
										</div>
									</div>
								</a>
							</div>
							<div class="doc-article-item all hardware-ascend hardware-gpu hardware-cpu stage-Model-Development experience-local hidden">
								<a href="https://www.mindspore.cn/docs/programming_guide/zh-CN/master/callback.html" class="article-link">
									<div>
										<div class="doc-article-head">
											<span class="doc-head-content">Callback机制</span>
										</div>
										<div class="doc-article-desc">
										Callback机制让用户可以及时有效地掌握网络模型的训练状态，并根据需要随时作出调整，可以极大地提升用户的开发效率。
										</div>
									</div>
								</a>
							</div>
							<div class="doc-article-item all hardware-ascend hardware-gpu hardware-cpu stage-Beginner experience-local experience-online hidden">
								<a href="https://www.mindspore.cn/docs/programming_guide/zh-CN/master/cell.html" class="article-link">
									<div>
										<div class="doc-article-head">
											<span class="doc-head-content">Cell</span>
										</div>
										<div class="doc-article-desc">
										MindSpore的 <em>Cell</em> 类是构建所有网络的基类，也是网络的基本单元。
										</div>
									</div>
								</a>
							</div>
							<div class="doc-article-item all hardware-ascend hardware-gpu hardware-cpu stage-Model-Development experience-local hidden">
								<a href="https://www.mindspore.cn/docs/programming_guide/zh-CN/master/constexpr.html" class="article-link">
									<div>
										<div class="doc-article-head">
											<span class="doc-head-content">网络内构造常量</span>
										</div>
										<div class="doc-article-desc">
										 <em>mindspore.ops.constexpr</em>中提供了一个@constexpr的Python 装饰器，该装饰器可以用于修饰一个函数，该函数在编译阶段将会通过Python解释器执行，最终在MindSpore的类型推导阶段被常量折叠成为ANF图的一个常量节点(ValueNode)。
										</div>
									</div>
								</a>
							</div>
							<div class="doc-article-item all hardware-ascend hardware-gpu hardware-cpu stage-Model-Running experience-local experience-online hidden">
								<a href="https://www.mindspore.cn/docs/programming_guide/zh-CN/master/context.html" class="article-link">
									<div>
										<div class="doc-article-head">
											<span class="doc-head-content">配置运行信息</span>
										</div>
										<div class="doc-article-desc">
										初始化网络之前要配置context参数，用于控制程序执行的策略。比如选择执行模式、选择执行后端、配置分布式相关参数等。按照context参数设置实现的不同功能，可以将其分为执行模式管理、硬件管理、分布式管理和维测管理等。
										</div>
									</div>
								</a>
							</div>
							<div class="doc-article-item all hardware-ascend hardware-gpu hardware-cpu stage-Model-Development experience-local hidden">
								<a href="https://www.mindspore.cn/docs/programming_guide/zh-CN/master/control_flow.html" class="article-link">
									<div>
										<div class="doc-article-head">
											<span class="doc-head-content">使用流程控制语句</span>
										</div>
										<div class="doc-article-desc">
										MindSpore流程控制语句的使用与Python原生语法相似，尤其是在 <em>PYNATIVE_MODE</em> 模式下，与Python原生语法基本一致，但是在 <em>GRAPH_MODE</em> 模式下，会有一些特殊的约束。
										</div>
									</div>
								</a>
							</div>
							<div class="doc-article-item all hardware-ascend hardware-gpu hardware-cpu stage-Data-Preparation experience-local experience-online hidden">
								<a href="https://www.mindspore.cn/docs/programming_guide/zh-CN/master/convert_dataset.html" class="article-link">
									<div>
										<div class="doc-article-head">
											<span class="doc-head-content">转换数据集为MindRecord</span>
										</div>
										<div class="doc-article-desc">
										用户可以将非标准的数据集和常用的数据集转换为MindSpore数据格式，即MindRecord，从而方便地加载到MindSpore中进行训练。
										</div>
									</div>
								</a>
							</div>
							<div class="doc-article-item all hardware-ascend hardware-gpu hardware-cpu stage-Model-Optimization experience-local experience-online hidden">
								<a href="https://www.mindspore.cn/docs/programming_guide/zh-CN/master/custom_debugging_info.html" class="article-link">
									<div>
										<div class="doc-article-head">
											<span class="doc-head-content">自定义调试信息</span>
										</div>
										<div class="doc-article-desc">
										本文介绍如何使用MindSpore提供的 <em>Callback</em> 、 <em>metrics</em> 、 <em>Print</em> 算子、日志打印等自定义能力，帮助用户快速调试训练网络。
										</div>
									</div>
								</a>
							</div>
							<div class="doc-article-item all hardware-ascend stage-Model-Development experience-local hidden">
								<a href="https://www.mindspore.cn/docs/programming_guide/zh-CN/master/custom_operator_ascend.html" class="article-link">
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
							<div class="doc-article-item all hardware-cpu stage-Model-Development experience-local hidden">
								<a href="https://www.mindspore.cn/docs/programming_guide/zh-CN/master/custom_operator_cpu.html" class="article-link">
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
							<div class="doc-article-item all hardware-gpu stage-Model-Development experience-local hidden">
								<a href="https://www.mindspore.cn/docs/programming_guide/zh-CN/master/custom_operator_gpu.html" class="article-link">
									<div>
										<div class="doc-article-head">
											<span class="doc-head-content">自定义算子（GPU）</span>
										</div>
										<div class="doc-article-desc">
										算子是构建神经网络的基本要素，当开发网络遇到内置算子无法满足要求时,你可以利用MindSpore方便地实现一个GPU算子。
										</div>
									</div>
								</a>
							</div>
							<div class="doc-article-item all hardware-ascend hardware-gpu hardware-cpu stage-Whole-Process experience-local hidden">
								<a href="https://www.mindspore.cn/docs/programming_guide/zh-CN/master/cv_mobilenetv2_fine_tune.html" class="article-link">
									<div>
										<div class="doc-article-head">
											<span class="doc-head-content">使用MobileNetV2网络实现微调（Fine Tune）</span>
										</div>
										<div class="doc-article-desc">
										计算机视觉任务中，大部分任务都会选择预训练模型，在其上做微调（也称为Fine Tune）。
										</div>
									</div>
								</a>
							</div>
							<div class="doc-article-item all hardware-ascend hardware-gpu stage-Whole-Process experience-local experience-online hidden">
								<a href="https://www.mindspore.cn/docs/programming_guide/zh-CN/master/cv_resnet50.html" class="article-link">
									<div>
										<div class="doc-article-head">
											<span class="doc-head-content">使用ResNet-50网络实现图像分类</span>
										</div>
										<div class="doc-article-desc">
										深度神经网络可以逐层提取图像特征，并保持局部不变性，被广泛应用于分类、检测、分割、检索、识别、提升、重建等视觉任务中。
										</div>
									</div>
								</a>
							</div>
							<div class="doc-article-item all hardware-ascend hardware-gpu stage-Whole-Process stage-Function-Extension experience-local hidden">
								<a href="https://www.mindspore.cn/docs/programming_guide/zh-CN/master/cv_resnet50_second_order_optimizer.html" class="article-link">
									<div>
										<div class="doc-article-head">
											<span class="doc-head-content">在ResNet-50网络上应用二阶优化实践</span>
										</div>
										<div class="doc-article-desc">
										常见的优化算法可分为一阶优化算法和二阶优化算法。
										</div>
									</div>
								</a>
							</div>
							<div class="doc-article-item all hardware-ascend hardware-gpu hardware-cpu stage-Data-Preparation experience-local experience-online hidden">
								<a href="https://www.mindspore.cn/docs/programming_guide/zh-CN/master/dataset_conversion.html" class="article-link">
									<div>
										<div class="doc-article-head">
											<span class="doc-head-content">MindSpore数据格式转换</span>
										</div>
										<div class="doc-article-desc">
										用户可以将非标准的数据集和常用的数据集转换为MindSpore数据格式，即MindRecord，从而方便地加载到MindSpore中进行训练。
										</div>
									</div>
								</a>
							</div>
							<div class="doc-article-item all hardware-ascend hardware-gpu hardware-cpu stage-Beginner experience-local hidden">
								<a href="https://www.mindspore.cn/docs/programming_guide/zh-CN/master/dataset_introduction.html" class="article-link">
									<div>
										<div class="doc-article-head">
											<span class="doc-head-content">Dataset</span>
										</div>
										<div class="doc-article-desc">
										数据是深度学习的基础，高质量数据输入会在整个深度神经网络中起到积极作用。
										</div>
									</div>
								</a>
							</div>
							<div class="doc-article-item all hardware-ascend hardware-gpu hardware-cpu stage-Data-Preparation experience-local experience-online hidden">
								<a href="https://www.mindspore.cn/docs/programming_guide/zh-CN/master/dataset_loading.html" class="article-link">
									<div>
										<div class="doc-article-head">
											<span class="doc-head-content">数据集加载总览</span>
										</div>
										<div class="doc-article-desc">
										MindSpore支持加载图像领域常用的数据集，用户可以直接使用 <em>mindspore.dataset</em> 中对应的类实现数据集的加载。
										</div>
									</div>
								</a>
							</div>
							<div class="doc-article-item all hardware-ascend hardware-gpu hardware-cpu stage-Data-Preparation experience-local hidden">
								<a href="https://www.mindspore.cn/docs/programming_guide/zh-CN/master/dataset_usage.html" class="article-link">
									<div>
										<div class="doc-article-head">
											<span class="doc-head-content">数据迭代</span>
										</div>
										<div class="doc-article-desc">
										原始数据集通过数据集加载接口读取到内存，再通过数据增强操作进行数据变换，得到的数据集对象有两种常规的数据迭代方法。
										</div>
									</div>
								</a>
							</div>
							<div class="doc-article-item all hardware-ascend hardware-gpu hardware-cpu stage-Model-Running experience-local experience-online hidden">
								<a href="https://www.mindspore.cn/docs/programming_guide/zh-CN/master/debug_in_pynative_mode.html" class="article-link">
									<div>
										<div class="doc-article-head">
											<span class="doc-head-content">PyNative模式应用</span>
										</div>
										<div class="doc-article-desc">
										MindSpore支持两种运行模式，在调试或者运行方面做了不同的优化。
										</div>
									</div>
								</a>
							</div>
							<div class="doc-article-item all hardware-ascend stage-Inference experience-local hidden">
								<a href="https://www.mindspore.cn/docs/programming_guide/zh-CN/master/distributed_inference.html" class="article-link">
									<div>
										<div class="doc-article-head">
											<span class="doc-head-content">分布式推理</span>
										</div>
										<div class="doc-article-desc">
										分布式推理是指推理阶段采用多卡进行推理。
										</div>
									</div>
								</a>
							</div>
							<div class="doc-article-item all hardware-ascend stage-Distributed-Parallel stage-Whole-Process experience-local hidden">
								<a href="https://www.mindspore.cn/docs/programming_guide/zh-CN/master/distributed_training_ascend.html" class="article-link">
									<div>
										<div class="doc-article-head">
											<span class="doc-head-content">分布式并行训练基础样例（Ascend）</span>
										</div>
										<div class="doc-article-desc">
										本篇教程我们主要讲解，如何在Ascend 910 AI处理器硬件平台上，利用MindSpore通过数据并行及自动并行模式训练ResNet-50网络。
										</div>
									</div>
								</a>
							</div>
							<div class="doc-article-item all hardware-ascend stage-Distributed-Parallel experience-local hidden">
								<a href="https://www.mindspore.cn/docs/programming_guide/zh-CN/master/distributed_training_dataset_slice.html" class="article-link">
									<div>
										<div class="doc-article-head">
											<span class="doc-head-content">大幅面数据集切分</span>
										</div>
										<div class="doc-article-desc">
										在进行分布式训练时，以图片数据为例，当单张图片的大小过大时需要对图片进行切分。
										</div>
									</div>
								</a>
							</div>
							<div class="doc-article-item all hardware-gpu stage-Distributed-Parallel stage-Whole-Process experience-local hidden">
								<a href="https://www.mindspore.cn/docs/programming_guide/zh-CN/master/distributed_training_gpu.html" class="article-link">
									<div>
										<div class="doc-article-head">
											<span class="doc-head-content">分布式并行训练基础样例（GPU）</span>
										</div>
										<div class="doc-article-desc">
										本篇教程我们主要讲解，如何在GPU硬件平台上，利用MindSpore的数据并行及自动并行模式训练ResNet-50网络。
										</div>
									</div>
								</a>
							</div>
							<div class="doc-article-item all hardware-ascend hardware-gpu stage-Distributed-Parallel experience-local hidden">
								<a href="https://www.mindspore.cn/docs/programming_guide/zh-CN/master/distributed_training_mode.html" class="article-link">
									<div>
										<div class="doc-article-head">
											<span class="doc-head-content">分布式并行训练模式</span>
										</div>
										<div class="doc-article-desc">
										目前MindSpore支持四种并行模式，本教程将会详细介绍。
										</div>
									</div>
								</a>
							</div>
							<div class="doc-article-item all hardware-ascend hardware-gpu stage-Distributed-Parallel experience-local hidden">
								<a href="https://www.mindspore.cn/docs/programming_guide/zh-CN/master/distributed_training_ops.html" class="article-link">
									<div>
										<div class="doc-article-head">
											<span class="doc-head-content">分布式集合通信原语</span>
										</div>
										<div class="doc-article-desc">
										在分布式训练中涉及例如 <em>AllReduce</em> 、 <em>ReduceScatter</em> 、 <em>AllGather</em> 和 <em>Broadcast</em> 等通信操作进行数据传输，我们将在本章节分别阐述其含义和示例代码。
										</div>
									</div>
								</a>
							</div>
							<div class="doc-article-item all hardware-ascend stage-Distributed-Parallel experience-local hidden">
								<a href="https://www.mindspore.cn/docs/programming_guide/zh-CN/master/distributed_training_parallel_opt.html" class="article-link">
									<div>
										<div class="doc-article-head">
											<span class="doc-head-content">优化器并行</span>
										</div>
										<div class="doc-article-desc">
										在进行数据并行训练时，优化器并行通过将优化器的计算量分散到数据并行维度的卡上，在大规模网络上（比如Bert、GPT）可以有效减少内存消耗并提升网络性能。
										</div>
									</div>
								</a>
							</div>
							<div class="doc-article-item all hardware-ascend stage-Distributed-Parallel experience-local hidden">
								<a href="https://www.mindspore.cn/docs/programming_guide/zh-CN/master/distributed_training_transformer.html" class="article-link">
									<div>
										<div class="doc-article-head">
											<span class="doc-head-content">分布式并行训练Transformer模型</span>
										</div>
										<div class="doc-article-desc">
										MindSpore提供了一个分布式的Transformer接口 <em>mindspore.parallel.nn.transformer</em> ，将Transformer内部用到的每个算子都配置了并行策略，而用户只需要配置全局的 <em>data_parallel</em> 和 <em>model_parallel</em> 属性，即可完成分布式并行策略的配置。
										</div>
									</div>
								</a>
							</div>
							<div class="doc-article-item all hardware-ascend hardware-gpu hardware-cpu stage-Beginner experience-local experience-online hidden">
								<a href="https://www.mindspore.cn/docs/programming_guide/zh-CN/master/dtype.html" class="article-link">
									<div>
										<div class="doc-article-head">
											<span class="doc-head-content">DataType</span>
										</div>
										<div class="doc-article-desc">
										MindSpore张量支持不同的数据类型，与NumPy的数据类型一一对应。
										</div>
									</div>
								</a>
							</div>
							<div class="doc-article-item all hardware-ascend hardware-gpu hardware-cpu stage-Model-Optimization experience-local hidden">
								<a href="https://www.mindspore.cn/docs/programming_guide/zh-CN/master/dump_in_graph_mode.html" class="article-link">
									<div>
										<div class="doc-article-head">
											<span class="doc-head-content">使用Dump功能在Graph模式调试</span>
										</div>
										<div class="doc-article-desc">
										为了对训练过程进行分析，用户需要感知训练过程中算子的输入和输出数据。
										</div>
									</div>
								</a>
							</div>
							<div class="doc-article-item all hardware-ascend hardware-gpu hardware-cpu stage-Data-Preparation experience-local experience-online hidden">
								<a href="https://www.mindspore.cn/docs/programming_guide/zh-CN/master/eager.html" class="article-link">
									<div>
										<div class="doc-article-head">
											<span class="doc-head-content">数据处理的Eager模式</span>
										</div>
										<div class="doc-article-desc">
										在资源条件允许的情况下，为了追求更高的性能，一般使用数据管道模式执行数据增强算子。
										</div>
									</div>
								</a>
							</div>
							<div class="doc-article-item all hardware-ascend hardware-gpu hardware-cpu stage-Data-Preparation experience-local experience-online hidden">
								<a href="https://www.mindspore.cn/docs/programming_guide/zh-CN/master/enable_auto_augmentation.html" class="article-link">
									<div>
										<div class="doc-article-head">
											<span class="doc-head-content">应用自动数据增强</span>
										</div>
										<div class="doc-article-desc">
										自动数据增强（AutoAugment）是在一系列图像增强子策略的搜索空间中，通过搜索算法找到适合特定数据集的图像增强方案。
										</div>
									</div>
								</a>
							</div>
							<div class="doc-article-item all hardware-ascend stage-Model-Optimization experience-local hidden">
								<a href="https://www.mindspore.cn/docs/programming_guide/zh-CN/master/enable_auto_tune.html" class="article-link">
									<div>
										<div class="doc-article-head">
											<span class="doc-head-content">使能算子调优工具</span>
										</div>
										<div class="doc-article-desc">
										。本文档主要介绍AutoTune的在线调优使用方法。
										</div>
									</div>
								</a>
							</div>
							<div class="doc-article-item all hardware-ascend hardware-gpu hardware-cpu stage-Data-Preparation experience-local experience-online hidden">
								<a href="https://www.mindspore.cn/docs/programming_guide/zh-CN/master/enable_cache.html" class="article-link">
									<div>
										<div class="doc-article-head">
											<span class="doc-head-content">应用单节点数据缓存</span>
										</div>
										<div class="doc-article-desc">
										对于需要重复访问远程的数据集或需要重复从磁盘中读取数据集的情况，可以使用单节点缓存算子将数据集缓存于本地内存中，以加速数据集的读取。
										</div>
									</div>
								</a>
							</div>
							<div class="doc-article-item all hardware-ascend hardware-gpu stage-Model-Optimization experience-local experience-online hidden">
								<a href="https://www.mindspore.cn/docs/programming_guide/zh-CN/master/enable_graph_kernel_fusion.html" class="article-link">
									<div>
										<div class="doc-article-head">
											<span class="doc-head-content">使能图算融合</span>
										</div>
										<div class="doc-article-desc">
										图算融合是MindSpore特有的网络性能优化技术。
										</div>
									</div>
								</a>
							</div>
							<div class="doc-article-item all hardware-ascend hardware-gpu stage-Model-Optimization experience-local experience-online hidden">
								<a href="https://www.mindspore.cn/docs/programming_guide/zh-CN/master/enable_mixed_precision.html" class="article-link">
									<div>
										<div class="doc-article-head">
											<span class="doc-head-content">使能混合精度</span>
										</div>
										<div class="doc-article-desc">
										混合精度训练方法是通过混合使用单精度和半精度数据格式来加速深度神经网络训练的过程，同时保持了单精度训练所能达到的网络精度。
										</div>
									</div>
								</a>
							</div>
							<div class="doc-article-item all hardware-ascend hardware-gpu hardware-cpu stage-Model-Running experience-local experience-online hidden">
								<a href="https://www.mindspore.cn/docs/programming_guide/zh-CN/master/evaluate_the_model_during_training.html" class="article-link">
									<div>
										<div class="doc-article-head">
											<span class="doc-head-content">训练时验证模型</span>
										</div>
										<div class="doc-article-desc">
										本文将采用训练时验证模型方法，以LeNet网络为样本，进行示例。
										</div>
									</div>
								</a>
							</div>
							<div class="doc-article-item all hardware-ascend hardware-gpu hardware-cpu stage-Model-Development experience-local hidden">
								<a href="https://www.mindspore.cn/docs/programming_guide/zh-CN/master/grad_operation.html" class="article-link">
									<div>
										<div class="doc-article-head">
											<span class="doc-head-content">求导</span>
										</div>
										<div class="doc-article-desc">
										MindSpore的GradOperation接口用于生成输入函数的梯度，利用get_all、get_by_list和sens_param参数控制梯度的计算方式。
										</div>
									</div>
								</a>
							</div>
							<div class="doc-article-item all hardware-gpu stage-Whole-Process experience-local hidden">
								<a href="https://www.mindspore.cn/docs/programming_guide/zh-CN/master/hpc_gomo.html" class="article-link">
									<div>
										<div class="doc-article-head">
											<span class="doc-head-content">实现区域海洋模型GOMO</span>
										</div>
										<div class="doc-article-desc">
										GOMO（Generalized Operator Modelling of the Ocean）是基于 OpenArray 的三维区域海洋模型，我们使用MindSpore对GOMO模型进行框架加速，结合GPU，能获得较大的性能提升。
										</div>
									</div>
								</a>
							</div>
							<div class="doc-article-item all hardware-ascend hardware-gpu hardware-cpu stage-Model-Development experience-local hidden">
								<a href="https://www.mindspore.cn/docs/programming_guide/zh-CN/master/hypermap.html" class="article-link">
									<div>
										<div class="doc-article-head">
											<span class="doc-head-content">运算重载</span>
										</div>
										<div class="doc-article-desc">
										 <em>mindspore.ops.composite</em> 中提供了一些涉及图变换的组合类算子，例如 <em>MultitypeFuncGraph</em> 、 <em>HyperMap</em> 等。
										</div>
									</div>
								</a>
							</div>
							<div class="doc-article-item all hardware-ascend experience-local hidden">
								<a href="https://www.mindspore.cn/docs/programming_guide/zh-CN/master/incremental_operator_build.html" class="article-link">
									<div>
										<div class="doc-article-head">
											<span class="doc-head-content">算子增量编译</span>
										</div>
										<div class="doc-article-desc">
										本章节介绍算子增量编译，目前该功能仅支持在昇腾AI芯片上使用。
										</div>
									</div>
								</a>
							</div>
							<div class="doc-article-item all hardware-ascend hardware-gpu hardware-cpu stage-Model-Development experience-local hidden">
								<a href="https://www.mindspore.cn/docs/programming_guide/zh-CN/master/indefinite_parameter.html" class="article-link">
									<div>
										<div class="doc-article-head">
											<span class="doc-head-content">参数传递</span>
										</div>
										<div class="doc-article-desc">
										本文介绍不定参数在网络构建中的使用。
										</div>
									</div>
								</a>
							</div>
							<div class="doc-article-item all hardware-ascend hardware-gpu hardware-cpu stage-Model-Development experience-local experience-online hidden">
								<a href="https://www.mindspore.cn/docs/programming_guide/zh-CN/master/initializer.html" class="article-link">
									<div>
										<div class="doc-article-head">
											<span class="doc-head-content">Initializer初始化器</span>
										</div>
										<div class="doc-article-desc">
										Initializer类是MindSpore中用于进行初始化的基本数据结构，本文针对使用initializer对参数进行初始化的方法进行详细介绍。
										</div>
									</div>
								</a>
							</div>
							<div class="doc-article-item all hardware-ascend hardware-gpu hardware-cpu stage-Data-Preparation experience-local experience-online hidden">
								<a href="https://www.mindspore.cn/docs/programming_guide/zh-CN/master/load_dataset_image.html" class="article-link">
									<div>
										<div class="doc-article-head">
											<span class="doc-head-content">加载图像数据集</span>
										</div>
										<div class="doc-article-desc">
										MindSpore提供的 <em>mindspore.dataset</em> 模块可以帮助用户构建数据集对象，分批次地读取图像数据。
										</div>
									</div>
								</a>
							</div>
							<div class="doc-article-item all hardware-ascend hardware-gpu hardware-cpu stage-Data-Preparation experience-local hidden">
								<a href="https://www.mindspore.cn/docs/programming_guide/zh-CN/master/load_dataset_networks.html" class="article-link">
									<div>
										<div class="doc-article-head">
											<span class="doc-head-content">典型网络数据加载和处理</span>
										</div>
										<div class="doc-article-desc">
										本文介绍典型网络数据加载和处理。
										</div>
									</div>
								</a>
							</div>
							<div class="doc-article-item all hardware-ascend hardware-gpu hardware-cpu stage-Data-Preparation experience-local experience-online hidden">
								<a href="https://www.mindspore.cn/docs/programming_guide/zh-CN/master/load_dataset_text.html" class="article-link">
									<div>
										<div class="doc-article-head">
											<span class="doc-head-content">加载文本数据集</span>
										</div>
										<div class="doc-article-desc">
										MindSpore提供的 <em>mindspore.dataset</em> 模块可以帮助用户构建数据集对象，分批次地读取文本数据。
										</div>
									</div>
								</a>
							</div>
							<div class="doc-article-item all hardware-ascend hardware-gpu hardware-cpu stage-Model-Loading experience-local experience-online hidden">
								<a href="https://www.mindspore.cn/docs/programming_guide/zh-CN/master/load_model_for_inference_and_transfer.html" class="article-link">
									<div>
										<div class="doc-article-head">
											<span class="doc-head-content">加载模型用于推理或迁移学习</span>
										</div>
										<div class="doc-article-desc">
										在模型训练过程中保存在本地的CheckPoint文件，可以帮助用户进行推理或迁移学习使用。
										</div>
									</div>
								</a>
							</div>
							<div class="doc-article-item all hardware-ascend hardware-gpu hardware-cpu stage-Model-Development experience-local hidden">
								<a href="https://www.mindspore.cn/docs/programming_guide/zh-CN/master/loss.html" class="article-link">
									<div>
										<div class="doc-article-head">
											<span class="doc-head-content">损失函数</span>
										</div>
										<div class="doc-article-desc">
										损失函数，又叫目标函数，用于衡量预测值与真实值差异的程度。定义一个好的损失函数，可以有效提高模型的性能。
										</div>
									</div>
								</a>
							</div>
							<div class="doc-article-item all hardware-ascend hardware-gpu hardware-cpu stage-Model-Development experience-local hidden">
								<a href="https://www.mindspore.cn/docs/programming_guide/zh-CN/master/lossscale.html" class="article-link">
									<div>
										<div class="doc-article-head">
											<span class="doc-head-content">LossScale</span>
										</div>
										<div class="doc-article-desc">
										LossScale的主要思想是在计算loss时，将loss扩大一定的倍数，由于链式法则的存在，梯度也会相应扩大，然后在优化器更新权重时再缩小相应的倍数，从而避免了数据下溢的情况又不影响计算结果。
										</div>
									</div>
								</a>
							</div>
							<div class="doc-article-item all hardware-ascend hardware-gpu hardware-cpu stage-Model-Development experience-local hidden">
								<a href="https://www.mindspore.cn/docs/programming_guide/zh-CN/master/model_use_guide.html" class="article-link">
									<div>
										<div class="doc-article-head">
											<span class="doc-head-content">Model基本使用</span>
										</div>
										<div class="doc-article-desc">
										本文档讲述如何使用高阶API <em>Model</em> 进行模型训练和评估。
										</div>
									</div>
								</a>
							</div>
							<div class="doc-article-item all hardware-ascend hardware-gpu hardware-cpu stage-Model-Running experience-local hidden">
								<a href="https://www.mindspore.cn/docs/programming_guide/zh-CN/master/ms_function.html" class="article-link">
									<div>
										<div class="doc-article-head">
											<span class="doc-head-content">ms_function动静结合</span>
										</div>
										<div class="doc-article-desc">
										ms_function的作用是在PyNative模式下提升执行性能。在MindSpore框架中，PyNative模式（即动态图模式）下，用户可以使用完整的Python语法，更加简单方便地使用MindSpore进行网络调优。与此同时，PyNative模式也会导致一部分性能的损失。
										</div>
									</div>
								</a>
							</div>
							<div class="doc-article-item all hardware-ascend hardware-gpu hardware-cpu stage-Inference experience-local hidden">
								<a href="https://www.mindspore.cn/docs/programming_guide/zh-CN/master/multi_platform_inference.html" class="article-link">
									<div>
										<div class="doc-article-head">
											<span class="doc-head-content">推理模型总览</span>
										</div>
										<div class="doc-article-desc">
										MindSpore可以基于训练好的模型，在不同的硬件平台上执行推理任务。
										</div>
									</div>
								</a>
							</div>
							<div class="doc-article-item all hardware-ascend stage-Inference experience-local hidden">
								<a href="https://www.mindspore.cn/docs/programming_guide/zh-CN/master/multi_platform_inference_ascend_310_air.html" class="article-link">
									<div>
										<div class="doc-article-head">
											<span class="doc-head-content">Ascend 310 AI处理器上使用AIR模型进行推理</span>
										</div>
										<div class="doc-article-desc">
										Ascend 310是面向边缘场景的高能效高集成度AI处理器，可以实现图像、视频等多种数据分析与推理计算，可广泛用于智能监控、机器人、无人机、视频服务器等场景。
										</div>
									</div>
								</a>
							</div>
							<div class="doc-article-item all hardware-ascend stage-Inference experience-local hidden">
								<a href="https://www.mindspore.cn/docs/programming_guide/zh-CN/master/multi_platform_inference_ascend_310_mindir.html" class="article-link">
									<div>
										<div class="doc-article-head">
											<span class="doc-head-content">Ascend 310 AI处理器上使用MindIR模型进行推理</span>
										</div>
										<div class="doc-article-desc">
										Ascend 310是面向边缘场景的高能效高集成度AI处理器，本教程介绍如何在Ascend 310上使用MindSpore基于MindIR模型文件执行推理。
										</div>
									</div>
								</a>
							</div>
							<div class="doc-article-item all hardware-ascend stage-Inference experience-local hidden">
								<a href="https://www.mindspore.cn/docs/programming_guide/zh-CN/master/multi_platform_inference_ascend_910.html" class="article-link">
									<div>
										<div class="doc-article-head">
											<span class="doc-head-content">Ascend 910 AI处理器上推理</span>
										</div>
										<div class="doc-article-desc">
										用户可以创建C++应用程序，调用MindSpore的C++接口推理MindIR模型。
										</div>
									</div>
								</a>
							</div>
							<div class="doc-article-item all hardware-cpu stage-Inference experience-local hidden">
								<a href="https://www.mindspore.cn/docs/programming_guide/zh-CN/master/multi_platform_inference_cpu.html" class="article-link">
									<div>
										<div class="doc-article-head">
											<span class="doc-head-content">CPU上推理</span>
										</div>
										<div class="doc-article-desc">
										本文介绍如何在CPU上进行推理。
										</div>
									</div>
								</a>
							</div>
							<div class="doc-article-item all hardware-gpu stage-Inference experience-local hidden">
								<a href="https://www.mindspore.cn/docs/programming_guide/zh-CN/master/multi_platform_inference_gpu.html" class="article-link">
									<div>
										<div class="doc-article-head">
											<span class="doc-head-content">GPU上推理</span>
										</div>
										<div class="doc-article-desc">
										用户可以创建C++应用程序，调用MindSpore的C++接口推理MindIR模型。
										</div>
									</div>
								</a>
							</div>
							<div class="doc-article-item all hardware-ascend stage-Whole-Process experience-local hidden">
								<a href="https://www.mindspore.cn/docs/programming_guide/zh-CN/master/nlp_bert_poetry.html" class="article-link">
									<div>
										<div class="doc-article-head">
											<span class="doc-head-content">使用BERT网络实现智能写诗</span>
										</div>
										<div class="doc-article-desc">
										五千年历史孕育了深厚的中华文化，而诗词是中华文化不可或缺的一部分，今天理科生MindSpore也来秀一秀文艺范儿！
										</div>
									</div>
								</a>
							</div>
							<div class="doc-article-item all hardware-gpu hardware-cpu stage-Whole-Process experience-local experience-online hidden">
								<a href="https://www.mindspore.cn/docs/programming_guide/zh-CN/master/nlp_sentimentnet.html" class="article-link">
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
							<div class="doc-article-item all hardware-ascend stage-Whole-Process experience-local hidden">
								<a href="https://www.mindspore.cn/docs/programming_guide/zh-CN/master/nlp_tprr.html" class="article-link">
									<div>
										<div class="doc-article-head">
											<span class="doc-head-content">多跳知识推理问答模型TPRR</span>
										</div>
										<div class="doc-article-desc">
										TPRR(Thinking Path Re-Ranker)是由华为提出的基于开放域多跳问答的通用模型，用以实现多跳知识推理问答。使用MindSpore混合精度特性对TPRR模型进行框架加速，结合Ascend，能获得显著的性能提升。
										</div>
									</div>
								</a>
							</div>
							<div class="doc-article-item all hardware-ascend hardware-gpu hardware-cpu stage-Model-Development experience-local experience-online hidden">
								<a href="https://www.mindspore.cn/docs/programming_guide/zh-CN/master/numpy.html" class="article-link">
									<div>
										<div class="doc-article-head">
											<span class="doc-head-content">MindSpore NumPy函数</span>
										</div>
										<div class="doc-article-desc">
										MindSpore NumPy工具包提供了一系列类NumPy接口。用户可以使用类NumPy语法在MindSpore上进行模型的搭建。
										</div>
									</div>
								</a>
							</div>
							<div class="doc-article-item all hardware-ascend stage-Inference experience-local hidden">
								<a href="https://www.mindspore.cn/docs/programming_guide/zh-CN/master/online_inference.html" class="article-link">
									<div>
										<div class="doc-article-head">
											<span class="doc-head-content">加载Checkpoint在线推理</span>
										</div>
										<div class="doc-article-desc">
										首先构建模型，然后使用 <em>mindspore</em> 模块的 <em>load_checkpoint</em> 和 <em>load_param_into_net</em> 从本地加载模型与参数，传入验证数据集后即可进行模型推理，验证数据集的处理方式与训练数据集相同。
										</div>
									</div>
								</a>
							</div>
							<div class="doc-article-item all hardware-ascend hardware-gpu hardware-cpu stage-Model-Running experience-local hidden">
								<a href="https://www.mindspore.cn/docs/programming_guide/zh-CN/master/on_device.html" class="article-link">
									<div>
										<div class="doc-article-head">
											<span class="doc-head-content">on-device执行</span>
										</div>
										<div class="doc-article-desc">
										MindSpore支持的后端包括Ascend、GPU、CPU，所谓On Device中的Device通常指Ascend（昇腾）AI处理器。
										</div>
									</div>
								</a>
							</div>
							<div class="doc-article-item all hardware-ascend hardware-gpu hardware-cpu stage-Beginner experience-local hidden">
								<a href="https://www.mindspore.cn/docs/programming_guide/zh-CN/master/operators_classification.html" class="article-link">
									<div>
										<div class="doc-article-head">
											<span class="doc-head-content">算子分类</span>
										</div>
										<div class="doc-article-desc">
										算子主要分为Primitivie算子和nn算子。
										</div>
									</div>
								</a>
							</div>
							<div class="doc-article-item all hardware-ascend hardware-gpu hardware-cpu stage-Model-Development experience-local hidden">
								<a href="https://www.mindspore.cn/docs/programming_guide/zh-CN/master/optim.html" class="article-link">
									<div>
										<div class="doc-article-head">
											<span class="doc-head-content">优化器</span>
										</div>
										<div class="doc-article-desc">
										优化器在模型训练过程中，用于计算和更新网络参数，合适的优化器可以有效减少训练时间，提高最终模型性能。
										</div>
									</div>
								</a>
							</div>
							<div class="doc-article-item all hardware-ascend hardware-gpu hardware-cpu stage-Data-Preparation experience-local experience-online hidden">
								<a href="https://www.mindspore.cn/docs/programming_guide/zh-CN/master/optimize_data_processing.html" class="article-link">
									<div>
										<div class="doc-article-head">
											<span class="doc-head-content">优化数据处理</span>
										</div>
										<div class="doc-article-desc">
										数据是整个深度学习中最重要的一环，因为数据的好坏决定了最终结果的上限，模型的好坏只是去无限逼近这个上限，所以高质量的数据输入，会在整个深度神经网络中起到积极作用。
										</div>
									</div>
								</a>
							</div>
							<div class="doc-article-item all hardware-ascend hardware-gpu stage-Distributed-Parallel experience-local hidden">
								<a href="https://www.mindspore.cn/docs/programming_guide/zh-CN/master/pangu_alpha.html" class="article-link">
									<div>
										<div class="doc-article-head">
											<span class="doc-head-content">鹏程·盘古模型网络多维度混合并行解析</span>
										</div>
										<div class="doc-article-desc">
										在MindSpore发布的鹏程·盘古模型模型中，我们看到借助多维度自动混合并行可以实现超大规模Transformer网络的分布式训练。这篇文章将从网络脚本出发，详解模型各个组成部分的切分方式。
										</div>
									</div>
								</a>
							</div>
							<div class="doc-article-item all hardware-ascend hardware-gpu hardware-cpu stage-Model-Development experience-local experience-online hidden">
								<a href="https://www.mindspore.cn/docs/programming_guide/zh-CN/master/parameter.html" class="article-link">
									<div>
										<div class="doc-article-head">
											<span class="doc-head-content">网络参数</span>
										</div>
										<div class="doc-article-desc">
										MindSpore提供了网络参数初始化模块，用户可以通过封装算子来调用字符串、Initializer子类或自定义Tensor等方式完成对网络参数进行初始化。本章主要介绍了 <em>Parameter</em> 的初始化以及属性和方法的使用，同时介绍了 <em>ParameterTuple</em> 和参数的依赖控制。
										</div>
									</div>
								</a>
							</div>
							<div class="doc-article-item all hardware-ascend hardware-gpu hardware-cpu stage-Beginner experience-local experience-online hidden">
								<a href="https://www.mindspore.cn/docs/programming_guide/zh-CN/master/parameter_introduction.html" class="article-link">
									<div>
										<div class="doc-article-head">
											<span class="doc-head-content">Parameter</span>
										</div>
										<div class="doc-article-desc">
										 <em>Parameter</em> 是变量张量，代表在训练网络时，需要被更新的参数，一般包括权重 <em>weight</em> 和偏置 <em>bias</em> 。
										</div>
									</div>
								</a>
							</div>
							<div class="doc-article-item all hardware-ascend hardware-gpu hardware-cpu stage-Data-Preparation experience-local experience-online hidden">
								<a href="https://www.mindspore.cn/docs/programming_guide/zh-CN/master/pipeline_common.html" class="article-link">
									<div>
										<div class="doc-article-head">
											<span class="doc-head-content">通用数据处理</span>
										</div>
										<div class="doc-article-desc">
										数据是深度学习的基础，良好的数据输入可以对整个深度神经网络训练起到非常积极的作用。在训练前对已加载的数据集进行数据处理，可以解决诸如数据量过大、样本分布不均等问题，从而获得更加优化的数据输入。
										</div>
									</div>
								</a>
							</div>
							<div class="doc-article-item all hardware-ascend hardware-gpu hardware-cpu experience-local hidden">
								<a href="https://www.mindspore.cn/docs/programming_guide/zh-CN/master/read_ir_files.html" class="article-link">
									<div>
										<div class="doc-article-head">
											<span class="doc-head-content">如何查看IR文件</span>
										</div>
										<div class="doc-article-desc">
										在图模式 <em>context.set_context(mode=context.GRAPH_MODE)</em> 下运行用MindSpore编写的模型时，若配置中设置了 <em>context.set_context(save_graphs=True)</em> ，运行时会输出一些图编译过程中生成的一些中间文件，我们称为IR文件。
										</div>
									</div>
								</a>
							</div>
							<div class="doc-article-item all hardware-ascend hardware-gpu hardware-cpu stage-Model-Running experience-local experience-online hidden">
								<a href="https://www.mindspore.cn/docs/programming_guide/zh-CN/master/run.html" class="article-link">
									<div>
										<div class="doc-article-head">
											<span class="doc-head-content">运行方式</span>
										</div>
										<div class="doc-article-desc">
										执行主要有三种方式：单算子、普通函数和网络训练模型。
										</div>
									</div>
								</a>
							</div>
							<div class="doc-article-item all hardware-ascend hardware-gpu hardware-cpu stage-Data-Preparation experience-local experience-online hidden">
								<a href="https://www.mindspore.cn/docs/programming_guide/zh-CN/master/sampler.html" class="article-link">
									<div>
										<div class="doc-article-head">
											<span class="doc-head-content">数据采样</span>
										</div>
										<div class="doc-article-desc">
										MindSpore提供了多种用途的采样器（Sampler），帮助用户对数据集进行不同形式的采样，以满足训练需求，能够解决诸如数据集过大或样本类别分布不均等问题。
										</div>
									</div>
								</a>
							</div>
							<div class="doc-article-item all hardware-ascend hardware-gpu stage-Distributed-Parallel stage-Model-Export stage-Model-Loading experience-local hidden">
								<a href="https://www.mindspore.cn/docs/programming_guide/zh-CN/master/save_load_model_hybrid_parallel.html" class="article-link">
									<div>
										<div class="doc-article-head">
											<span class="doc-head-content">保存和加载模型（HyBrid Parallel模式）</span>
										</div>
										<div class="doc-article-desc">
										MindSpore模型并行场景下，每个实例进程只保存有本节点对应的参数数据。对于模型并行的Cell，其在每个节点上的参数数据，都是完整参数数据的一个切片。
										</div>
									</div>
								</a>
							</div>
							<div class="doc-article-item all hardware-ascend hardware-gpu hardware-cpu stage-Model-Export experience-local experience-online hidden">
								<a href="https://www.mindspore.cn/docs/programming_guide/zh-CN/master/save_model.html" class="article-link">
									<div>
										<div class="doc-article-head">
											<span class="doc-head-content">保存模型</span>
										</div>
										<div class="doc-article-desc">
										在模型训练过程中，可以添加检查点(CheckPoint)用于保存模型的参数，以便执行推理及再训练使用。如果想继续在不同硬件平台上做推理，可通过网络和CheckPoint格式文件生成对应的MindIR、AIR和ONNX格式文件。
										</div>
									</div>
								</a>
							</div>
							<div class="doc-article-item all hardware-ascend hardware-gpu hardware-cpu stage-Model-Evaluation experience-local hidden">
								<a href="https://www.mindspore.cn/docs/programming_guide/zh-CN/master/self_define_metric.html" class="article-link">
									<div>
										<div class="doc-article-head">
											<span class="doc-head-content">自定义Metrics验证模型推理精度</span>
										</div>
										<div class="doc-article-desc">
										当训练任务结束，常常需要评价函数(Metrics)来评估模型的好坏。不同的训练任务往往需要不同的Metrics函数。虽然MindSpore提供了大部分常见任务的评价指标，但是无法满足所有任务的需求。因此使用者可针对具体的任务自定义Metrics来评估训练的模型。
										</div>
									</div>
								</a>
							</div>
							<div class="doc-article-item all hardware-ascend hardware-gpu hardware-cpu stage-Beginner experience-local experience-online hidden">
								<a href="https://www.mindspore.cn/docs/programming_guide/zh-CN/master/tensor.html" class="article-link">
									<div>
										<div class="doc-article-head">
											<span class="doc-head-content">Tensor</span>
										</div>
										<div class="doc-article-desc">
										张量（Tensor）是MindSpore网络运算中的基本数据结构。
										</div>
									</div>
								</a>
							</div>
							<div class="doc-article-item all hardware-ascend stage-Function-Extension experience-local hidden">
								<a href="https://www.mindspore.cn/docs/programming_guide/zh-CN/master/thor.html" class="article-link">
									<div>
										<div class="doc-article-head">
											<span class="doc-head-content">二阶优化器THOR介绍</span>
										</div>
										<div class="doc-article-desc">
										深度学习训练过程可以看成损失函数损失值下降过程，合适的优化器可以让深度学习训练时间大大减少。
										</div>
									</div>
								</a>
							</div>
							<div class="doc-article-item all hardware-ascend hardware-gpu hardware-cpu stage-Data-Preparation experience-local experience-online hidden">
								<a href="https://www.mindspore.cn/docs/programming_guide/zh-CN/master/tokenizer.html" class="article-link">
									<div>
										<div class="doc-article-head">
											<span class="doc-head-content">文本处理与增强</span>
										</div>
										<div class="doc-article-desc">
										分词就是将连续的字序列按照一定的规范重新组合成词序列的过程，合理的进行分词有助于语义的理解。
										</div>
									</div>
								</a>
							</div>
							<div class="doc-article-item all hardware-ascend hardware-gpu hardware-cpu stage-Model-Development stage-Model-Running stage-Model-Evaluation experience-local hidden">
								<a href="https://www.mindspore.cn/docs/programming_guide/zh-CN/master/train_and_eval.html" class="article-link">
									<div>
										<div class="doc-article-head">
											<span class="doc-head-content">构建训练与评估网络</span>
										</div>
										<div class="doc-article-desc">
										本文档重点介绍如何使用这些元素组成训练和评估网络。
										</div>
									</div>
								</a>
							</div>
							<div class="doc-article-item all hardware-ascend stage-Whole-Process experience-local hidden">
								<a href="https://www.mindspore.cn/docs/programming_guide/zh-CN/master/use_on_the_cloud.html" class="article-link">
									<div>
										<div class="doc-article-head">
											<span class="doc-head-content">在云上使用MindSpore</span>
										</div>
										<div class="doc-article-desc">
										ModelArts是华为云提供的面向开发者的一站式AI开发平台，集成了昇腾AI处理器资源池，用户可以在该平台下体验MindSpore。
										</div>
									</div>
								</a>
							</div>
							<div class="doc-article-item all hardware-ascend hardware-gpu hardware-cpu stage-Design experience-local hidden">
								<a href="https://www.mindspore.cn/docs/programming_guide/zh-CN/master/design/all_scenarios_architecture.html" class="article-link">
									<div>
										<div class="doc-article-head">
											<span class="doc-head-content">全场景统一架构</span>
										</div>
										<div class="doc-article-desc">
										MindSpore旨在提供端边云全场景的AI框架。MindSpore可部署于端、边、云不同的硬件环境，满足不同环境的差异化需求，如支持端侧的轻量化部署，支持云侧丰富的训练功能如自动微分、混合精度、模型易用编程等。
										</div>
									</div>
								</a>
							</div>
							<div class="doc-article-item all hardware-ascend hardware-gpu hardware-cpu stage-Design stage-Data-Preparation experience-local hidden">
								<a href="https://www.mindspore.cn/docs/programming_guide/zh-CN/master/design/data_engine.html" class="article-link">
									<div>
										<div class="doc-article-head">
											<span class="doc-head-content">高性能数据处理引擎</span>
										</div>
										<div class="doc-article-desc">
										MindSpore训练数据处理引擎核心是将训练样本（数据集）高效、灵活的转换至Tensor，并将该Tensor提供给训练网络用于训练。
										</div>
									</div>
								</a>
							</div>
							<div class="doc-article-item all hardware-ascend hardware-gpu stage-Design stage-Distributed-Parallel experience-local hidden">
								<a href="https://www.mindspore.cn/docs/programming_guide/zh-CN/master/design/distributed_training_design.html" class="article-link">
									<div>
										<div class="doc-article-head">
											<span class="doc-head-content">分布式训练</span>
										</div>
										<div class="doc-article-desc">
										MindSpore支持了当前主流的分布式训练范式并开发了一套自动混合并行解决方案。本篇设计文档将会集中介绍几种并行训练方式的设计原理，同时指导用户进行自定义开发。
										</div>
									</div>
								</a>
							</div>
							<div class="doc-article-item all hardware-ascend hardware-gpu hardware-cpu stage-Design stage-Model-Running experience-local hidden">
								<a href="https://www.mindspore.cn/docs/programming_guide/zh-CN/master/design/dynamic_graph_and_static_graph.html" class="article-link">
									<div>
										<div class="doc-article-head">
											<span class="doc-head-content">动态图和静态图</span>
										</div>
										<div class="doc-article-desc">
										目前主流的深度学习框架的执行模式有两种，分别为静态图模式和动态图模式。
										</div>
									</div>
								</a>
							</div>
							<div class="doc-article-item all hardware-ascend hardware-gpu hardware-cpu stage-Design experience-local hidden">
								<a href="https://www.mindspore.cn/docs/programming_guide/zh-CN/master/design/glossary.html" class="article-link">
									<div>
										<div class="doc-article-head">
											<span class="doc-head-content">术语</span>
										</div>
										<div class="doc-article-desc">
										本文整理了MindSpore术语。
										</div>
									</div>
								</a>
							</div>
							<div class="doc-article-item all hardware-ascend hardware-gpu hardware-cpu stage-Design stage-Model-Development experience-local hidden">
								<a href="https://www.mindspore.cn/docs/programming_guide/zh-CN/master/design/gradient.html" class="article-link">
									<div>
										<div class="doc-article-head">
											<span class="doc-head-content">函数式可微分编程</span>
										</div>
										<div class="doc-article-desc">
										深度学习等现代AI算法通过使用大量的数据来学习拟合出一个优化后带参模型，其中使用的学习算法多是基于现实数据自模型中的经验误差来反向传播以更新模型的参数，自动微分技术（Automatic Differentiation， AD）正是其中的关键技术。
										</div>
									</div>
								</a>
							</div>
							<div class="doc-article-item all hardware-ascend hardware-gpu stage-Design stage-Model-Running experience-local hidden">
								<a href="https://www.mindspore.cn/docs/programming_guide/zh-CN/master/design/heterogeneous_training.html" class="article-link">
									<div>
										<div class="doc-article-head">
											<span class="doc-head-content">异构并行训练</span>
										</div>
										<div class="doc-article-desc">
										异构并行训练方法是通过分析图上算子内存占用和计算密集度，将内存消耗巨大或适合CPU逻辑处理的算子切分到CPU子图，将内存消耗较小计算密集型算子切分到硬件加速器子图，框架协同不同子图进行网络训练，使得处于不同硬件且无依赖关系的子图能够并行进行执行的过程。
										</div>
									</div>
								</a>
							</div>
							<div class="doc-article-item all hardware-ascend hardware-gpu stage-Design stage-Model-Development experience-local hidden">
								<a href="https://www.mindspore.cn/docs/programming_guide/zh-CN/master/design/mindir.html" class="article-link">
									<div>
										<div class="doc-article-head">
											<span class="doc-head-content">MindSpore IR（MindIR）</span>
										</div>
										<div class="doc-article-desc">
										中间表示（IR）是程序编译过程中介于源语言和目标语言之间的程序表示，以方便编译器进行程序分析和优化，因此IR的设计需要考虑从源语言到目标语言的转换难度，同时考虑程序分析和优化的易用性和性能。
										</div>
									</div>
								</a>
							</div>
							<div class="doc-article-item all hardware-ascend hardware-gpu hardware-cpu stage-Design experience-local hidden">
								<a href="https://www.mindspore.cn/docs/programming_guide/zh-CN/master/design/technical_white_paper.html" class="article-link">
									<div>
										<div class="doc-article-head">
											<span class="doc-head-content">技术白皮书</span>
										</div>
										<div class="doc-article-desc">
										深度学习研究和应用在近几十年得到了爆炸式的发展，掀起了人工智能的第三次浪潮，并且在图像识别、语音识别与合成、无人驾驶、机器视觉等方面取得了巨大的成功。
										</div>
									</div>
								</a>
							</div>
							<div class="doc-article-item all experience-local hidden">
								<a href="https://www.mindspore.cn/docs/programming_guide/zh-CN/master/quick_start/quick_video.html" class="article-link">
									<div>
										<div class="doc-article-head">
											<span class="doc-head-content">手把手安装和体验</span>
										</div>
										<div class="doc-article-desc">
										 本文档中包含手把手系列视频，码云Gitee不支持展示，请于官方网站对应教程中查看。
										</div>
									</div>
								</a>
							</div>
							<div class="doc-article-item all experience-local hidden">
								<a href="https://www.mindspore.cn/docs/programming_guide/zh-CN/master/quick_start/quick_video/ascend310.html" class="article-link">
									<div>
										<div class="doc-article-head">
											<span class="doc-head-content">Ascend 310上安装MindSpore</span>
										</div>
										<div class="doc-article-desc">
										 本文档中包含手把手系列视频，码云Gitee不支持展示，请于官方网站对应教程中查看。
										</div>
									</div>
								</a>
							</div>
							<div class="doc-article-item all experience-local hidden">
								<a href="https://www.mindspore.cn/docs/programming_guide/zh-CN/master/quick_start/quick_video/ascend910.html" class="article-link">
									<div>
										<div class="doc-article-head">
											<span class="doc-head-content">Ascend 910上安装MindSpore</span>
										</div>
										<div class="doc-article-desc">
										 本文档中包含手把手系列视频，码云Gitee不支持展示，请于官方网站对应教程中查看。
										</div>
									</div>
								</a>
							</div>
							<div class="doc-article-item all experience-local hidden">
								<a href="https://www.mindspore.cn/docs/programming_guide/zh-CN/master/quick_start/quick_video/ascend910_operator_development.html" class="article-link">
									<div>
										<div class="doc-article-head">
											<span class="doc-head-content">Ascend 910算子开发</span>
										</div>
										<div class="doc-article-desc">
										 本文档中包含手把手系列视频，码云Gitee不支持展示，请于官方网站对应教程中查看。
										</div>
									</div>
								</a>
							</div>
							<div class="doc-article-item all experience-local hidden">
								<a href="https://www.mindspore.cn/docs/programming_guide/zh-CN/master/quick_start/quick_video/community.html" class="article-link">
									<div>
										<div class="doc-article-head">
											<span class="doc-head-content">参与社区建设</span>
										</div>
										<div class="doc-article-desc">
										 本文档中包含手把手系列视频，码云Gitee不支持展示，请于官方网站对应教程中查看。
										</div>
									</div>
								</a>
							</div>
							<div class="doc-article-item all experience-local hidden">
								<a href="https://www.mindspore.cn/docs/programming_guide/zh-CN/master/quick_start/quick_video/cpu_operator_development.html" class="article-link">
									<div>
										<div class="doc-article-head">
											<span class="doc-head-content">CPU算子开发</span>
										</div>
										<div class="doc-article-desc">
										 本文档中包含手把手系列视频，码云Gitee不支持展示，请于官方网站对应教程中查看。
										</div>
									</div>
								</a>
							</div>
							<div class="doc-article-item all experience-local hidden">
								<a href="https://www.mindspore.cn/docs/programming_guide/zh-CN/master/quick_start/quick_video/cpu_ubuntu.html" class="article-link">
									<div>
										<div class="doc-article-head">
											<span class="doc-head-content">CPU-Ubuntu上安装MindSpore</span>
										</div>
										<div class="doc-article-desc">
										 本文档中包含手把手系列视频，码云Gitee不支持展示，请于官方网站对应教程中查看。
										</div>
									</div>
								</a>
							</div>
							<div class="doc-article-item all experience-local hidden">
								<a href="https://www.mindspore.cn/docs/programming_guide/zh-CN/master/quick_start/quick_video/cpu_windows.html" class="article-link">
									<div>
										<div class="doc-article-head">
											<span class="doc-head-content">CPU-Windows上安装MindSpore</span>
										</div>
										<div class="doc-article-desc">
										 本文档中包含手把手系列视频，码云Gitee不支持展示，请于官方网站对应教程中查看。
										</div>
									</div>
								</a>
							</div>
							<div class="doc-article-item all experience-local hidden">
								<a href="https://www.mindspore.cn/docs/programming_guide/zh-CN/master/quick_start/quick_video/customized_debugging.html" class="article-link">
									<div>
										<div class="doc-article-head">
											<span class="doc-head-content">自定义调试</span>
										</div>
										<div class="doc-article-desc">
										 本文档中包含手把手系列视频，码云Gitee不支持展示，请于官方网站对应教程中查看。
										</div>
									</div>
								</a>
							</div>
							<div class="doc-article-item all experience-local hidden">
								<a href="https://www.mindspore.cn/docs/programming_guide/zh-CN/master/quick_start/quick_video/debug_in_pynative_mode.html" class="article-link">
									<div>
										<div class="doc-article-head">
											<span class="doc-head-content">使用PyNative模式调试</span>
										</div>
										<div class="doc-article-desc">
										 本文档中包含手把手系列视频，码云Gitee不支持展示，请于官方网站对应教程中查看。
										</div>
									</div>
								</a>
							</div>
							<div class="doc-article-item all experience-local hidden">
								<a href="https://www.mindspore.cn/docs/programming_guide/zh-CN/master/quick_start/quick_video/distributed_parallel_training_network_training.html" class="article-link">
									<div>
										<div class="doc-article-head">
											<span class="doc-head-content">分布式并行训练-网络训练</span>
										</div>
										<div class="doc-article-desc">
										 本文档中包含手把手系列视频，码云Gitee不支持展示，请于官方网站对应教程中查看。
										</div>
									</div>
								</a>
							</div>
							<div class="doc-article-item all experience-local hidden">
								<a href="https://www.mindspore.cn/docs/programming_guide/zh-CN/master/quick_start/quick_video/distributed_parallel_training_preparation.html" class="article-link">
									<div>
										<div class="doc-article-head">
											<span class="doc-head-content">分布式并行训练-准备工作</span>
										</div>
										<div class="doc-article-desc">
										 本文档中包含手把手系列视频，码云Gitee不支持展示，请于官方网站对应教程中查看。
										</div>
									</div>
								</a>
							</div>
							<div class="doc-article-item all experience-local hidden">
								<a href="https://www.mindspore.cn/docs/programming_guide/zh-CN/master/quick_start/quick_video/dump_in_graph_mode.html" class="article-link">
									<div>
										<div class="doc-article-head">
											<span class="doc-head-content">使用Dump功能在Graph模式调试</span>
										</div>
										<div class="doc-article-desc">
										 本文档中包含手把手系列视频，码云Gitee不支持展示，请于官方网站对应教程中查看。
										</div>
									</div>
								</a>
							</div>
							<div class="doc-article-item all experience-local hidden">
								<a href="https://www.mindspore.cn/docs/programming_guide/zh-CN/master/quick_start/quick_video/experience_on_modelarts.html" class="article-link">
									<div>
										<div class="doc-article-head">
											<span class="doc-head-content">云平台-ModelArts上使用MindSpore</span>
										</div>
										<div class="doc-article-desc">
										 本文档中包含手把手系列视频，码云Gitee不支持展示，请于官方网站对应教程中查看。
										</div>
									</div>
								</a>
							</div>
							<div class="doc-article-item all experience-local hidden">
								<a href="https://www.mindspore.cn/docs/programming_guide/zh-CN/master/quick_start/quick_video/gpu.html" class="article-link">
									<div>
										<div class="doc-article-head">
											<span class="doc-head-content">GPU上安装MindSpore</span>
										</div>
										<div class="doc-article-desc">
										 本文档中包含手把手系列视频，码云Gitee不支持展示，请于官方网站对应教程中查看。
										</div>
									</div>
								</a>
							</div>
							<div class="doc-article-item all experience-local hidden">
								<a href="https://www.mindspore.cn/docs/programming_guide/zh-CN/master/quick_start/quick_video/gpu_operator_development.html" class="article-link">
									<div>
										<div class="doc-article-head">
											<span class="doc-head-content">GPU算子开发</span>
										</div>
										<div class="doc-article-desc">
										 本文档中包含手把手系列视频，码云Gitee不支持展示，请于官方网站对应教程中查看。
										</div>
									</div>
								</a>
							</div>
							<div class="doc-article-item all experience-local hidden">
								<a href="https://www.mindspore.cn/docs/programming_guide/zh-CN/master/quick_start/quick_video/inference.html" class="article-link">
									<div>
										<div class="doc-article-head">
											<span class="doc-head-content">多平台推理</span>
										</div>
										<div class="doc-article-desc">
										 本文档中包含手把手系列视频，码云Gitee不支持展示，请于官方网站对应教程中查看。
										</div>
									</div>
								</a>
							</div>
							<div class="doc-article-item all experience-local hidden">
								<a href="https://www.mindspore.cn/docs/programming_guide/zh-CN/master/quick_start/quick_video/loading_the_dataset_and_converting_data_format.html" class="article-link">
									<div>
										<div class="doc-article-head">
											<span class="doc-head-content">加载数据集与转换格式</span>
										</div>
										<div class="doc-article-desc">
										 本文档中包含手把手系列视频，码云Gitee不支持展示，请于官方网站对应教程中查看。
										</div>
									</div>
								</a>
							</div>
							<div class="doc-article-item all experience-local hidden">
								<a href="https://www.mindspore.cn/docs/programming_guide/zh-CN/master/quick_start/quick_video/loading_the_model_from_hub.html" class="article-link">
									<div>
										<div class="doc-article-head">
											<span class="doc-head-content">从Hub中加载模型</span>
										</div>
										<div class="doc-article-desc">
										 本文档中包含手把手系列视频，码云Gitee不支持展示，请于官方网站对应教程中查看。
										</div>
									</div>
								</a>
							</div>
							<div class="doc-article-item all experience-local hidden">
								<a href="https://www.mindspore.cn/docs/programming_guide/zh-CN/master/quick_start/quick_video/mindArmour_differential_privacy.html" class="article-link">
									<div>
										<div class="doc-article-head">
											<span class="doc-head-content">MindArmour差分隐私</span>
										</div>
										<div class="doc-article-desc">
										 本文档中包含手把手系列视频，码云Gitee不支持展示，请于官方网站对应教程中查看。
										</div>
									</div>
								</a>
							</div>
							<div class="doc-article-item all experience-local hidden">
								<a href="https://www.mindspore.cn/docs/programming_guide/zh-CN/master/quick_start/quick_video/mindArmour_fuzzing.html" class="article-link">
									<div>
										<div class="doc-article-head">
											<span class="doc-head-content">MindArmour测试模型安全性之AI Fuzzer</span>
										</div>
										<div class="doc-article-desc">
										 本文档中包含手把手系列视频，码云Gitee不支持展示，请于官方网站对应教程中查看。
										</div>
									</div>
								</a>
							</div>
							<div class="doc-article-item all experience-local hidden">
								<a href="https://www.mindspore.cn/docs/programming_guide/zh-CN/master/quick_start/quick_video/mindArmour_installation_and_adversarial_attack_and_defense.html" class="article-link">
									<div>
										<div class="doc-article-head">
											<span class="doc-head-content">MindArmour安装与对抗攻防</span>
										</div>
										<div class="doc-article-desc">
										 本文档中包含手把手系列视频，码云Gitee不支持展示，请于官方网站对应教程中查看。
										</div>
									</div>
								</a>
							</div>
							<div class="doc-article-item all experience-local hidden">
								<a href="https://www.mindspore.cn/docs/programming_guide/zh-CN/master/quick_start/quick_video/mindArmour_membership_inference.html" class="article-link">
									<div>
										<div class="doc-article-head">
											<span class="doc-head-content">MindArmour测试模型安全性之成员推理</span>
										</div>
										<div class="doc-article-desc">
										 本文档中包含手把手系列视频，码云Gitee不支持展示，请于官方网站对应教程中查看。
										</div>
									</div>
								</a>
							</div>
							<div class="doc-article-item all experience-local hidden">
								<a href="https://www.mindspore.cn/docs/programming_guide/zh-CN/master/quick_start/quick_video/mindInsight_dashboard.html" class="article-link">
									<div>
										<div class="doc-article-head">
											<span class="doc-head-content">MindInsight训练看板</span>
										</div>
										<div class="doc-article-desc">
										 本文档中包含手把手系列视频，码云Gitee不支持展示，请于官方网站对应教程中查看。
										</div>
									</div>
								</a>
							</div>
							<div class="doc-article-item all experience-local hidden">
								<a href="https://www.mindspore.cn/docs/programming_guide/zh-CN/master/quick_start/quick_video/mindInsight_debugger.html" class="article-link">
									<div>
										<div class="doc-article-head">
											<span class="doc-head-content">使用调试器</span>
										</div>
										<div class="doc-article-desc">
										 本文档中包含手把手系列视频，码云Gitee不支持展示，请于官方网站对应教程中查看。
										</div>
									</div>
								</a>
							</div>
							<div class="doc-article-item all experience-local hidden">
								<a href="https://www.mindspore.cn/docs/programming_guide/zh-CN/master/quick_start/quick_video/mindInsight_installation_and_common_commands.html" class="article-link">
									<div>
										<div class="doc-article-head">
											<span class="doc-head-content">MindInsight安装与常用命令</span>
										</div>
										<div class="doc-article-desc">
										 本文档中包含手把手系列视频，码云Gitee不支持展示，请于官方网站对应教程中查看。
										</div>
									</div>
								</a>
							</div>
							<div class="doc-article-item all experience-local hidden">
								<a href="https://www.mindspore.cn/docs/programming_guide/zh-CN/master/quick_start/quick_video/mindInsight_lineage_and_scalars_comparison.html" class="article-link">
									<div>
										<div class="doc-article-head">
											<span class="doc-head-content">MindInsight溯源与对比看板</span>
										</div>
										<div class="doc-article-desc">
										 本文档中包含手把手系列视频，码云Gitee不支持展示，请于官方网站对应教程中查看。
										</div>
									</div>
								</a>
							</div>
							<div class="doc-article-item all experience-local hidden">
								<a href="https://www.mindspore.cn/docs/programming_guide/zh-CN/master/quick_start/quick_video/mindInsight_model_explanation.html" class="article-link">
									<div>
										<div class="doc-article-head">
											<span class="doc-head-content">MindInsight模型解释</span>
										</div>
										<div class="doc-article-desc">
										 本文档中包含手把手系列视频，码云Gitee不支持展示，请于官方网站对应教程中查看。
										</div>
									</div>
								</a>
							</div>
							<div class="doc-article-item all experience-local hidden">
								<a href="https://www.mindspore.cn/docs/programming_guide/zh-CN/master/quick_start/quick_video/mindInsight_performance_profiling.html" class="article-link">
									<div>
										<div class="doc-article-head">
											<span class="doc-head-content">MindInsight性能调试</span>
										</div>
										<div class="doc-article-desc">
										 本文档中包含手把手系列视频，码云Gitee不支持展示，请于官方网站对应教程中查看。
										</div>
									</div>
								</a>
							</div>
							<div class="doc-article-item all experience-local hidden">
								<a href="https://www.mindspore.cn/docs/programming_guide/zh-CN/master/quick_start/quick_video/mindspore_lite_converter.html" class="article-link">
									<div>
										<div class="doc-article-head">
											<span class="doc-head-content">MindSpore Lite转换工具converter</span>
										</div>
										<div class="doc-article-desc">
										 本文档中包含手把手系列视频，码云Gitee不支持展示，请于官方网站对应教程中查看。
										</div>
									</div>
								</a>
							</div>
							<div class="doc-article-item all experience-local hidden">
								<a href="https://www.mindspore.cn/docs/programming_guide/zh-CN/master/quick_start/quick_video/mindspore_lite_quantization.html" class="article-link">
									<div>
										<div class="doc-article-head">
											<span class="doc-head-content">MindSpore Lite模型量化</span>
										</div>
										<div class="doc-article-desc">
										 本文档中包含手把手系列视频，码云Gitee不支持展示，请于官方网站对应教程中查看。
										</div>
									</div>
								</a>
							</div>
							<div class="doc-article-item all experience-local hidden">
								<a href="https://www.mindspore.cn/docs/programming_guide/zh-CN/master/quick_start/quick_video/mindspore_lite_quick_start.html" class="article-link">
									<div>
										<div class="doc-article-head">
											<span class="doc-head-content">MindSpore Lite快速体验</span>
										</div>
										<div class="doc-article-desc">
										 本文档中包含手把手系列视频，码云Gitee不支持展示，请于官方网站对应教程中查看。
										</div>
									</div>
								</a>
							</div>
							<div class="doc-article-item all experience-local hidden">
								<a href="https://www.mindspore.cn/docs/programming_guide/zh-CN/master/quick_start/quick_video/network_migration_process.html" class="article-link">
									<div>
										<div class="doc-article-head">
											<span class="doc-head-content">网络迁移流程</span>
										</div>
										<div class="doc-article-desc">
										 本文档中包含手把手系列视频，码云Gitee不支持展示，请于官方网站对应教程中查看。
										</div>
									</div>
								</a>
							</div>
							<div class="doc-article-item all experience-local hidden">
								<a href="https://www.mindspore.cn/docs/programming_guide/zh-CN/master/quick_start/quick_video/network_migration_tuning.html" class="article-link">
									<div>
										<div class="doc-article-head">
											<span class="doc-head-content">网络迁移调优</span>
										</div>
										<div class="doc-article-desc">
										 本文档中包含手把手系列视频，码云Gitee不支持展示，请于官方网站对应教程中查看。
										</div>
									</div>
								</a>
							</div>
							<div class="doc-article-item all experience-local hidden">
								<a href="https://www.mindspore.cn/docs/programming_guide/zh-CN/master/quick_start/quick_video/optimize_data_processing.html" class="article-link">
									<div>
										<div class="doc-article-head">
											<span class="doc-head-content">优化数据处理</span>
										</div>
										<div class="doc-article-desc">
										 本文档中包含手把手系列视频，码云Gitee不支持展示，请于官方网站对应教程中查看。
										</div>
									</div>
								</a>
							</div>
							<div class="doc-article-item all experience-local hidden">
								<a href="https://www.mindspore.cn/docs/programming_guide/zh-CN/master/quick_start/quick_video/quick_start_video.html" class="article-link">
									<div>
										<div class="doc-article-head">
											<span class="doc-head-content">体验快速入门</span>
										</div>
										<div class="doc-article-desc">
										 本文档中包含手把手系列视频，码云Gitee不支持展示，请于官方网站对应教程中查看。
										</div>
									</div>
								</a>
							</div>
							<div class="doc-article-item all experience-local hidden">
								<a href="https://www.mindspore.cn/docs/programming_guide/zh-CN/master/quick_start/quick_video/saving_and_loading_model_parameters.html" class="article-link">
									<div>
										<div class="doc-article-head">
											<span class="doc-head-content">模型参数保存与加载</span>
										</div>
										<div class="doc-article-desc">
										 本文档中包含手把手系列视频，码云Gitee不支持展示，请于官方网站对应教程中查看。
										</div>
									</div>
								</a>
							</div>
							<div class="doc-article-item all os-Linux hardware-Ascend stage-Whole-Process user-Beginner user-Intermediate user-Expert hidden">
								<a href="https://www.mindspore.cn/tutorial/training/zh-CN/r1.2/advanced_use/use_on_the_cloud.html" class="article-link">
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
		
