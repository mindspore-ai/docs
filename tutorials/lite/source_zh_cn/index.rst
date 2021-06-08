.. MindSpore documentation master file, created by
   sphinx-quickstart on Thu Aug 17 09:00:00 2020.
   You can adapt this file completely to your liking, but it should at least
   contain the root `toctree` directive.

在手机或IoT设备上使用MindSpore
=================================

.. toctree::
    :glob:
    :maxdepth: 1
    :caption: 获取MindSpore Lite
    :hidden:
 
    use/downloads
    use/build
 
.. toctree::
   :glob:
   :maxdepth: 1
   :caption: 快速入门
   :hidden:

   quick_start/quick_start_cpp
   quick_start/quick_start_java
   quick_start/quick_start
   quick_start/image_segmentation
   quick_start/quick_start_codegen
   quick_start/train_lenet
   
.. toctree::
   :glob:
   :maxdepth: 1
   :caption: 端侧推理
   :hidden:

   use/converter_tool
   use/code_generator
   use/post_training_quantization
   use/data_preprocessing
   use/runtime
   use/asic
   use/tools

.. toctree::
   :glob:
   :maxdepth: 1
   :caption: 端侧训练
   :hidden:

   use/converter_train
   use/runtime_train
   use/tools_train

.. toctree::
   :glob:
   :maxdepth: 1
   :caption: 参考文档
   :hidden:

   operator_list_lite
   operator_list_codegen
   model_lite

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
										<div class="doc-os">环境</div>
									</div>
									<div class="col-sm-10 col-sm-pull-1">
										<button class="doc-filter-btn doc-btn" id="os-Windows">Windows</button>
										<button class="doc-filter-btn doc-btn" id="os-Linux">Linux</button>
										<button class="doc-filter-btn doc-btn" id="os-Android">Android</button>
										<button class="doc-filter-btn doc-btn" id="os-iot">IoT</button>
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
									</div>
								</div>
							</div>
							
							<div class="doc-label-choice">
								<div class="row">
								<div class="col-sm-2">
									<div class="doc-hardware">专用芯片</div>
								</div>
								<div class="col-sm-10 col-sm-pull-1">
									<button class="doc-filter-btn doc-btn" id="hardware-NPU">NPU</button>
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
										<button class="doc-filter-btn doc-btn" id="stage-Environment-Preparation">环境准备</button>
										<button class="doc-filter-btn doc-btn" id="stage-Data-Preparation">数据准备</button>
										<button class="doc-filter-btn doc-btn" id="stage-Model-Export">模型导出</button>
										<button class="doc-filter-btn doc-btn" id="stage-Model-Converting">模型转换</button>
										<button class="doc-filter-btn doc-btn" id="stage-Model-Loading">模型加载</button>
										<button class="doc-filter-btn doc-btn" id="stage-Model-Training">模型训练</button>
										<button class="doc-filter-btn doc-btn" id="stage-Model-Optimization">模型调优</button>
										<button class="doc-filter-btn doc-btn" id="stage-Model-Compiling">模型编译</button>
										<button class="doc-filter-btn doc-btn" id="stage-Model-Code-Generation">模型代码生成</button>
										<button class="doc-filter-btn doc-btn" id="stage-Inference">推理应用</button>
										<button class="doc-filter-btn doc-btn" id="stage-Benchmark-Testing">基准测试</button>
										<button class="doc-filter-btn doc-btn" id="stage-Static-Library-Cropping">静态库裁剪</button>
									</div>
								</div>
							</div>

							<div class="doc-label-choice">
								<div class="row">
									<div class="col-sm-2">
										<div class="doc-language">编程语言</div>
									</div>
									<div class="col-sm-10 col-sm-pull-1">
										<button class="doc-filter-btn doc-btn" id="language-cpp">C++</button>
										<button class="doc-filter-btn doc-btn" id="language-java">Java</button>
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
							<div class="doc-article-item all os-Windows os-Linux language-cpp stage-Whole-Process stage-Inference stage-Data-Preparation user-Beginner hidden">
								<a href="https://www.mindspore.cn/tutorial/lite/zh-CN/r1.2/quick_start/quick_start_cpp.html" class="article-link">
									<div>
										<div class="doc-article-head">
											<span class="doc-head-content">体验MindSpore Lite C++ 极简Demo</span>
										</div>
							
										<div class="doc-article-desc">
										本教程提供了MindSpore Lite执行推理的示例程序，通过随机输入、执行推理、打印推理结果的方式，演示了C++进行端侧推理的基本流程，用户能够快速了解MindSpore Lite执行推理相关API的使用。
										</div>
									</div>
								</a>
							</div>
							<div class="doc-article-item all os-Linux language-java stage-Whole-Process stage-Inference stage-Data-Preparation user-Beginner hidden">
								<a href="https://www.mindspore.cn/tutorial/lite/zh-CN/r1.2/quick_start/quick_start_java.html" class="article-link">
									<div>
										<div class="doc-article-head">
											<span class="doc-head-content">体验MindSpore Lite Java极简Demo</span>
										</div>
							
										<div class="doc-article-desc">
										本教程提供了MindSpore Lite执行推理的示例程序，通过随机输入、执行推理、打印推理结果的方式，演示了利用MindSpore Lite Java API进行端侧推理的基本流程，用户能够快速了解MindSpore Lite执行推理相关Java API的使用。
										</div>
									</div>
								</a>
							</div>
							<div class="doc-article-item all os-Linux os-Android language-cpp stage-Whole-Process stage-Model-Converting stage-Model-Loading stage-Inference stage-Data-Preparation user-Beginner user-Intermediate user-Expert hidden">
								<a href="https://www.mindspore.cn/tutorial/lite/zh-CN/r1.2/quick_start/quick_start.html" class="article-link">
									<div>
										<div class="doc-article-head">
											<span class="doc-head-content">实现一个图像分类应用（C++）</span>
										</div>
							
										<div class="doc-article-desc">
											本教程从端侧Android图像分类demo入手，帮助用户了解MindSpore Lite应用工程的构建、依赖项配置以及相关API的使用。
										</div>
									</div>
								</a>
							</div>
							<div class="doc-article-item all os-Android language-java stage-Whole-Process stage-Model-Converting stage-Model-Loading stage-Inference stage-Data-Preparation user-Beginner user-Intermediate user-Expert hidden">
								<a href="https://www.mindspore.cn/tutorial/lite/zh-CN/r1.2/quick_start/image_segmentation.html" class="article-link">
									<div>
										<div class="doc-article-head">
											<span class="doc-head-content">实现一个图像分割应用（Java）</span>
										</div>
							
										<div class="doc-article-desc">
											本教程基于MindSpore团队提供的Android“端侧图像分割”示例程序，演示了端侧部署的流程。
										</div>
									</div>
								</a>
							</div>
							<div class="doc-article-item all os-Linux os-iot language-cpp stage-Whole-Process stage-Model-Compiling stage-Model-Code-Generation stage-Inference user-Beginner user-Intermediate user-Expert hidden">
								<a href="https://www.mindspore.cn/tutorial/lite/zh-CN/r1.2/quick_start/quick_start_codegen.html" class="article-link">
									<div>
										<div class="doc-article-head">
											<span class="doc-head-content">使用CodeGen编译一个MNIST分类模型</span>
										</div>
							
										<div class="doc-article-desc">
										本教程介绍如何使用MindSpore Lite代码生成工具Codegen，快速生成以及部署轻量化推理代码。
										</div>
									</div>
								</a>
							</div>
							<div class="doc-article-item all os-Linux os-Android stage-Whole-Process stage-Model-Export stage-Model-Converting stage-Model-Training user-Beginner user-Intermediate user-Expert hidden">
								<a href="https://www.mindspore.cn/tutorial/lite/zh-CN/r1.2/quick_start/train_lenet.html" class="article-link">
									<div>
										<div class="doc-article-head">
											<span class="doc-head-content">训练一个LeNet模型</span>
										</div>
							
										<div class="doc-article-desc"> 
											本教程基于LeNet训练示例代码，演示MindSpore Lite训练功能的使用。
										</div>
									</div>
								</a>
							</div>
							<div class="doc-article-item all os-Windows os-Linux os-Android stage-Environment-Preparation user-Beginner user-Intermediate user-Expert hidden">
								<a href="https://www.mindspore.cn/tutorial/lite/zh-CN/r1.2/use/downloads.html" class="article-link">
									<div>
										<div class="doc-article-head">
											<span class="doc-head-content">下载MindSpore Lite</span>
										</div>
										<div class="doc-article-desc">
											本教程介绍如何快速下载MindSpore Lite。
										</div>
									</div>
								</a>
							</div>
							
							<div class="doc-article-item all os-Windows os-Linux os-Android stage-Environment-Preparation user-Beginner user-Intermediate user-Expert hidden">
								<a href="https://www.mindspore.cn/tutorial/lite/zh-CN/r1.2/use/build.html" class="article-link">
									<div>
										<div class="doc-article-head">
											<span class="doc-head-content">编译MindSpore Lite</span>
										</div>
										<div class="doc-article-desc">
											本章节介绍如何快速编译出MindSpore Lite。
												
										</div>
									</div>
								</a>
							</div>
							<div class="doc-article-item all os-Windows os-Linux stage-Model-Converting user-Intermediate user-Expert hidden">
								<a href="https://www.mindspore.cn/tutorial/lite/zh-CN/r1.2/use/converter_tool.html" class="article-link">
									<div>
										<div class="doc-article-head">
											<span class="doc-head-content">推理模型转换</span>
										</div>
										<div class="doc-article-desc">
											MindSpore Lite提供离线转换模型功能的工具，支持多种类型的模型转换，转换后的模型可用于推理。
										</div>
									</div>
								</a>
							</div>
							<div class="doc-article-item all os-Linux os-iot language-cpp stage-Model-Code-Generation stage-Inference user-Beginner user-Intermediate hidden">
								<a href="https://www.mindspore.cn/tutorial/lite/zh-CN/r1.2/use/code_generator.html" class="article-link">
									<div>
										<div class="doc-article-head">
											<span class="doc-head-content">代码生成工具</span>
										</div>
										<div class="doc-article-desc">
										MindSpore Lite提供代码生成工具codegen，将运行时编译、解释计算图，移至离线编译阶段。仅保留推理所必须的信息，生成极简的推理代码。
										</div>
									</div>
								</a>
							</div>
							<div class="doc-article-item all os-Windows os-Linux stage-Model-Converting stage-Model-Optimization user-Intermediate user-Expert hidden">
								<a href="https://www.mindspore.cn/tutorial/lite/zh-CN/r1.2/use/post_training_quantization.html" class="article-link">
									<div>
										<div class="doc-article-head">
											<span class="doc-head-content">优化模型(训练后量化)</span>
										</div>
										<div class="doc-article-desc">
											对于已经训练好的float32模型，通过训练后量化将其转为int8，不仅能减小模型大小，而且能显著提高推理性能。本教程介绍了模型训练后量化的具体方法。
										</div>
									</div>
								</a>
							</div>
							<div class="doc-article-item all os-Windows os-Linux language-cpp stage-Data-Preparation user-Beginner user-Intermediate user-Expert hidden">
								<a href="https://www.mindspore.cn/tutorial/lite/zh-CN/r1.2/use/image_processing.html" class="article-link">
									<div>
										<div class="doc-article-head">
											<span class="doc-head-content">预处理图像数据</span>
										</div>
										<div class="doc-article-desc">
											本教程介绍如何通过创建LiteMat对象，在推理前对图像数据进行处理，达到模型推理所需要的数据格式要求。
										</div>
									</div>
								</a>
							</div>
							<div class="doc-article-item all os-Windows os-Linux os-Android language-cpp stage-Inference stage-Model-Loading stage-Data-Preparation user-Intermediate user-Expert hidden">
								<a href="https://www.mindspore.cn/tutorial/lite/zh-CN/r1.2/use/runtime_cpp.html" class="article-link">
									<div>
										<div class="doc-article-head">
											<span class="doc-head-content">使用Runtime执行推理（C++）</span>
										</div>
										<div class="doc-article-desc">
											通过MindSpore Lite模型转换后，需在Runtime中完成模型的推理执行流程。本教程介绍如何使用Java接口编写推理代码。
										</div>
									</div>
								</a>
							</div>
							<div class="doc-article-item all os-Android language-java stage-Inference stage-Model-Loading stage-Data-Preparation user-Intermediate user-Expert hidden">
								<a href="https://www.mindspore.cn/tutorial/lite/zh-CN/r1.2/use/runtime_java.html" class="article-link">
									<div>
										<div class="doc-article-head">
											<span class="doc-head-content">使用Runtime执行推理（Java）</span>
										</div>
										<div class="doc-article-desc">
											通过MindSpore Lite模型转换后，需在Runtime中完成模型的推理执行流程。本教程介绍如何使用C++接口编写推理代码。
										</div>
									</div>
								</a>
							</div>
							<div class="doc-article-item all hardware-NPU os-Android os-Linux stage-Environment-Preparation user-Intermediate user-Expert hidden">
								<a href="https://www.mindspore.cn/tutorial/lite/zh-CN/r1.2/use/npu_info.html" class="article-link">
									<div>
										<div class="doc-article-head">
											<span class="doc-head-content">集成NPU使用说明</span>
										</div>
										<div class="doc-article-desc">
											该教程介绍了集成NPU的使用说明，包含了使用步骤、芯片支持和算子支持。
										</div>
									</div>
								</a>
							</div>
							<div class="doc-article-item all os-Linux os-Windows stage-Environment-Preparation stage-Benchmark-Testing user-Intermediate user-Expert hidden">
								<a href="https://www.mindspore.cn/tutorial/lite/zh-CN/r1.2/use/benchmark_tool.html" class="article-link">
									<div>
										<div class="doc-article-head">
											<span class="doc-head-content">使用Benchmark进行基准测试</span>
										</div>
										<div class="doc-article-desc">
											转换模型后执行推理前，你可以使用Benchmark工具对MindSpore Lite模型进行基准测试。它不仅可以对MindSpore Lite模型前向推理执行耗时进行定量分析（性能），还可以通过指定模型输出进行可对比的误差分析（精度）。
										</div>
									</div>
								</a>
							</div>
							<div class="doc-article-item all os-Linux stage-Environment-Preparation stage-Static-Library-Cropping user-Intermediate user-Expert hidden">
								<a href="https://www.mindspore.cn/tutorial/lite/zh-CN/r1.2/use/cropper_tool.html" class="article-link">
									<div>
										<div class="doc-article-head">
											<span class="doc-head-content">使用裁剪工具降低库文件大小</span>
										</div>
										<div class="doc-article-desc">
											MindSpore Lite提供对Runtime的libmindspore-lite.a静态库裁剪工具，能够筛选出ms模型中存在的算子，对静态库文件进行裁剪，有效降低库文件大小。
										</div>
									</div>
								</a>
							</div>
							<div class="doc-article-item all os-Linux stage-Environment-Preparation stage-Model-Export stage-Model-Converting user-Intermediate user-Expert hidden">
								<a href="https://www.mindspore.cn/tutorial/lite/zh-CN/r1.2/use/converter_train.html" class="article-link">
									<div>
										<div class="doc-article-head">
											<span class="doc-head-content">训练模型转换</span>
										</div>
										<div class="doc-article-desc">
											本教程介绍了如何进行训练模型的转换。
										</div>
									</div>
								</a>
							</div>
							<div class="doc-article-item all os-Linux os-Android language-cpp stage-Model-Training stage-Model-Loading stage-Data-Preparation user-Intermediate user-Expert hidden">
								<a href="https://www.mindspore.cn/tutorial/lite/zh-CN/r1.2/use/runtime_train_cpp.html" class="article-link">
									<div>
										<div class="doc-article-head">
											<span class="doc-head-content">使用Runtime执行训练 (C++)</span>
										</div>
										<div class="doc-article-desc">
											通过MindSpore Lite模型转换后，需在Runtime中完成模型的训练执行流程。本教程介绍如何使用C++接口编写训练代码。
										</div>
									</div>
								</a>
							</div>
							<div class="doc-article-item all os-Linux stage-Data-Preparation stage-Benchmark-Testing user-Intermediate user-Expert hidden">
								<a href="https://www.mindspore.cn/tutorial/lite/zh-CN/r1.2/use/benchmark_train_tool.html" class="article-link">
									<div>
										<div class="doc-article-head">
											<span class="doc-head-content">使用benchmark_train进行基准测试</span>
										</div>
										<div class="doc-article-desc">
										与benchmark工具类似，MindSpore端侧训练为你提供了benchmark_train工具对训练后的模型进行基准测试。它不仅可以对模型前向推理执行耗时进行定量分析（性能），还可以通过指定模型输出进行可对比的误差分析（精度）。
										</div>
									</div>
								</a>
							</div>
						</div>
				
					</div>
					
				</div>
			</div>
		</div>
		
