.. MindSpore documentation master file, created by
   sphinx-quickstart on Thu Aug 17 09:00:00 2020.
   You can adapt this file completely to your liking, but it should at least
   contain the root `toctree` directive.

MindSpore Lite文档
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

   quick_start/one_hour_introduction
   quick_start/quick_start_cpp
   quick_start/quick_start_server_inference_cpp
   quick_start/quick_start_java
   quick_start/quick_start_server_inference_java
   quick_start/quick_start
   quick_start/image_segmentation
   quick_start/train_lenet
   quick_start/train_lenet_java

.. toctree::
   :glob:
   :maxdepth: 1
   :caption: 端侧推理
   :hidden:

   use/converter_tool
   use/post_training_quantization
   use/data_preprocessing
   use/runtime
   use/micro
   use/asic

.. toctree::
   :glob:
   :maxdepth: 1
   :caption: 端侧训练
   :hidden:

   use/converter_train
   use/runtime_train

.. toctree::
   :glob:
   :maxdepth: 1
   :caption: 第三方接入
   :hidden:

   use/register
   use/delegate

.. toctree::
   :glob:
   :maxdepth: 1
   :caption: 其他工具
   :hidden:

   use/benchmark
   use/cropper_tool
   use/visual_tool
   use/obfuscator_tool

.. toctree::
   :glob:
   :maxdepth: 1
   :caption: 参考文档
   :hidden:

   architecture_lite
   operator_list_lite
   operator_list_codegen
   model_lite
   troubleshooting_guide

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
										<button class="doc-filter-btn doc-btn" id="os-mac">Mac</button>
										<button class="doc-filter-btn doc-btn" id="os-Android">Android</button>
										<button class="doc-filter-btn doc-btn" id="os-ios">iOS</button>
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
										<button class="doc-filter-btn doc-btn" id="stage-Model-Code-Generation">模型代码生成</button>
										<button class="doc-filter-btn doc-btn" id="stage-Model-Obfuscator">模型混淆</button>
										<button class="doc-filter-btn doc-btn" id="stage-Inference">推理应用</button>
										<button class="doc-filter-btn doc-btn" id="stage-Benchmark-Testing">基准测试</button>
										<button class="doc-filter-btn doc-btn" id="stage-Static-Library-Cropping">静态库裁剪</button>
										<button class="doc-filter-btn doc-btn" id="stage-Visualization">可视化</button>
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
									<button class="doc-filter-btn doc-btn" id="hardware-NNIE">NNIE</button>
									<button class="doc-filter-btn doc-btn" id="hardware-TensorRT">TensorRT</button>
									<button class="doc-filter-btn doc-btn" id="hardware-Ascend">Ascend</button>
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
						    <div class="doc-article-item all os-Windows os-Linux os-Android os-ios stage-Environment-Preparation user-Beginner user-Intermediate user-Expert hidden">
								<a href="https://www.mindspore.cn/lite/docs/zh-CN/r1.7/use/downloads.html" class="article-link">
									<div>
										<div class="doc-article-head">
											<span class="doc-head-content">下载MindSpore Lite</span>
										</div>
							
										<div class="doc-article-desc">
										欢迎使用MindSpore Lite，我们提供了支持多种操作系统和硬件平台的模型转换、模型推理、图像处理等功能，你可以下载适用于本地环境的版本包直接使用。
										</div>
									</div>
								</a>
							</div>
							<div class="doc-article-item all os-Windows os-Linux os-Android language-cpp stage-Whole-Process stage-Model-Converting stage-Model-Loading stage-Data-Preparation stage-Inference user-Beginner user-Intermediate user-Expert hidden">
								<a href="https://www.mindspore.cn/lite/docs/zh-CN/r1.7/quick_start/one_hour_introduction.html" class="article-link">
									<div>
										<div class="doc-article-head">
											<span class="doc-head-content">一小时入门</span>
										</div>
							
										<div class="doc-article-desc">
										本文通过使用MindSpore Lite对一个模型执行推理为例，向大家介绍MindSpore Lite的基础功能和用法。
										</div>
									</div>
								</a>
							</div>
							<div class="doc-article-item all os-Windows os-Linux os-Android os-mac os-ios stage-Environment-Preparation user-Intermediate user-Expert hidden">
								<a href="https://www.mindspore.cn/lite/docs/zh-CN/r1.7/use/build.html" class="article-link">
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
							<div class="doc-article-item all os-Windows os-Linux language-cpp stage-Whole-Process stage-Inference stage-Data-Preparation user-Beginner hidden">
								<a href="https://www.mindspore.cn/lite/docs/zh-CN/r1.7/quick_start/quick_start_cpp.html" class="article-link">
									<div>
										<div class="doc-article-head">
											<span class="doc-head-content">体验C++极简推理Demo</span>
										</div>
							
										<div class="doc-article-desc">
										本教程提供了MindSpore Lite执行推理的示例程序，通过随机输入、执行推理、打印推理结果的方式，演示了C++进行端侧推理的基本流程，用户能够快速了解MindSpore Lite执行推理相关API的使用。
										</div>
									</div>
								</a>
							</div>
							<div class="doc-article-item all os-Linux language-java stage-Whole-Process stage-Inference stage-Data-Preparation user-Beginner hidden">
								<a href="https://www.mindspore.cn/lite/docs/zh-CN/r1.7/quick_start/quick_start_java.html" class="article-link">
									<div>
										<div class="doc-article-head">
											<span class="doc-head-content">体验Java极简推理Demo</span>
										</div>
							
										<div class="doc-article-desc">
										本教程提供了MindSpore Lite执行推理的示例程序，通过随机输入、执行推理、打印推理结果的方式，演示了利用MindSpore Lite Java API进行端侧推理的基本流程，用户能够快速了解MindSpore Lite执行推理相关Java API的使用。
										</div>
									</div>
								</a>
							</div>
							<div class="doc-article-item all os-Linux os-Android language-cpp stage-Whole-Process stage-Model-Converting stage-Model-Loading stage-Inference stage-Data-Preparation user-Beginner user-Intermediate user-Expert hidden">
								<a href="https://www.mindspore.cn/lite/docs/zh-CN/r1.7/quick_start/quick_start.html" class="article-link">
									<div>
										<div class="doc-article-head">
											<span class="doc-head-content">基于JNI接口的Android应用开发</span>
										</div>
							
										<div class="doc-article-desc">
											本教程从端侧Android图像分类demo入手，帮助用户了解MindSpore Lite应用工程的构建、依赖项配置以及相关API的使用。
										</div>
									</div>
								</a>
							</div>
							<div class="doc-article-item all os-Android language-java stage-Whole-Process stage-Model-Converting stage-Model-Loading stage-Inference stage-Data-Preparation user-Beginner user-Intermediate user-Expert hidden">
								<a href="https://www.mindspore.cn/lite/docs/zh-CN/r1.7/quick_start/image_segmentation.html" class="article-link">
									<div>
										<div class="doc-article-head">
											<span class="doc-head-content">基于Java接口的Android应用开发</span>
										</div>
							
										<div class="doc-article-desc">
											本教程基于MindSpore团队提供的Android“端侧图像分割”示例程序，演示了端侧部署的流程。
										</div>
									</div>
								</a>
							</div>
							<div class="doc-article-item all os-Linux os-Android language-cpp stage-Whole-Process stage-Model-Export stage-Model-Converting stage-Model-Training user-Beginner user-Intermediate user-Expert hidden">
								<a href="https://www.mindspore.cn/lite/docs/zh-CN/r1.7/quick_start/train_lenet.html" class="article-link">
									<div>
										<div class="doc-article-head">
											<span class="doc-head-content">基于C++接口实现端侧训练</span>
										</div>
							
										<div class="doc-article-desc"> 
											本教程基于LeNet训练示例代码，演示MindSpore Lite训练功能的使用。
										</div>
									</div>
								</a>
							</div>
							<div class="doc-article-item all os-Linux os-Android stage-Whole-Process stage-Model-Loading stage-Model-Training stage-Data-Preparation user-Beginner user-Intermediate user-Expert hidden">
								<a href="https://www.mindspore.cn/lite/docs/zh-CN/r1.7/quick_start/train_lenet_java.html" class="article-link">
									<div>
										<div class="doc-article-head">
											<span class="doc-head-content">基于Java接口实现端侧训练</span>
										</div>
							
										<div class="doc-article-desc"> 
										本教程通过构建并部署Java版本的LeNet网络的训练，演示MindSpore Lite端侧训练Java接口的使用。 首先指导您在本地成功训练LeNet模型，然后讲解示例代码。
										</div>
									</div>
								</a>
							</div>
							<div class="doc-article-item all os-Windows os-Linux stage-Model-Converting user-Intermediate user-Expert hidden">
								<a href="https://www.mindspore.cn/lite/docs/zh-CN/r1.7/use/converter_tool.html" class="article-link">
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
							<div class="doc-article-item all os-Linux stage-Model-Converting user-Intermediate user-Expert hidden">
								<a href="https://www.mindspore.cn/lite/docs/zh-CN/r1.7/use/converter_register.html" class="article-link">
									<div>
										<div class="doc-article-head">
											<span class="doc-head-content">离线构建自定义算子</span>
										</div>
										<div class="doc-article-desc">
                      					MindSpore Lite提供一个具有高度灵活性的离线模型转换工具，支持用户基于该工具进行扩展，比如，可将用户特有硬件与MindSpore Lite推理引擎结合。
										</div>
									</div>
								</a>
							</div>
							<div class="doc-article-item all os-Windows os-Linux stage-Model-Converting stage-Model-Optimization user-Intermediate user-Expert hidden">
								<a href="https://www.mindspore.cn/lite/docs/zh-CN/r1.7/use/post_training_quantization.html" class="article-link">
									<div>
										<div class="doc-article-head">
											<span class="doc-head-content">训练后量化</span>
										</div>
										<div class="doc-article-desc">
											对于已经训练好的float32模型，通过训练后量化将其转为int8，不仅能减小模型大小，而且能显著提高推理性能。本教程介绍了模型训练后量化的具体方法。
										</div>
									</div>
								</a>
							</div>
							<div class="doc-article-item all os-Linux os-iot language-cpp stage-Model-Code-Generation stage-Inference user-Beginner user-Intermediate hidden">
								<a href="https://www.mindspore.cn/lite/docs/zh-CN/r1.7/use/micro.html" class="article-link">
									<div>
										<div class="doc-article-head">
											<span class="doc-head-content">在轻量和小型系统上执行推理</span>
										</div>
										<div class="doc-article-desc">
										MindSpore Lite提供代码生成工具codegen，将运行时编译、解释计算图，移至离线编译阶段。仅保留推理所必须的信息，生成极简的推理代码。
										</div>
									</div>
								</a>
							</div>
							<div class="doc-article-item all os-Windows os-Linux language-cpp stage-Data-Preparation user-Beginner user-Intermediate user-Expert hidden">
								<a href="https://www.mindspore.cn/lite/docs/zh-CN/r1.7/use/image_processing.html" class="article-link">
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
							<div class="doc-article-item all os-Windows os-Linux os-Android os-mac os-ios language-cpp stage-Inference stage-Model-Loading stage-Data-Preparation user-Intermediate user-Expert hidden">
								<a href="https://www.mindspore.cn/lite/docs/zh-CN/r1.7/use/runtime_cpp.html" class="article-link">
									<div>
										<div class="doc-article-head">
											<span class="doc-head-content">使用C++接口执行推理</span>
										</div>
										<div class="doc-article-desc">
											通过MindSpore Lite模型转换后，需在Runtime中完成模型的推理执行流程。本教程介绍如何使用C++接口编写推理代码。
										</div>
									</div>
								</a>
							</div>
							<div class="doc-article-item all os-Android language-java stage-Inference stage-Model-Loading stage-Data-Preparation user-Intermediate user-Expert hidden">
								<a href="https://www.mindspore.cn/lite/docs/zh-CN/r1.7/use/runtime_java.html" class="article-link">
									<div>
										<div class="doc-article-head">
											<span class="doc-head-content">使用Java接口执行推理</span>
										</div>
										<div class="doc-article-desc">
											通过MindSpore Lite模型转换后，需在Runtime中完成模型的推理执行流程。本教程介绍如何使用Java接口编写推理代码。
										</div>
									</div>
								</a>
							</div>
							<div class="doc-article-item all hardware-NPU os-Android os-Linux stage-Environment-Preparation user-Intermediate user-Expert hidden">
								<a href="https://www.mindspore.cn/lite/docs/zh-CN/r1.7/use/npu_info.html" class="article-link">
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
							<div class="doc-article-item all hardware-NNIE os-Linux stage-Environment-Preparation user-Intermediate user-Expert hidden">
								<a href="https://www.mindspore.cn/lite/docs/zh-CN/r1.7/use/nnie.html" class="article-link">
									<div>
										<div class="doc-article-head">
											<span class="doc-head-content">集成NNIE使用说明</span>
										</div>
										<div class="doc-article-desc">
											该教程介绍了集成NNIE的使用说明。
										</div>
									</div>
								</a>
							</div>
							<div class="doc-article-item all hardware-TensorRT os-Linux stage-Environment-Preparation user-Intermediate user-Expert hidden">
								<a href="https://www.mindspore.cn/lite/docs/zh-CN/r1.7/use/tensorrt_info.html" class="article-link">
									<div>
										<div class="doc-article-head">
											<span class="doc-head-content">集成TensorRT使用说明</span>
										</div>
										<div class="doc-article-desc">
											该教程介绍了集成TensorRT的使用说明。
										</div>
									</div>
								</a>
							</div>  
                            <div class="doc-article-item all os-Windows os-Linux os-Android language-cpp stage-Inference user-Expert hidden">
								<a href="https://www.mindspore.cn/lite/docs/zh-CN/r1.7/use/register_kernel.html" class="article-link">
									<div>
										<div class="doc-article-head">
											<span class="doc-head-content">在线构建自定义算子</span>
										</div>
										<div class="doc-article-desc">
											MindSpore Lite当前提供了一套南向算子的注册机制，南向算子可以理解为用户自己的算子实现，如果用户想通过MindSpore Lite框架调度到自己的算子实现上，可参考本文。
										</div>
									</div>
								</a>
							</div>
							<div class="doc-article-item all os-Linux user-Expert hidden">
								<a href="https://www.mindspore.cn/lite/docs/zh-CN/r1.7/use/delegate.html" class="article-link">
									<div>
										<div class="doc-article-head">
											<span class="doc-head-content">使用Delegate支持第三方AI框架接入</span>
										</div>
										<div class="doc-article-desc">
											如果用户想通过MindSpore Lite框架调度到其他框架的推理流程，可参考本文。
										</div>
									</div>
								</a>
							</div>
							<div class="doc-article-item all os-Linux os-Windows stage-Environment-Preparation stage-Benchmark-Testing user-Intermediate user-Expert hidden">
								<a href="https://www.mindspore.cn/lite/docs/zh-CN/r1.7/use/benchmark_tool.html" class="article-link">
									<div>
										<div class="doc-article-head">
											<span class="doc-head-content">benchmark</span>
										</div>
										<div class="doc-article-desc">
											转换模型后执行推理前，你可以使用Benchmark工具对MindSpore Lite模型进行基准测试。它不仅可以对MindSpore Lite模型前向推理执行耗时进行定量分析（性能），还可以通过指定模型输出进行可对比的误差分析（精度）。
										</div>
									</div>
								</a>
							</div>
							<div class="doc-article-item all os-Linux stage-Environment-Preparation stage-Static-Library-Cropping user-Intermediate user-Expert hidden">
								<a href="https://www.mindspore.cn/lite/docs/zh-CN/r1.7/use/cropper_tool.html" class="article-link">
									<div>
										<div class="doc-article-head">
											<span class="doc-head-content">静态库裁剪工具</span>
										</div>
										<div class="doc-article-desc">
											MindSpore Lite提供对Runtime的libmindspore-lite.a静态库裁剪工具，能够筛选出ms模型中存在的算子，对静态库文件进行裁剪，有效降低库文件大小。
										</div>
									</div>
								</a>
							</div>
							<div class="doc-article-item all os-Linux stage-Environment-Preparation stage-Model-Export stage-Model-Converting user-Intermediate user-Expert hidden">
								<a href="https://www.mindspore.cn/lite/docs/zh-CN/r1.7/use/converter_train.html" class="article-link">
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
								<a href="https://www.mindspore.cn/lite/docs/zh-CN/r1.7/use/runtime_train_cpp.html" class="article-link">
									<div>
										<div class="doc-article-head">
											<span class="doc-head-content">使用C++接口执行训练</span>
										</div>
										<div class="doc-article-desc">
											通过MindSpore Lite模型转换后，需在Runtime中完成模型的训练执行流程。本教程介绍如何使用C++接口编写训练代码。
										</div>
									</div>
								</a>
							</div>
							<div class="doc-article-item all os-Linux os-Android language-java stage-Model-Training stage-Model-Loading stage-Data-Preparation user-Intermediate user-Expert hidden">
								<a href="https://www.mindspore.cn/lite/docs/zh-CN/r1.7/use/runtime_train_java.html" class="article-link">
									<div>
										<div class="doc-article-head">
											<span class="doc-head-content">使用Java接口执行训练</span>
										</div>
										<div class="doc-article-desc">
											通过MindSpore Lite模型转换后，需在Runtime中完成模型的训练执行流程。本教程介绍如何使用Java接口编写训练代码。
										</div>
									</div>
								</a>
							</div>
							<div class="doc-article-item all os-Linux hardware-Ascend stage-Environment-Preparation user-Intermediate user-Expert hidden">
								<a href="https://www.mindspore.cn/lite/docs/zh-CN/r1.7/use/ascend_info.html" class="article-link">
									<div>
										<div class="doc-article-head">
											<span class="doc-head-content">集成Ascend使用说明</span>
										</div>
										<div class="doc-article-desc">
										本文档介绍如何在Ascend环境的Linux系统上，使用MindSpore Lite 进行推理，以及动态shape功能的使用。
										</div>
									</div>
								</a>
							</div>
							<div class="doc-article-item all os-Linux stage-Data-Preparation stage-Benchmark-Testing user-Intermediate user-Expert hidden">
								<a href="https://www.mindspore.cn/lite/docs/zh-CN/r1.7/use/benchmark_train_tool.html" class="article-link">
									<div>
										<div class="doc-article-head">
											<span class="doc-head-content">benchmark_train</span>
										</div>
										<div class="doc-article-desc">
										与benchmark工具类似，MindSpore端侧训练为你提供了benchmark_train工具对训练后的模型进行基准测试。它不仅可以对模型前向推理执行耗时进行定量分析（性能），还可以通过指定模型输出进行可对比的误差分析（精度）。
										</div>
									</div>
								</a>
							</div>
							<div class="doc-article-item all os-Windows os-Linux os-mac stage-Visualization">
								<a href="https://www.mindspore.cn/lite/docs/zh-CN/r1.7/use/visual_tool.html" class="article-link">
									<div>
										<div class="doc-article-head">
											<span class="doc-head-content">可视化工具</span>
										</div>
										<div class="doc-article-desc">
										Netron是一个基于Electron平台开发的神经网络模型可视化工具，支持MindSpore Lite模型，可以方便地查看模型信息。
										</div>
									</div>
								</a>
							</div>
							<div class="doc-article-item all os-Linux stage-Model-Obfuscator user-Expert">
								<a href="https://www.mindspore.cn/lite/docs/zh-CN/r1.7/use/obfuscator_tool.html" class="article-link">
									<div>
										<div class="doc-article-head">
											<span class="doc-head-content">模型混淆工具</span>
										</div>
										<div class="doc-article-desc">
										MindSpore Lite提供一个轻量级的离线模型混淆工具，可用于保护IOT或端侧设备上部署的模型文件的机密性。
										</div>
									</div>
								</a>
							</div>
						</div>
				
					</div>
					
				</div>
			</div>
		</div>
		
