.. MindSpore documentation master file, created by
   sphinx-quickstart on Thu Aug 17 09:00:00 2020.
   You can adapt this file completely to your liking, but it should at least
   contain the root `toctree` directive.

MindSpore Lite文档
=================================

.. toctree::
    :glob:
    :maxdepth: 1
    :caption: 快速入门
    :hidden:

    quick_start/one_hour_introduction

.. toctree::
    :glob:
    :maxdepth: 1
    :caption: 编译
    :hidden:

    build/build

.. toctree::
   :glob:
   :maxdepth: 1
   :caption: 模型转换
   :hidden:

   converter/converter_tool

.. toctree::
   :glob:
   :maxdepth: 1
   :caption: 模型推理
   :hidden:

   infer/runtime_cpp
   infer/runtime_java
   infer/device_infer_example

.. toctree::
   :glob:
   :maxdepth: 1
   :caption: MindIR离线推理
   :hidden:

   mindir/build
   mindir/runtime
   mindir/runtime_parallel
   mindir/runtime_distributed
   mindir/converter
   mindir/benchmark

.. toctree::
   :glob:
   :maxdepth: 1
   :caption: 端侧训练
   :hidden:

   train/converter_train
   train/runtime_train
   train/device_train_example

.. toctree::
   :glob:
   :maxdepth: 1
   :caption: 高阶开发
   :hidden:

   advanced/image_processing
   advanced/quantization
   advanced/micro
   advanced/third_party

.. toctree::
   :glob:
   :maxdepth: 1
   :caption: 工具
   :hidden:

   tools/visual_tool
   tools/benchmark
   tools/cropper_tool
   tools/obfuscator_tool
   tools/benchmark_golden_data

.. toctree::
   :glob:
   :maxdepth: 1
   :caption: 参考文档
   :hidden:

   reference/architecture_lite
   reference/operator_list_lite
   reference/operator_list_codegen
   reference/model_lite
   reference/faq
   reference/log

.. toctree::
   :maxdepth: 1
   :caption: RELEASE NOTES
   :hidden:

   RELEASE

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
										<button class="doc-filter-btn doc-btn" id="os-Android">Android</button>
										<button class="doc-filter-btn doc-btn" id="os-Linux">Linux</button>
										<button class="doc-filter-btn doc-btn" id="os-Windows">Windows</button>
										<button class="doc-filter-btn doc-btn" id="os-ios">iOS</button>
										<button class="doc-filter-btn doc-btn" id="os-ohos">OpenHarmony</button>
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
										<button class="doc-filter-btn doc-btn" id="language-c">C</button>
										<button class="doc-filter-btn doc-btn" id="language-python">Python</button>
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
						    <div class="doc-article-item all os-Windows os-Linux os-Android os-ios os-ohos os-iot stage-Environment-Preparation user-Beginner user-Intermediate user-Expert hidden">
								<a href="https://www.mindspore.cn/lite/docs/zh-CN/r2.6.0rc1/use/downloads.html" class="article-link">
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
								<a href="https://www.mindspore.cn/lite/docs/zh-CN/r2.6.0rc1/quick_start/one_hour_introduction.html" class="article-link">
									<div>
										<div class="doc-article-head">
											<span class="doc-head-content">端侧推理快速入门</span>
										</div>
										<div class="doc-article-desc">
										本文通过使用MindSpore Lite对一个模型执行端侧推理为例，向大家介绍MindSpore Lite的基础功能和用法。
										</div>
									</div>
								</a>
							</div>
							<div class="doc-article-item all os-Windows os-Linux os-Android os-ios os-ohos stage-Environment-Preparation user-Intermediate user-Expert hidden">
								<a href="https://www.mindspore.cn/lite/docs/zh-CN/r2.6.0rc1/build/build.html" class="article-link">
									<div>
										<div class="doc-article-head">
											<span class="doc-head-content">端侧编译</span>
										</div>
										<div class="doc-article-desc">
										本章节介绍如何快速编译出MindSpore Lite。
										</div>
									</div>
								</a>
							</div>
							<div class="doc-article-item all os-Linux stage-Environment-Preparation user-Intermediate user-Expert hidden">
								<a href="https://www.mindspore.cn/lite/docs/zh-CN/r2.6.0rc1/mindir/build.html" class="article-link">
									<div>
										<div class="doc-article-head">
											<span class="doc-head-content">编译云侧MindSpore Lite</span>
										</div>
										<div class="doc-article-desc">
										本章节介绍如何快速编译出云侧MindSpore Lite。
										</div>
									</div>
								</a>
							</div>
                            <div class="doc-article-item all os-Linux stage-Environment-Preparation user-Intermediate user-Expert hidden">
								<a href="https://www.mindspore.cn/lite/docs/zh-CN/r2.6.0rc1/mindir/converter_tool_ascend.html" class="article-link">
									<div>
										<div class="doc-article-head">
											<span class="doc-head-content">Ascend转换工具功能说明</span>
										</div>
										<div class="doc-article-desc">
										本文档介绍云侧推理模型转换工具在Ascend后端的相关功能，如配置文件的选项、动态shape、AOE、自定义算子等。
										</div>
									</div>
								</a>
							</div>
                            <div class="doc-article-item all os-Linux stage-Environment-Preparation user-Intermediate user-Expert hidden">
								<a href="https://www.mindspore.cn/lite/docs/zh-CN/r2.6.0rc1/mindir/converter_tool_graph_kernel.html" class="article-link">
									<div>
										<div class="doc-article-head">
											<span class="doc-head-content">图算融合配置说明（beta特性）</span>
										</div>
										<div class="doc-article-desc">
										图算融合是MindSpore特有的网络性能优化技术。它可以通过自动分析和优化现有网络计算图逻辑，并结合目标硬件能力，对计算图进行计算化简和替代、算子拆分和融合、算子特例化编译等优化，以提升设备计算资源利用率，实现对网络性能的整体优化。
										</div>
									</div>
								</a>
							</div>
							<div class="doc-article-item all os-Windows os-Linux language-cpp stage-Whole-Process stage-Inference stage-Data-Preparation user-Beginner hidden">
								<a href="https://www.mindspore.cn/lite/docs/zh-CN/r2.6.0rc1/infer/quick_start_cpp.html" class="article-link">
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
								<a href="https://www.mindspore.cn/lite/docs/zh-CN/r2.6.0rc1/infer/quick_start_java.html" class="article-link">
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
							<div class="doc-article-item all os-Windows os-Linux language-c stage-Whole-Process stage-Inference stage-Data-Preparation user-Beginner hidden">
								<a href="https://www.mindspore.cn/lite/docs/zh-CN/r2.6.0rc1/infer/quick_start_c.html" class="article-link">
									<div>
										<div class="doc-article-head">
											<span class="doc-head-content">体验C语言极简推理Demo</span>
										</div>
							
										<div class="doc-article-desc">
										本教程提供了MindSpore Lite执行推理的示例程序，通过随机输入、执行推理、打印推理结果的方式，演示了C语言进行端侧推理的基本流程，用户能够快速了解MindSpore Lite执行推理相关API的使用。
										</div>
									</div>
								</a>
							</div>
							<div class="doc-article-item all os-Linux os-Android language-cpp stage-Whole-Process stage-Model-Converting stage-Model-Loading stage-Inference stage-Data-Preparation user-Beginner user-Intermediate user-Expert hidden">
								<a href="https://www.mindspore.cn/lite/docs/zh-CN/r2.6.0rc1/infer/quick_start.html" class="article-link">
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
								<a href="https://www.mindspore.cn/lite/docs/zh-CN/r2.6.0rc1/infer/image_segmentation.html" class="article-link">
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
							<div class="doc-article-item all os-Windows os-Linux stage-Model-Converting stage-Model-Optimization user-Intermediate user-Expert hidden">
								<a href="https://www.mindspore.cn/lite/docs/zh-CN/r2.6.0rc1/advanced/quantization.html" class="article-link">
									<div>
										<div class="doc-article-head">
											<span class="doc-head-content">量化</span>
										</div>
										<div class="doc-article-desc">
										对于已经训练好的float32模型，通过训练后量化将其转为int8，不仅能减小模型大小，而且能显著提高推理性能。
										</div>
									</div>
								</a>
							</div>
							<div class="doc-article-item all os-Windows os-Linux language-cpp stage-Data-Preparation user-Beginner user-Intermediate user-Expert hidden">
								<a href="https://www.mindspore.cn/lite/docs/zh-CN/r2.6.0rc1/advanced/image_processing.html" class="article-link">
									<div>
										<div class="doc-article-head">
											<span class="doc-head-content">数据预处理</span>
										</div>
							
										<div class="doc-article-desc">
										此处是通过创建LiteMat对象，在推理前对图像数据进行处理，达到模型推理所需要的数据格式要求。
										</div>
									</div>
								</a>
							</div>
							<div class="doc-article-item all os-Windows os-Linux os-Android os-ios language-cpp stage-Inference stage-Model-Loading stage-Data-Preparation user-Intermediate user-Expert hidden">
								<a href="https://www.mindspore.cn/lite/docs/zh-CN/r2.6.0rc1/infer/runtime_cpp.html" class="article-link">
									<div>
										<div class="doc-article-head">
											<span class="doc-head-content">模型推理（C++接口）</span>
										</div>
							
										<div class="doc-article-desc">
										通过MindSpore Lite模型转换工具转换成.ms模型后，即可在Runtime中执行模型的推理流程。本教程介绍如何使用C++接口执行推理。
										</div>
									</div>
								</a>
							</div>
							<div class="doc-article-item all os-Android language-java stage-Inference stage-Model-Loading stage-Data-Preparation user-Intermediate user-Expert hidden">
								<a href="https://www.mindspore.cn/lite/docs/zh-CN/r2.6.0rc1/infer/runtime_java.html" class="article-link">
									<div>
										<div class="doc-article-head">
											<span class="doc-head-content">模型推理（Java接口）</span>
										</div>
							
										<div class="doc-article-desc">
										通过MindSpore Lite模型转换工具转换成.ms模型后，即可在Runtime中执行模型的推理流程。本教程介绍如何使用JAVA接口执行推理。
										</div>
									</div>
								</a>
							</div>
							<div class="doc-article-item all os-Linux os-iot language-cpp stage-Model-Code-Generation stage-Inference user-Beginner user-Intermediate hidden">
								<a href="https://www.mindspore.cn/lite/docs/zh-CN/r2.6.0rc1/advanced/micro.html" class="article-link">
									<div>
										<div class="doc-article-head">
											<span class="doc-head-content">在MCU或小型系统上执行推理</span>
										</div>
										<div class="doc-article-desc">
										MindSpore Lite为IOT场景提供了超轻量Micro AI部署解决方案，该方案将模型生成为简单算子调用的纯c代码，不再需要在线解析模型及图编译，适用内存及算力受限的环境。
										</div>
									</div>
								</a>
							</div>
							<div class="doc-article-item all hardware-NPU os-Android os-Linux stage-Environment-Preparation user-Intermediate user-Expert hidden">
								<a href="https://www.mindspore.cn/lite/docs/zh-CN/r2.6.0rc1/advanced/third_party/npu_info.html" class="article-link">
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
							<div class="doc-article-item all hardware-TensorRT os-Linux stage-Environment-Preparation user-Intermediate user-Expert hidden">
								<a href="https://www.mindspore.cn/lite/docs/zh-CN/r2.6.0rc1/advanced/third_party/tensorrt_info.html" class="article-link">
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
							<div class="doc-article-item all os-Linux os-Android hardware-Ascend stage-Environment-Preparation stage-Data-Preparation user-Beginner user-Intermediate user-Expert hidden">
								<a href="https://www.mindspore.cn/lite/docs/zh-CN/r2.6.0rc1/advanced/third_party/ascend_info.html" class="article-link">
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
							<div class="doc-article-item all os-Linux os-Android language-cpp stage-Environment-Preparation stage-Model-Export stage-Model-Converting stage-Model-Loading stage-Model-Training stage-Data-Preparation stage-Inference user-Beginner user-Intermediate user-Expert hidden">
								<a href="https://www.mindspore.cn/lite/docs/zh-CN/r2.6.0rc1/train/train_lenet.html" class="article-link">
									<div>
										<div class="doc-article-head">
											<span class="doc-head-content">基于C++接口实现端侧训练</span>
										</div>
							
										<div class="doc-article-desc">
										本教程基于LeNet训练示例代码，演示在Android设备上训练一个LeNet。
										</div>
									</div>
								</a>
							</div>
							<div class="doc-article-item all os-Linux os-Android language-java stage-Environment-Preparation stage-Model-Converting stage-Model-Loading stage-Model-Training stage-Data-Preparation stage-Inference user-Beginner user-Intermediate user-Expert hidden">
								<a href="https://www.mindspore.cn/lite/docs/zh-CN/r2.6.0rc1/train/train_lenet_java.html" class="article-link">
									<div>
										<div class="doc-article-head">
											<span class="doc-head-content">基于Java接口实现端侧训练</span>
										</div>
							
										<div class="doc-article-desc">
										本教程通过构建并部署Java版本的LeNet网络的训练，演示MindSpore Lite端侧训练Java接口的使用。
										</div>
									</div>
								</a>
							</div>
							<div class="doc-article-item all os-Linux os-Android language-cpp stage-Model-Training stage-Model-Loading stage-Data-Preparation user-Intermediate user-Expert hidden">
								<a href="https://www.mindspore.cn/lite/docs/zh-CN/r2.6.0rc1/train/runtime_train_cpp.html" class="article-link">
									<div>
										<div class="doc-article-head">
											<span class="doc-head-content">端侧训练（C++接口）</span>
										</div>
										<div class="doc-article-desc">
										通过MindSpore Lite模型转换后，需在Runtime中完成模型的训练执行流程。本教程介绍如何使用C++接口编写训练代码。
										</div>
									</div>
								</a>
							</div>
							<div class="doc-article-item all os-Linux os-Android language-java stage-Model-Training stage-Model-Loading stage-Data-Preparation user-Intermediate user-Expert hidden">
								<a href="https://www.mindspore.cn/lite/docs/zh-CN/r2.6.0rc1/train/runtime_train_java.html" class="article-link">
									<div>
										<div class="doc-article-head">
											<span class="doc-head-content">端侧训练（Java接口）</span>
										</div>
										<div class="doc-article-desc">
										通过MindSpore Lite模型转换后，需在Runtime中完成模型的训练执行流程。本教程介绍如何使用Java接口编写训练代码。
										</div>
									</div>
								</a>
							</div>
							<div class="doc-article-item all os-Linux language-cpp stage-Model-Converting user-Intermediate user-Expert hidden">
								<a href="https://www.mindspore.cn/lite/docs/zh-CN/r2.6.0rc1/advanced/third_party/converter_register.html" class="article-link">
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
							<div class="doc-article-item all os-Linux language-cpp stage-Inference user-Expert hidden">
								<a href="https://www.mindspore.cn/lite/docs/zh-CN/r2.6.0rc1/advanced/third_party/register_kernel.html" class="article-link">
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
							<div class="doc-article-item all os-Linux language-cpp user-Expert hidden">
								<a href="https://mindspore.cn/lite/docs/zh-CN/r2.6.0rc1/advanced/third_party/delegate.html" class="article-link">
									<div>
										<div class="doc-article-head">
											<span class="doc-head-content">使用Delegate支持第三方AI框架接入（端上）</span>
										</div>
										<div class="doc-article-desc">
										如果用户想通过MindSpore Lite框架调度到其他框架的推理流程，可参考本文。
										</div>
									</div>
								</a>
							</div>
							<div class="doc-article-item all os-Windows os-Linux stage-Environment-Preparation stage-Data-Preparation stage-Inference user-Beginner user-Intermediate user-Expert hidden">
								<a href="https://www.mindspore.cn/lite/docs/zh-CN/r2.6.0rc1/converter/converter_tool.html" class="article-link">
									<div>
										<div class="doc-article-head">
											<span class="doc-head-content">端侧模型转换</span>
										</div>
							
										<div class="doc-article-desc">
										MindSpore Lite提供离线转换模型功能的工具，支持多种类型的模型转换，转换后的模型可用于推理。
										</div>
									</div>
								</a>
							</div>
							<div class="doc-article-item all os-Linux stage-Environment-Preparation stage-Model-Export stage-Model-Converting user-Intermediate user-Expert hidden">
								<a href="https://www.mindspore.cn/lite/docs/zh-CN/r2.6.0rc1/train/converter_train.html" class="article-link">
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
							<div class="doc-article-item all os-Linux os-Windows stage-Environment-Preparation stage-Benchmark-Testing user-Intermediate user-Expert hidden">
								<a href="https://www.mindspore.cn/lite/docs/zh-CN/r2.6.0rc1/tools/benchmark_tool.html" class="article-link">
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
							<div class="doc-article-item all os-Linux stage-Data-Preparation stage-Benchmark-Testing user-Intermediate user-Expert hidden">
								<a href="https://www.mindspore.cn/lite/docs/zh-CN/r2.6.0rc1/tools/benchmark_train_tool.html" class="article-link">
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
							<div class="doc-article-item all os-Linux stage-Environment-Preparation stage-Static-Library-Cropping user-Intermediate user-Expert hidden">
								<a href="https://www.mindspore.cn/lite/docs/zh-CN/r2.6.0rc1/tools/cropper_tool.html" class="article-link">
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
							<div class="doc-article-item all os-Windows os-Linux stage-Visualization">
								<a href="https://www.mindspore.cn/lite/docs/zh-CN/r2.6.0rc1/tools/visual_tool.html" class="article-link">
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
								<a href="https://www.mindspore.cn/lite/docs/zh-CN/r2.6.0rc1/tools/obfuscator_tool.html" class="article-link">
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
							<div class="doc-article-item all os-Linux language-cpp stage-Whole-Process stage-Inference stage-Data-Preparation user-Beginner hidden">
								<a href="https://www.mindspore.cn/lite/docs/zh-CN/r2.6.0rc1/mindir/runtime_cpp.html" class="article-link">
									<div>
										<div class="doc-article-head">
											<span class="doc-head-content">使用C++接口执行云侧推理</span>
										</div>
							
										<div class="doc-article-desc">
										本教程介绍如何使用C++接口执行云侧推理。
										</div>
									</div>
								</a>
							</div>
							<div class="doc-article-item all os-Linux language-java stage-Whole-Process stage-Inference stage-Data-Preparation user-Beginner hidden">
								<a href="https://www.mindspore.cn/lite/docs/zh-CN/r2.6.0rc1/mindir/runtime_java.html" class="article-link">
									<div>
										<div class="doc-article-head">
											<span class="doc-head-content">使用Java接口执行云侧推理</span>
										</div>
							
										<div class="doc-article-desc">
										本教程介绍如何使用Java接口执行云侧推理。
										</div>
									</div>
								</a>
							</div>
							<div class="doc-article-item all os-Linux language-python stage-Whole-Process stage-Inference stage-Data-Preparation user-Beginner hidden">
								<a href="https://www.mindspore.cn/lite/docs/zh-CN/r2.6.0rc1/mindir/runtime_python.html" class="article-link">
									<div>
										<div class="doc-article-head">
											<span class="doc-head-content">使用Python接口执行云侧推理</span>
										</div>
							
										<div class="doc-article-desc">
										本教程介绍如何使用Python接口执行云侧推理。
										</div>
									</div>
								</a>
							</div>
							<div class="doc-article-item all os-Linux language-cpp stage-Inference stage-Model-Loading stage-Data-Preparation user-Intermediate user-Expert hidden">
								<a href="https://www.mindspore.cn/lite/docs/zh-CN/r2.6.0rc1/mindir/runtime_parallel_cpp.html" class="article-link">
									<div>
										<div class="doc-article-head">
											<span class="doc-head-content">使用C++接口执行并发推理</span>
										</div>
							
										<div class="doc-article-desc">
										本教程介绍如何使用C++接口执行多model并发推理。
										</div>
									</div>
								</a>
							</div>
							<div class="doc-article-item all os-Linux language-java stage-Inference stage-Model-Loading stage-Data-Preparation user-Intermediate user-Expert hidden">
								<a href="https://www.mindspore.cn/lite/docs/zh-CN/r2.6.0rc1/mindir/runtime_parallel_java.html" class="article-link">
									<div>
										<div class="doc-article-head">
											<span class="doc-head-content">使用Java接口执行并发推理</span>
										</div>
							
										<div class="doc-article-desc">
										本教程介绍如何使用Java接口执行多model并发推理。
										</div>
									</div>
								</a>
							</div>
							<div class="doc-article-item all os-Linux language-python stage-Inference stage-Model-Loading stage-Data-Preparation user-Intermediate user-Expert hidden">
								<a href="https://www.mindspore.cn/lite/docs/zh-CN/r2.6.0rc1/mindir/runtime_parallel_python.html" class="article-link">
									<div>
										<div class="doc-article-head">
											<span class="doc-head-content">使用Python接口执行并发推理</span>
										</div>
							
										<div class="doc-article-desc">
										本教程介绍如何使用Python接口执行多model并发推理。
										</div>
									</div>
								</a>
							</div>
							<div class="doc-article-item all os-Linux stage-Environment-Preparation stage-Data-Preparation stage-Inference user-Beginner user-Intermediate user-Expert hidden">
								<a href="https://www.mindspore.cn/lite/docs/zh-CN/r2.6.0rc1/mindir/converter_tool.html" class="article-link">
									<div>
										<div class="doc-article-head">
											<span class="doc-head-content">推理模型离线转换</span>
										</div>
							
										<div class="doc-article-desc">
										MindSpore Lite提供云侧离线转换模型功能的工具，支持多种类型的模型转换，转换后的模型可用于推理。
										</div>
									</div>
								</a>
							</div>
							<div class="doc-article-item all os-Linux language-python stage-Environment-Preparation stage-Model-Export stage-Model-Converting user-Intermediate user-Expert hidden">
								<a href="https://mindspore.cn/lite/docs/zh-CN/r2.6.0rc1/mindir/converter_python.html" class="article-link">
									<div>
										<div class="doc-article-head">
											<span class="doc-head-content">使用Python接口模型转换</span>
										</div>
										<div class="doc-article-desc">
										MindSpore Lite支持通过Python接口进行云侧模型转换，支持多种类型的模型转换，转换后的模型可用于推理。
										</div>
									</div>
								</a>
							</div>
							<div class="doc-article-item all os-Linux os-Windows stage-Environment-Preparation stage-Benchmark-Testing user-Intermediate user-Expert hidden">
								<a href="https://www.mindspore.cn/lite/docs/zh-CN/r2.6.0rc1/mindir/benchmark_tool.html" class="article-link">
									<div>
										<div class="doc-article-head">
											<span class="doc-head-content">benchmark</span>
										</div>
										<div class="doc-article-desc">
										转换模型后执行云侧推理前，你可以使用Benchmark工具对MindSpore Lite模型进行基准测试。它不仅可以对MindSpore Lite模型前向推理执行耗时进行定量分析（性能），还可以通过指定模型输出进行可对比的误差分析（精度）。
										</div>
									</div>
								</a>
							</div>
							<div class="doc-article-item all os-Windows os-Linux os-Android os-ios os-ohos os-iot user-Beginner user-Intermediate user-Expert hidden">
								<a href="https://mindspore.cn/lite/docs/zh-CN/r2.6.0rc1/reference/architecture_lite.html" class="article-link">
									<div>
										<div class="doc-article-head">
											<span class="doc-head-content">总体架构</span>
										</div>
							
										<div class="doc-article-desc">
										MindSpore Lite是一款极速、极智、极简的AI引擎，使能全场景智能应用，为用户提供端到端的解决方案，帮助用户使能AI能力。
										</div>
									</div>
								</a>
							</div>
							<div class="doc-article-item all os-Windows os-Linux os-Android os-ios os-ohos user-Beginner user-Intermediate user-Expert hidden">
								<a href="https://mindspore.cn/lite/docs/zh-CN/r2.6.0rc1/reference/operator_list_lite.html" class="article-link">
									<div>
										<div class="doc-article-head">
											<span class="doc-head-content">Lite算子支持</span>
										</div>
							
										<div class="doc-article-desc">
										本文列举MindSpore Lite支持的算子。
										</div>
									</div>
								</a>
							</div>
							<div class="doc-article-item all os-iot user-Beginner user-Intermediate user-Expert hidden">
								<a href="https://mindspore.cn/lite/docs/zh-CN/r2.6.0rc1/reference/operator_list_codegen.html" class="article-link">
									<div>
										<div class="doc-article-head">
											<span class="doc-head-content">Codegen算子支持</span>
										</div>
							
										<div class="doc-article-desc">
										本文列举MindSpore Lite Codegen支持的算子。
										</div>
									</div>
								</a>
							</div>
							<div class="doc-article-item all os-Android user-Beginner user-Intermediate user-Expert hidden">
								<a href="https://mindspore.cn/lite/docs/zh-CN/r2.6.0rc1/reference/image_classification_lite.html" class="article-link">
									<div>
										<div class="doc-article-head">
											<span class="doc-head-content">图像分类模型</span>
										</div>
							
										<div class="doc-article-desc">
										图像分类模型可以预测图片中出现哪些物体，识别出图片中出现物体列表及其概率。
										</div>
									</div>
								</a>
							</div>
							<div class="doc-article-item all os-Android user-Beginner user-Intermediate user-Expert hidden">
								<a href="https://mindspore.cn/lite/docs/zh-CN/r2.6.0rc1/reference/object_detection_lite.html" class="article-link">
									<div>
										<div class="doc-article-head">
											<span class="doc-head-content">目标检测模型</span>
										</div>
							
										<div class="doc-article-desc">
										目标检测可以识别出图片中的对象和该对象在图片中的位置。
										</div>
									</div>
								</a>
							</div>
							<div class="doc-article-item all os-Android user-Beginner user-Intermediate user-Expert hidden">
								<a href="https://mindspore.cn/lite/docs/zh-CN/r2.6.0rc1/reference/image_segmentation_lite.html" class="article-link">
									<div>
										<div class="doc-article-head">
											<span class="doc-head-content">图像分割模型</span>
										</div>
							
										<div class="doc-article-desc">
										图像分割是用于检测目标在图片中的位置或者图片中某一像素是输入何种对象的。
										</div>
									</div>
								</a>
							</div>
							<div class="doc-article-item all os-Android user-Beginner user-Intermediate user-Expert hidden">
								<a href="https://mindspore.cn/lite/docs/zh-CN/r2.6.0rc1/reference/style_transfer_lite.html" class="article-link">
									<div>
										<div class="doc-article-head">
											<span class="doc-head-content">风格迁移模型</span>
										</div>
							
										<div class="doc-article-desc">
										风格迁移模型可以根据demo内置的标准图片改变用户目标图片的艺术风格，并在App图像预览界面中显示出来。
										</div>
									</div>
								</a>
							</div>
							<div class="doc-article-item all os-Android user-Beginner user-Intermediate user-Expert hidden">
								<a href="https://mindspore.cn/lite/docs/zh-CN/r2.6.0rc1/reference/scene_detection_lite.html" class="article-link">
									<div>
										<div class="doc-article-head">
											<span class="doc-head-content">场景检测模型</span>
										</div>
							
										<div class="doc-article-desc">
										场景检测可以识别设备摄像头中场景的类型。
										</div>
									</div>
								</a>
							</div>
							<div class="doc-article-item all os-Windows os-Linux os-Android os-ios os-ohos os-iot user-Beginner user-Intermediate user-Expert hidden">
								<a href="https://mindspore.cn/lite/docs/zh-CN/r2.6.0rc1/reference/faq.html" class="article-link">
									<div>
										<div class="doc-article-head">
											<span class="doc-head-content">问题定位指南</span>
										</div>
							
										<div class="doc-article-desc">
										在MindSpore Lite使用中遇到问题时，可首先查看日志，多数场景下的问题可以通过日志报错信息直接定位（通过设置环境变量GLOG_v 调整日志等级可以打印更多调试日志），这里简单介绍几种常见报错场景的问题定位与解决方法。
										</div>
									</div>
								</a>
							</div>
							<div class="doc-article-item all os-Windows os-Linux user-Beginner user-Intermediate user-Expert hidden">
								<a href="https://mindspore.cn/lite/docs/zh-CN/r2.6.0rc1/reference/log.html" class="article-link">
									<div>
										<div class="doc-article-head">
											<span class="doc-head-content">日志</span>
										</div>
							
										<div class="doc-article-desc">
										可以通过日志报错信息定位错误。
										</div>
									</div>
								</a>
							</div>
							<div class="doc-article-item all os-Windows os-Linux os-Android os-ios os-ohos os-iot user-Beginner user-Intermediate user-Expert hidden">
								<a href="https://mindspore.cn/lite/docs/zh-CN/r2.6.0rc1/RELEASE.html" class="article-link">
									<div>
										<div class="doc-article-head">
											<span class="doc-head-content">Release Notes</span>
										</div>
							
										<div class="doc-article-desc">
										Release Notes
										</div>
									</div>
								</a>
							</div>
						</div>
				
					</div>
					
				</div>
			</div>
		</div>
  
