.. MindSpore documentation master file, created by
   sphinx-quickstart on Thu Aug 17 09:00:00 2020.
   You can adapt this file completely to your liking, but it should at least
   contain the root `toctree` directive.

MindSpore Lite Documentation
=======================================

.. toctree::
    :glob:
    :maxdepth: 1
    :caption: Quick Start
    :hidden:

    quick_start/one_hour_introduction
   
.. toctree::
    :glob:
    :maxdepth: 1
    :caption: Building
    :hidden:

    build/build

.. toctree::
   :glob:
   :maxdepth: 1
   :caption: Model Converter
   :hidden:

   converter/converter_tool

.. toctree::
   :glob:
   :maxdepth: 1
   :caption: Model Inference
   :hidden:

   infer/runtime_cpp
   infer/runtime_java
   infer/device_infer_example

.. toctree::
   :glob:
   :maxdepth: 1
   :caption: MindIR Offline Inference
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
   :caption: Device-side Training
   :hidden:

   train/converter_train
   train/runtime_train
   train/device_train_example

.. toctree::
   :glob:
   :maxdepth: 1
   :caption: Advanced Development
   :hidden:

   advanced/image_processing
   advanced/quantization
   advanced/micro
   advanced/third_party

.. toctree::
   :glob:
   :maxdepth: 1
   :caption: Tools
   :hidden:

   tools/visual_tool
   tools/benchmark
   tools/cropper_tool
   tools/obfuscator_tool
   tools/benchmark_golden_data

.. toctree::
   :glob:
   :maxdepth: 1
   :caption: References
   :hidden:

   reference/architecture_lite
   reference/operator_list_lite
   reference/operator_list_codegen
   reference/model_lite
   reference/faq
   reference/log

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
										<div class="doc-os">Environment</div>
									</div>
									<div class="col-sm-10 col-sm-pull-1 doc-environment-detail">
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
										<div class="doc-user">User</div>
									</div>
									<div class="col-sm-10 col-sm-pull-1">
										<button class="doc-filter-btn doc-btn" id="user-Beginner">Beginner</button>
										<button class="doc-filter-btn doc-btn" id="user-Intermediate">Intermediate</button>
										<button class="doc-filter-btn doc-btn" id="user-Expert">Expert</button>
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
										<button class="doc-filter-btn doc-btn" id="stage-Environment-Preparation">Environment Preparation</button>
										<button class="doc-filter-btn doc-btn" id="stage-Data-Preparation">Data Preparation</button>
										<button class="doc-filter-btn doc-btn" id="stage-Model-Export">Model Export</button>
										<button class="doc-filter-btn doc-btn" id="stage-Model-Converting">Model Converting</button>
										<button class="doc-filter-btn doc-btn" id="stage-Model-Loading">Model Loading</button>
										<button class="doc-filter-btn doc-btn" id="stage-Model-Training">Model Training</button>
										<button class="doc-filter-btn doc-btn" id="stage-Model-Optimization">Model Optimization</button>
										<button class="doc-filter-btn doc-btn" id="stage-Model-Code-Generation">Model Code Generation</button>
										<button class="doc-filter-btn doc-btn" id="stage-Model-Obfuscator">Model Obfuscator</button>
										<button class="doc-filter-btn doc-btn" id="stage-Inference">Inference</button>
										<button class="doc-filter-btn doc-btn" id="stage-Benchmark-Testing">Benchmark Testing</button>
										<button class="doc-filter-btn doc-btn" id="stage-Static-Library-Cropping">Static Library Cropping</button>
										<button class="doc-filter-btn doc-btn" id="stage-Visualization">Visualization</button>
									</div>
								</div>
							</div>

							<div class="doc-label-choice">
								<div class="row">
								<div class="col-sm-2">
									<div class="doc-hardware">Application Specific Integrated Circuit</div>
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
										<div class="doc-language">Programming Language</div>
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
								<a href="https://www.mindspore.cn/lite/docs/en/r2.6.0rc1/use/downloads.html" class="article-link">
									<div>
										<div class="doc-article-head">
											<span class="doc-head-content">Downloading MindSpore Lite</span>
										</div>
							
										<div class="doc-article-desc">
										Welcome to MindSpore Lite. You can download the version package suitable for the local environment and use it directly.
										</div>
									</div>
								</a>
							</div>
							<div class="doc-article-item all os-Windows os-Linux os-Android language-cpp stage-Whole-Process stage-Model-Converting stage-Model-Loading stage-Data-Preparation stage-Inference user-Beginner user-Intermediate user-Expert hidden">
								<a href="https://www.mindspore.cn/lite/docs/en/r2.6.0rc1/quick_start/one_hour_introduction.html" class="article-link">
									<div>
										<div class="doc-article-head">
											<span class="doc-head-content">Quick Start to Device-side Inference</span>
										</div>

										<div class="doc-article-desc">
										This document uses a model inference example to describe how to use basic device-side MindSpore Lite functions.
										</div>
									</div>
								</a>
							</div>
							<div class="doc-article-item all os-Windows os-Linux os-Android os-ios os-ohos stage-Environment-Preparation user-Intermediate user-Expert hidden">
								<a href="https://www.mindspore.cn/lite/docs/en/r2.6.0rc1/build/build.html" class="article-link">
									<div>
										<div class="doc-article-head">
											<span class="doc-head-content">Building Device-side</span>
										</div>
							
										<div class="doc-article-desc">
										This chapter introduces how to quickly compile device-side MindSpore Lite.
										</div>
									</div>
								</a>
							</div>
							<div class="doc-article-item all os-Linux stage-Environment-Preparation user-Intermediate user-Expert hidden">
								<a href="https://www.mindspore.cn/lite/docs/en/r2.6.0rc1/mindir/build.html" class="article-link">
									<div>
										<div class="doc-article-head">
											<span class="doc-head-content">Building Cloud-side MindSpore Lite</span>
										</div>
							
										<div class="doc-article-desc">
										This chapter introduces how to quickly compile cloud-side MindSpore Lite.
										</div>
									</div>
								</a>
							</div>
                            <div class="doc-article-item all os-Linux stage-Environment-Preparation user-Intermediate user-Expert hidden">
								<a href="https://www.mindspore.cn/lite/docs/en/r2.6.0rc1/mindir/converter_tool_ascend.html" class="article-link">
									<div>
										<div class="doc-article-head">
											<span class="doc-head-content">Ascend Conversion Tool Description</span>
										</div>
										<div class="doc-article-desc">
										This article introduces the related features of the cloud-side inference model conversion tool in Ascend back-end, such as profile options, dynamic shape, AOE, custom operators.
										</div>
									</div>
								</a>
							</div>
                            <div class="doc-article-item all os-Linux stage-Environment-Preparation user-Intermediate user-Expert hidden">
								<a href="https://www.mindspore.cn/lite/docs/en/r2.6.0rc1/mindir/converter_tool_graph_kernel.html" class="article-link">
									<div>
										<div class="doc-article-head">
											<span class="doc-head-content">Graph Kernel Fusion Configuration Instructions (Beta Feature)</span>
										</div>
										<div class="doc-article-desc">
										Graph kernel fusion is a unique network performance optimization technique in MindSpore. It can automatically analyze and optimize the existing network computational graph logic and combine with the target hardware capabilities to perform optimizations, such as computational simplification and substitution, operator splitting and fusion, operator special case compilation, to improve the utilization of device computational resources and achieve the overall optimization of network performance.
										</div>
									</div>
								</a>
							</div>
							<div class="doc-article-item all os-Windows os-Linux language-cpp stage-Whole-Process stage-Inference stage-Data-Preparation user-Beginner hidden">
								<a href="https://www.mindspore.cn/lite/docs/en/r2.6.0rc1/infer/quick_start_cpp.html" class="article-link">
									<div>
										<div class="doc-article-head">
											<span class="doc-head-content">Experiencing C++ Simplified Inference Demo</span>
										</div>
							
										<div class="doc-article-desc">
										This tutorial provides a MindSpore Lite inference demo. It demonstrates the basic on-device inference process using C++ by inputting random data, executing inference, and printing the inference result.
										</div>
									</div>
								</a>
							</div>
							<div class="doc-article-item all os-Linux language-java stage-Whole-Process stage-Inference stage-Data-Preparation user-Beginner hidden">
								<a href="https://www.mindspore.cn/lite/docs/en/r2.6.0rc1/infer/quick_start_java.html" class="article-link">
									<div>
										<div class="doc-article-head">
											<span class="doc-head-content">Experiencing Java Simplified Inference Demo</span>
										</div>
							
										<div class="doc-article-desc">
										This tutorial provides an example program for MindSpore Lite to perform inference. It demonstrates the basic process of performing inference on the device side using MindSpore Lite Java interface by random inputting data, executing inference, and printing the inference result. 
										</div>
									</div>
								</a>
							</div>
							<div class="doc-article-item all os-Windows os-Linux language-c stage-Whole-Process stage-Inference stage-Data-Preparation user-Beginner hidden">
								<a href="https://www.mindspore.cn/lite/docs/en/r2.6.0rc1/infer/quick_start_c.html" class="article-link">
									<div>
										<div class="doc-article-head">
											<span class="doc-head-content">Experiencing C-language Simplified Inference Demo</span>
										</div>
							
										<div class="doc-article-desc">
										This tutorial provides a sample program for MindSpore Lite to perform inference, which demonstrates the basic process of end-side inference with C-language by randomly typing, performing inference, and printing inference results, so that users can quickly understand the use of MindSpore Lite to perform inference-related interfaces.
										</div>
									</div>
								</a>
							</div>
							<div class="doc-article-item all os-Linux os-Android language-cpp stage-Whole-Process stage-Model-Converting stage-Model-Loading stage-Inference stage-Data-Preparation user-Beginner user-Intermediate user-Expert hidden">
								<a href="https://www.mindspore.cn/lite/docs/en/r2.6.0rc1/infer/quick_start.html" class="article-link">
									<div>
										<div class="doc-article-head">
											<span class="doc-head-content">Android Application Development Based on JNI Interface</span>
										</div>
							
										<div class="doc-article-desc">
											It is recommended that you start from the image classification demo on the Android device to understand how to build the MindSpore Lite application project, configure dependencies, and use related APIs.
										</div>
									</div>
								</a>
							</div>
							<div class="doc-article-item all os-Android language-java stage-Whole-Process stage-Model-Converting stage-Model-Loading stage-Inference stage-Data-Preparation user-Beginner user-Intermediate user-Expert hidden">
								<a href="https://www.mindspore.cn/lite/docs/en/r2.6.0rc1/infer/image_segmentation.html" class="article-link">
									<div>
										<div class="doc-article-head">
											<span class="doc-head-content">Android Application Development Based on Java Interface</span>
										</div>
							
										<div class="doc-article-desc">
										This tutorial demonstrates the on-device deployment process based on the image segmentation demo on the Android device provided by the MindSpore team.
										</div>
									</div>
								</a>
							</div>
							<div class="doc-article-item all os-Windows os-Linux stage-Model-Converting stage-Model-Optimization user-Intermediate user-Expert hidden">
								<a href="https://www.mindspore.cn/lite/docs/en/r2.6.0rc1/advanced/quantization.html" class="article-link">
									<div>
										<div class="doc-article-head">
											<span class="doc-head-content">Quantization</span>
										</div>
										<div class="doc-article-desc">
										Converting a trained 'float32' model into an 'int8' model through quantization after training can reduce the model size and improve the inference performance.
										This tutorial introduces how to use the function.
										</div>
									</div>
								</a>
							</div>
							<div class="doc-article-item all os-Windows os-Linux language-cpp stage-Data-Preparation user-Beginner user-Intermediate user-Expert hidden">
								<a href="https://www.mindspore.cn/lite/docs/en/r2.6.0rc1/advanced/image_processing.html" class="article-link">
									<div>
										<div class="doc-article-head">
											<span class="doc-head-content">Preprocessing Data</span>
										</div>
										<div class="doc-article-desc">
										This tutorial introduces how to process the image data before inference to meet the data format requirements for model inference by creating a LiteMat object.
										</div>
									</div>
								</a>
							</div>
							<div class="doc-article-item all os-Windows os-Linux os-Android os-ios language-cpp stage-Inference stage-Model-Loading stage-Data-Preparation user-Intermediate user-Expert hidden">
								<a href="https://www.mindspore.cn/lite/docs/en/r2.6.0rc1/infer/runtime_cpp.html" class="article-link">
									<div>
										<div class="doc-article-head">
											<span class="doc-head-content">Model Inference (C++)</span>
										</div>
										<div class="doc-article-desc">
										After model conversion using MindSpore Lite, the model inference process needs to be completed in Runtime. This tutorial introduces how to use C++ interface to write inference code.
										</div>
									</div>
								</a>
							</div>
							<div class="doc-article-item all os-Android language-java stage-Inference stage-Model-Loading stage-Data-Preparation user-Intermediate user-Expert hidden">
								<a href="https://www.mindspore.cn/lite/docs/en/r2.6.0rc1/infer/runtime_java.html" class="article-link">
									<div>
										<div class="doc-article-head">
											<span class="doc-head-content">Model Inference (Java)</span>
										</div>
										<div class="doc-article-desc">
										After model conversion using MindSpore Lite, the model inference process needs to be completed in Runtime. This tutorial introduces how to use Java interface to write inference code.
										</div>
									</div>
								</a>
							</div>
							<div class="doc-article-item all os-Linux os-iot language-cpp stage-Model-Code-Generation stage-Inference user-Beginner user-Intermediate hidden">
								<a href="https://www.mindspore.cn/lite/docs/en/r2.6.0rc1/advanced/micro.html" class="article-link">
									<div>
										<div class="doc-article-head">
											<span class="doc-head-content">Performing Inference or Training on MCU or Small Systems</span>
										</div>
										<div class="doc-article-desc">
										MindSpore Lite provides a light-weight Micro solution for deploying AI models to IOT devices.
										In this solution, the model is generated into pure C code called by simple operators, and online model parsing and graph compilation are no longer required, which is suitable for the environment with limited memory and computing power.
										</div>
									</div>
								</a>
							</div>
							<div class="doc-article-item all os-Android os-Linux hardware-NPU stage-Environment-Preparation user-Intermediate user-Expert hidden">
								<a href="https://www.mindspore.cn/lite/docs/en/r2.6.0rc1/advanced/third_party/npu_info.html" class="article-link">
									<div>
										<div class="doc-article-head">
											<span class="doc-head-content">NPU Integration Information</span>
										</div>
										<div class="doc-article-desc">
										This tutorial introduces the instructions for using the integrated NPU, including steps to use, supported chips, and supported operators.
										</div>
									</div>
								</a>
							</div>
							<div class="doc-article-item all hardware-TensorRT os-Linux stage-Environment-Preparation user-Intermediate user-Expert hidden">
								<a href="https://www.mindspore.cn/lite/docs/en/r2.6.0rc1/advanced/third_party/tensorrt_info.html" class="article-link">
									<div>
										<div class="doc-article-head">
											<span class="doc-head-content">TensorRT Integration Information</span>
										</div>
										<div class="doc-article-desc">
										This tutorial introduces the instructions for using the integrated TensorRT.
										</div>
									</div>
								</a>
							</div>
							<div class="doc-article-item all os-Linux os-Android hardware-Ascend stage-Environment-Preparation stage-Data-Preparation user-Beginner user-Intermediate user-Expert hidden">
								<a href="https://www.mindspore.cn/lite/docs/en/r2.6.0rc1/advanced/third_party/ascend_info.html" class="article-link">
									<div>
										<div class="doc-article-head">
											<span class="doc-head-content">Integrated Ascend</span>
										</div>
							
										<div class="doc-article-desc">
										This document describes how to use MindSpore Lite to perform inference and use the dynamic shape function on Linux in the Ascend environment.
										</div>
									</div>
								</a>
							</div>
							<div class="doc-article-item all os-Linux os-Android language-cpp stage-Whole-Process stage-Model-Export stage-Model-Converting stage-Model-Training user-Beginner user-Intermediate user-Expert hidden">
								<a href="https://www.mindspore.cn/lite/docs/en/r2.6.0rc1/train/train_lenet.html" class="article-link">
									<div>
										<div class="doc-article-head">
											<span class="doc-head-content">Implementing Device Training Based On C++ Interface</span>
										</div>
							
										<div class="doc-article-desc">
										This tutorial explains the code that trains a LeNet model using Training-on-Device infrastructure.
										</div>
									</div>
								</a>
							</div>
							<div class="doc-article-item all os-Android language-java stage-Whole-Process stage-Model-Loading stage-Model-Training stage-Data-Preparation user-Beginner user-Intermediate user-Expert hidden">
								<a href="https://www.mindspore.cn/lite/docs/en/r2.6.0rc1/train/train_lenet_java.html" class="article-link">
									<div>
										<div class="doc-article-head">
											<span class="doc-head-content">Implementing Device Training Based On Java Interface</span>
										</div>
							
										<div class="doc-article-desc">
										This tutorial demonstrates how to use the Java interface on MindSpore Lite by building and deploying LeNet of the Java version. 
										</div>
									</div>
								</a>
							</div>
							<div class="doc-article-item all os-Linux os-Android language-cpp stage-Model-Training stage-Model-Loading stage-Data-Preparation user-Intermediate user-Expert hidden">
								<a href="https://www.mindspore.cn/lite/docs/en/r2.6.0rc1/train/runtime_train_cpp.html" class="article-link">
									<div>
										<div class="doc-article-head">
											<span class="doc-head-content">Device-side Training (C++)</span>
										</div>
										<div class="doc-article-desc">
										After model conversion using MindSpore Lite, the model training process needs to be completed in Runtime. This tutorial introduces how to use C++ interface to write training code.
										</div>
									</div>
								</a>
							</div>
							<div class="doc-article-item all os-Linux os-Android language-java stage-Model-Training stage-Model-Loading stage-Data-Preparation user-Intermediate user-Expert hidden">
								<a href="https://www.mindspore.cn/lite/docs/en/r2.6.0rc1/train/runtime_train_java.html" class="article-link">
									<div>
										<div class="doc-article-head">
											<span class="doc-head-content">Device-side Training (Java)</span>
										</div>
										<div class="doc-article-desc">
										After model conversion using MindSpore Lite, the model training process needs to be completed in Runtime. This tutorial introduces how to use Java interface to write training code.
										</div>
									</div>
								</a>
							</div>
							<div class="doc-article-item all os-Linux language-cpp stage-Model-Converting user-Intermediate user-Expert hidden">
								<a href="https://www.mindspore.cn/lite/docs/en/r2.6.0rc1/advanced/third_party/converter_register.html" class="article-link">
									<div>
										<div class="doc-article-head">
											<span class="doc-head-content">Building Custom Operators Offline</span>
										</div>
										<div class="doc-article-desc">
                                        MindSpore Lite provides a highly flexible tool for offline model conversion. It supports users to expand such as combining your own specific hardware with MindSpore Lite Inference Engine.
										</div>
									</div>
								</a>
							</div>
							<div class="doc-article-item all os-Linux language-cpp stage-Inference user-Expert hidden">
								<a href="https://www.mindspore.cn/lite/docs/en/r2.6.0rc1/advanced/third_party/register_kernel.html" class="article-link">
									<div>
										<div class="doc-article-head">
											<span class="doc-head-content">Building Custom Operators Online</span>
										</div>
										<div class="doc-article-desc">
										MindSpore Lite provides a southbound operator registration mechanism. This document describes how to schedule your own operators through the MindSpore Lite framework.
										</div>
									</div>
								</a>
							</div>
							<div class="doc-article-item all os-Linux language-cpp user-Expert hidden">
								<a href="https://www.mindspore.cn/lite/docs/en/r2.6.0rc1/advanced/third_party/delegate.html" class="article-link">
									<div>
										<div class="doc-article-head">
											<span class="doc-head-content">Using Delegate to Support Third-party AI Framework</span>
										</div>
										<div class="doc-article-desc">
										Delegate of MindSpore Lite is used to support third-party AI frameworks (such as NPU, TensorRT) to quickly access to the inference process in MindSpore Lite. 
										</div>
									</div>
								</a>
							</div>
							<div class="doc-article-item all os-Windows os-Linux stage-Model-Converting user-Intermediate user-Expert hidden">
								<a href="https://www.mindspore.cn/lite/docs/en/r2.6.0rc1/converter/converter_tool.html" class="article-link">
									<div>
										<div class="doc-article-head">
											<span class="doc-head-content">Device-side Models Conversion</span>
										</div>
										<div class="doc-article-desc">
										MindSpore Lite provides a tool for offline model conversion. It supports conversion of multiple types of models. The converted models can be used for inference. 
										</div>
									</div>
								</a>
							</div>
							<div class="doc-article-item all os-Linux stage-Environment-Preparation stage-Model-Export stage-Model-Converting user-Intermediate user-Expert hidden">
								<a href="https://www.mindspore.cn/lite/docs/en/r2.6.0rc1/train/converter_train.html" class="article-link">
									<div>
										<div class="doc-article-head">
											<span class="doc-head-content">Converting MindSpore Lite Models</span>
										</div>
										<div class="doc-article-desc">
										This tutorial introduces that how to convert your MindSpore ToD(Train on Device) model.
										</div>
									</div>
								</a>
							</div>
							<div class="doc-article-item all os-Linux os-Windows stage-Environment-Preparation stage-Benchmark-Testing user-Intermediate user-Expert hidden">
								<a href="https://www.mindspore.cn/lite/docs/en/r2.6.0rc1/tools/benchmark_tool.html" class="article-link">
									<div>
										<div class="doc-article-head">
											<span class="doc-head-content">benchmark</span>
										</div>
										<div class="doc-article-desc">
										After model conversion and before inference, you can use the Benchmark tool to perform benchmark testing on a MindSpore Lite model. 
										It can not only perform quantitative analysis (performance) on the forward inference execution duration of a MindSpore Lite model, but also perform comparative error analysis (accuracy) based on the output of the specified model.
										</div>
									</div>
								</a>
							</div>
							<div class="doc-article-item all os-Linux stage-Data-Preparation stage-Benchmark-Testing user-Intermediate user-Expert hidden">
								<a href="https://www.mindspore.cn/lite/docs/en/r2.6.0rc1/tools/benchmark_train_tool.html" class="article-link">
									<div>
										<div class="doc-article-head">
											<span class="doc-head-content">benchmark_train</span>
										</div>
										<div class="doc-article-desc">
										You can use the net_train tool to perform benchmark testing on a MindSpore ToD (Train on Device) model. It can not only perform quantitative analysis (performance) on the execution duration the model, but also perform comparative error analysis (accuracy) based on the output of the specified model.
										</div>
									</div>
								</a>
							</div>
							<div class="doc-article-item all os-Linux stage-Environment-Preparation stage-Static-Library-Cropping user-Intermediate user-Expert hidden">
								<a href="https://www.mindspore.cn/lite/docs/en/r2.6.0rc1/tools/cropper_tool.html" class="article-link">
									<div>
										<div class="doc-article-head">
											<span class="doc-head-content">Static Library Cropper Tool</span>
										</div>
										<div class="doc-article-desc">
										MindSpore Lite provides the libmindspore-lite.a static library cropping tool for runtime, which can filter out the operators in the ms model, crop the static library files, and effectively reduce the size of the library files.
										</div>
									</div>
								</a>
							</div>
							<div class="doc-article-item all os-Windows os-Linux stage-Visualization">
								<a href="https://www.mindspore.cn/lite/docs/en/r2.6.0rc1/tools/visual_tool.html" class="article-link">
									<div>
										<div class="doc-article-head">
											<span class="doc-head-content">Visualization Tool</span>
										</div>
										<div class="doc-article-desc">
										Netron is a neural network model visualization tool developed based on the Electron platform. Netron supports MindSpore Lite models, allowing you to easily view model information. 
										</div>
									</div>
								</a>
							</div>
							<div class="doc-article-item all os-Linux stage-Model-Obfuscator user-Expert">
								<a href="https://www.mindspore.cn/lite/docs/en/r2.6.0rc1/tools/obfuscator_tool.html" class="article-link">
									<div>
										<div class="doc-article-head">
											<span class="doc-head-content">Model Obfuscation Tool</span>
										</div>
										<div class="doc-article-desc">
										MindSpore Lite provides a lightweight offline model obfuscator to protect the confidentiality of model files deployed on the IoT devices.
										</div>
									</div>
								</a>
							</div>
							<div class="doc-article-item all os-Linux language-cpp stage-Whole-Process stage-Inference stage-Data-Preparation user-Beginner hidden">
								<a href="https://mindspore.cn/lite/docs/en/r2.6.0rc1/mindir/runtime_cpp.html" class="article-link">
									<div>
										<div class="doc-article-head">
											<span class="doc-head-content">Using C++ Interface to Perform Cloud-side Inference</span>
										</div>
							
										<div class="doc-article-desc">
										This tutorial introduces how to use C++ interface to write cloud-side inference code.
										</div>
									</div>
								</a>
							</div>
							<div class="doc-article-item all os-Linux language-java stage-Whole-Process stage-Inference stage-Data-Preparation user-Beginner hidden">
								<a href="https://mindspore.cn/lite/docs/en/r2.6.0rc1/mindir/runtime_java.html" class="article-link">
									<div>
										<div class="doc-article-head">
											<span class="doc-head-content">Using Java Interface to Perform Cloud-side Inference</span>
										</div>
							
										<div class="doc-article-desc">
										This tutorial introduces how to use Java interface to write cloud-side inference code.
										</div>
									</div>
								</a>
							</div>
							<div class="doc-article-item all os-Linux language-python stage-Whole-Process stage-Inference stage-Data-Preparation user-Beginner hidden">
								<a href="https://mindspore.cn/lite/docs/en/r2.6.0rc1/mindir/runtime_python.html" class="article-link">
									<div>
										<div class="doc-article-head">
											<span class="doc-head-content">Using Python Interface to Perform Cloud-side Inference</span>
										</div>
							
										<div class="doc-article-desc">
										This tutorial introduces how to use Python interface to write cloud-side inference code.
										</div>
									</div>
								</a>
							</div>
							<div class="doc-article-item all os-Linux language-cpp stage-Inference stage-Model-Loading stage-Data-Preparation user-Intermediate user-Expert hidden">
								<a href="https://www.mindspore.cn/lite/docs/en/r2.6.0rc1/mindir/runtime_parallel_cpp.html" class="article-link">
									<div>
										<div class="doc-article-head">
											<span class="doc-head-content">Using C++ Interface to Perform Concurrent Inference</span>
										</div>
							
										<div class="doc-article-desc">
										This tutorial provides an example program for MindSpore Lite to multi-model concurrent inference by using C++ interface.
										</div>
									</div>
								</a>
							</div>
							<div class="doc-article-item all os-Linux language-java stage-Inference stage-Model-Loading stage-Data-Preparation user-Intermediate user-Expert hidden">
								<a href="https://www.mindspore.cn/lite/docs/en/r2.6.0rc1/mindir/runtime_parallel_java.html" class="article-link">
									<div>
										<div class="doc-article-head">
											<span class="doc-head-content">Using Java Interface to Perform Concurrent Inference</span>
										</div>
							
										<div class="doc-article-desc">
										This tutorial provides an example program for MindSpore Lite to multi-model concurrent inference by using Java interface.
										</div>
									</div>
								</a>
							</div>
							<div class="doc-article-item all os-Linux language-python stage-Inference stage-Model-Loading stage-Data-Preparation user-Intermediate user-Expert hidden">
								<a href="https://www.mindspore.cn/lite/docs/en/r2.6.0rc1/mindir/runtime_parallel_python.html" class="article-link">
									<div>
										<div class="doc-article-head">
											<span class="doc-head-content">Using Python Interface to Perform Concurrent Inference</span>
										</div>
							
										<div class="doc-article-desc">
										This tutorial provides an example program for MindSpore Lite to multi-model concurrent inference by using Python interface.
										</div>
									</div>
								</a>
							</div>
							<div class="doc-article-item all os-Linux stage-Model-Converting user-Intermediate user-Expert hidden">
								<a href="https://www.mindspore.cn/lite/docs/en/r2.6.0rc1/mindir/converter_tool.html" class="article-link">
									<div>
										<div class="doc-article-head">
											<span class="doc-head-content">Offline Conversion of Inference Models</span>
										</div>
										<div class="doc-article-desc">
										MindSpore Lite provides a tool for cloud-side offline model conversion. It supports conversion of multiple types of models. The converted models can be used for inference. 
										</div>
									</div>
								</a>
							</div>
							<div class="doc-article-item all os-Linux language-python stage-Environment-Preparation stage-Model-Export stage-Model-Converting user-Intermediate user-Expert hidden">
								<a href="https://www.mindspore.cn/lite/docs/en/r2.6.0rc1/mindir/converter_python.html" class="article-link">
									<div>
										<div class="doc-article-head">
											<span class="doc-head-content">Using Python Interface to Perform Model Conversions</span>
										</div>
										<div class="doc-article-desc">
										MindSpore Lite supports cloud-side model conversion via Python interface, supporting multiple types of model conversion, and the converted models can be used for inference.
										</div>
									</div>
								</a>
							</div>
							<div class="doc-article-item all os-Linux stage-Environment-Preparation stage-Benchmark-Testing user-Intermediate user-Expert hidden">
								<a href="https://www.mindspore.cn/lite/docs/en/r2.6.0rc1/mindir/benchmark_tool.html" class="article-link">
									<div>
										<div class="doc-article-head">
											<span class="doc-head-content">benchmark</span>
										</div>
										<div class="doc-article-desc">
										After model conversion and before cloud-side inference, you can use the Benchmark tool to perform benchmark testing on a MindSpore Lite model. 
										It can not only perform quantitative analysis (performance) on the forward inference execution duration of a MindSpore Lite model, but also perform comparative error analysis (accuracy) based on the output of the specified model.
										</div>
									</div>
								</a>
							</div>
							<div class="doc-article-item all os-Windows os-Linux os-Android os-ios os-ohos os-iot user-Beginner user-Intermediate user-Expert hidden">
								<a href="https://mindspore.cn/lite/docs/en/r2.6.0rc1/reference/architecture_lite.html" class="article-link">
									<div>
										<div class="doc-article-head">
											<span class="doc-head-content">Overall Architecture (Lite)</span>
										</div>
							
										<div class="doc-article-desc">
										MindSpore Lite is an ultra-fast, intelligent, and simplified AI engine that enables intelligent applications in all scenarios, provides E2E solutions for users, and helps users enable AI capabilities.
										</div>
									</div>
								</a>
							</div>
							<div class="doc-article-item all os-Windows os-Linux os-Android os-ios os-ohos user-Beginner user-Intermediate user-Expert hidden">
								<a href="https://mindspore.cn/lite/docs/en/r2.6.0rc1/reference/operator_list_lite.html" class="article-link">
									<div>
										<div class="doc-article-head">
											<span class="doc-head-content">Lite Operator List</span>
										</div>
							
										<div class="doc-article-desc">
										This article lists the operators supported by MindSpore Lite.
										</div>
									</div>
								</a>
							</div>
							<div class="doc-article-item all os-iot user-Beginner user-Intermediate user-Expert hidden">
								<a href="https://mindspore.cn/lite/docs/en/r2.6.0rc1/reference/operator_list_codegen.html" class="article-link">
									<div>
										<div class="doc-article-head">
											<span class="doc-head-content">Codegen Operator List</span>
										</div>
							
										<div class="doc-article-desc">
										This article lists the operators supported by MindSpore Lite Codegen.
										</div>
									</div>
								</a>
							</div>
							<div class="doc-article-item all os-Android user-Beginner user-Intermediate user-Expert hidden">
								<a href="https://mindspore.cn/lite/docs/en/r2.6.0rc1/reference/image_classification_lite.html" class="article-link">
									<div>
										<div class="doc-article-head">
											<span class="doc-head-content">Image Classification Model</span>
										</div>
							
										<div class="doc-article-desc">
										Image classification is to identity what an image represents, to predict the object list and the probabilities.
										</div>
									</div>
								</a>
							</div>
							<div class="doc-article-item all os-Android user-Beginner user-Intermediate user-Expert hidden">
								<a href="https://mindspore.cn/lite/docs/en/r2.6.0rc1/reference/object_detection_lite.html" class="article-link">
									<div>
										<div class="doc-article-head">
											<span class="doc-head-content">Object Detection Model</span>
										</div>
							
										<div class="doc-article-desc">
										Object detection can identify the object in the image and its position in the image.
										</div>
									</div>
								</a>
							</div>
							<div class="doc-article-item all os-Android user-Beginner user-Intermediate user-Expert hidden">
								<a href="https://mindspore.cn/lite/docs/en/r2.6.0rc1/reference/image_segmentation_lite.html" class="article-link">
									<div>
										<div class="doc-article-head">
											<span class="doc-head-content">Image Segmentation Model</span>
										</div>
							
										<div class="doc-article-desc">
										Image segmentation is used to detect the position of the object in the picture or a pixel belongs to which object.
										</div>
									</div>
								</a>
							</div>
							<div class="doc-article-item all os-Android user-Beginner user-Intermediate user-Expert hidden">
								<a href="https://mindspore.cn/lite/docs/en/r2.6.0rc1/reference/style_transfer_lite.html" class="article-link">
									<div>
										<div class="doc-article-head">
											<span class="doc-head-content">Style Transfer Model</span>
										</div>
							
										<div class="doc-article-desc">
										The style transfer model can change the artistic style of the user’s target image according to the standard image built in this demo, and display it in the App image preview interface.
										</div>
									</div>
								</a>
							</div>
							<div class="doc-article-item all os-Android user-Beginner user-Intermediate user-Expert hidden">
								<a href="https://mindspore.cn/lite/docs/en/r2.6.0rc1/reference/scene_detection_lite.html" class="article-link">
									<div>
										<div class="doc-article-head">
											<span class="doc-head-content">Scene Detection Model</span>
										</div>
							
										<div class="doc-article-desc">
										Scene detection can identify the type of scene in the device’s camera.
										</div>
									</div>
								</a>
							</div>
							<div class="doc-article-item all os-Windows os-Linux os-Android os-ios os-ohos os-iot user-Beginner user-Intermediate user-Expert hidden">
								<a href="https://mindspore.cn/lite/docs/en/r2.6.0rc1/reference/faq.html" class="article-link">
									<div>
										<div class="doc-article-head">
											<span class="doc-head-content">Troubleshooting</span>
										</div>
							
										<div class="doc-article-desc">
										If you encounter an issue when using MindSpore Lite, you can view logs first. In most scenarios, you can locate the issue based on the error information reported in logs.
										</div>
									</div>
								</a>
							</div>
							<div class="doc-article-item all os-Windows os-Linux user-Beginner user-Intermediate user-Expert hidden">
								<a href="https://mindspore.cn/lite/docs/en/r2.6.0rc1/reference/log.html" class="article-link">
									<div>
										<div class="doc-article-head">
											<span class="doc-head-content">Log</span>
										</div>
							
										<div class="doc-article-desc">
										Only server inference version and windows version support environment variables below, except GLOG_v.
										</div>
									</div>
								</a>
							</div>
							<div class="doc-article-item all os-Windows os-Linux os-Android os-ios os-ohos os-iot user-Beginner user-Intermediate user-Expert hidden">
								<a href="https://mindspore.cn/lite/docs/en/r2.6.0rc1/RELEASE.html" class="article-link">
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

