.. MindSpore documentation master file, created by
   sphinx-quickstart on Thu Aug 17 09:00:00 2020.
   You can adapt this file completely to your liking, but it should at least
   contain the root `toctree` directive.

MindSpore Lite Documentation
=======================================

.. toctree::
   :glob:
   :maxdepth: 1
   :caption: Obtain MindSpore Lite
   :hidden:
 
   use/downloads
   use/build
   use/cloud_infer/build

.. toctree::
   :glob:
   :maxdepth: 1
   :caption: Quick Start
   :hidden:

   quick_start/one_hour_introduction

.. toctree::
   :glob:
   :maxdepth: 1
   :caption: Inference on Devices
   :hidden:

   device_infer_example
   use/post_training_quantization
   use/data_preprocessing
   use/runtime
   use/micro
   use/asic

.. toctree::
   :glob:
   :maxdepth: 1
   :caption: Training on Devices
   :hidden:

   device_train_example
   use/runtime_train

.. toctree::
   :glob:
   :maxdepth: 1
   :caption: Third-party hardware docking
   :hidden:

   use/register
   use/delegate

.. toctree::
   :glob:
   :maxdepth: 1
   :caption: Device-side Tools
   :hidden:

   use/converter
   use/benchmark
   use/cropper_tool
   use/visual_tool
   use/obfuscator_tool

.. toctree::
   :glob:
   :maxdepth: 1
   :caption: Cloud-side Inference
   :hidden:

   use/cloud_infer/runtime
   use/cloud_infer/runtime_parallel

.. toctree::
   :glob:
   :maxdepth: 1
   :caption: Cloud-side Tools
   :hidden:
   
   use/cloud_infer/converter
   use/cloud_infer/benchmark

.. toctree::
   :glob:
   :maxdepth: 1
   :caption: References
   :hidden:

   architecture_lite
   operator_list_lite
   operator_list_codegen
   model_lite
   troubleshooting_guide
   log

.. toctree::
   :glob:
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
									<button class="doc-filter-btn doc-btn" id="hardware-NNIE">NNIE</button>
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
							<div class="doc-article-item all os-Windows os-Linux os-Android os-ios stage-Environment-Preparation user-Beginner user-Intermediate user-Expert hidden">
								<a href="https://www.mindspore.cn/lite/docs/en/master/use/downloads.html" class="article-link">
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
							<div class="doc-article-item all os-Windows os-Linux os-Android os-mac os-ios stage-Environment-Preparation user-Intermediate user-Expert hidden">
								<a href="https://www.mindspore.cn/lite/docs/en/master/use/build.html" class="article-link">
									<div>
										<div class="doc-article-head">
											<span class="doc-head-content">Building MindSpore Lite</span>
										</div>
							
										<div class="doc-article-desc">
										This chapter introduces how to quickly compile MindSpore Lite.
										</div>
									</div>
								</a>
							</div>
							<div class="doc-article-item all os-Windows os-Linux os-Android language-cpp stage-Whole-Process stage-Model-Converting stage-Model-Loading stage-Data-Preparation stage-Inference user-Beginner user-Intermediate user-Expert hidden">
								<a href="https://www.mindspore.cn/lite/docs/en/master/quick_start/one_hour_introduction.html" class="article-link">
									<div>
										<div class="doc-article-head">
											<span class="doc-head-content">Getting Started in One Hour</span>
										</div>
							
										<div class="doc-article-desc">
										This document uses a model inference example to describe how to use basic MindSpore Lite functions.
										</div>
									</div>
								</a>
							</div>
							<div class="doc-article-item all os-Windows os-Linux language-cpp stage-Whole-Process stage-Inference stage-Data-Preparation user-Beginner hidden">
								<a href="https://www.mindspore.cn/lite/docs/en/master/quick_start/quick_start_cpp.html" class="article-link">
									<div>
										<div class="doc-article-head">
											<span class="doc-head-content">Experience C++ Simple Inference Demo</span>
										</div>
							
										<div class="doc-article-desc">
										This tutorial provides a MindSpore Lite inference demo. It demonstrates the basic on-device inference process using C++ by inputting random data, executing inference, and printing the inference result.
										</div>
									</div>
								</a>
							</div>
							<div class="doc-article-item all os-Linux language-java stage-Whole-Process stage-Inference stage-Data-Preparation user-Beginner hidden">
								<a href="https://www.mindspore.cn/lite/docs/en/master/quick_start/quick_start_java.html" class="article-link">
									<div>
										<div class="doc-article-head">
											<span class="doc-head-content">Experience Java Simple Inference Demo</span>
										</div>
							
										<div class="doc-article-desc">
										This tutorial provides an example program for MindSpore Lite to perform inference. It demonstrates the basic process of performing inference on the device side using MindSpore Lite Java API by random inputting data, executing inference, and printing the inference result. 
										</div>
									</div>
								</a>
							</div>
							<div class="doc-article-item all os-Windows os-Linux language-python stage-Whole-Process stage-Inference stage-Data-Preparation user-Beginner hidden">
								<a href="https://www.mindspore.cn/lite/docs/en/master/quick_start/quick_start_python.html" class="article-link">
									<div>
										<div class="doc-article-head">
											<span class="doc-head-content">Experiencing the Python Simplified Inference Demo</span>
										</div>
							
										<div class="doc-article-desc">
										This tutorial provides a sample program for MindSpore Lite to perform inference, demonstrating the Python interface to perform the basic process of device-side inference through file input, inference execution, and inference result printing, and enables users to quickly understand the use of MindSpore Lite APIs related to inference execution.
										</div>
									</div>
								</a>
							</div>
							<div class="doc-article-item all os-Windows os-Linux language-c stage-Whole-Process stage-Inference stage-Data-Preparation user-Beginner hidden">
								<a href="https://www.mindspore.cn/lite/docs/en/master/quick_start/quick_start_c.html" class="article-link">
									<div>
										<div class="doc-article-head">
											<span class="doc-head-content">Expriencing Simpcified Inference Demo with C-language</span>
										</div>
							
										<div class="doc-article-desc">
										This tutorial provides a sample program for MindSpore Lite to perform inference, which demonstrates the basic process of end-side inference with C-language by randomly typing, performing inference, and printing inference results, so that users can quickly understand the use of MindSpore Lite to perform inference-related APIs.
										</div>
									</div>
								</a>
							</div>
							<div class="doc-article-item all os-Linux os-Android language-cpp stage-Whole-Process stage-Model-Converting stage-Model-Loading stage-Inference stage-Data-Preparation user-Beginner user-Intermediate user-Expert hidden">
								<a href="https://www.mindspore.cn/lite/docs/en/master/quick_start/quick_start.html" class="article-link">
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
								<a href="https://www.mindspore.cn/lite/docs/en/master/quick_start/image_segmentation.html" class="article-link">
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
								<a href="https://www.mindspore.cn/lite/docs/en/master/use/post_training_quantization.html" class="article-link">
									<div>
										<div class="doc-article-head">
											<span class="doc-head-content">Post Training Quantization</span>
										</div>
										<div class="doc-article-desc">
											Converting a trained `float32` model into an `int8` model through quantization after training can reduce the model size and improve the inference performance.
											This tutorial introduces how to use the function.
										</div>
									</div>
								</a>
							</div>
							<div class="doc-article-item all os-Windows os-Linux language-cpp stage-Data-Preparation user-Beginner user-Intermediate user-Expert hidden">
								<a href="https://www.mindspore.cn/lite/docs/en/master/use/image_processing.html" class="article-link">
									<div>
										<div class="doc-article-head">
											<span class="doc-head-content">Preprocessing Image Data</span>
										</div>
										<div class="doc-article-desc">
											This tutorial introduces how to process the image data before inference to meet the data format requirements for model inference by creating a LiteMat object.
										</div>
									</div>
								</a>
							</div>
							<div class="doc-article-item all os-Windows os-Linux os-Android os-mac os-ios language-cpp stage-Inference stage-Model-Loading stage-Data-Preparation user-Intermediate user-Expert hidden">
								<a href="https://www.mindspore.cn/lite/docs/en/master/use/runtime_cpp.html" class="article-link">
									<div>
										<div class="doc-article-head">
											<span class="doc-head-content">Using C++ Interface to Perform Inference</span>
										</div>
										<div class="doc-article-desc">
											After model conversion using MindSpore Lite, the model inference process needs to be completed in Runtime. This tutorial introduces how to use C++ API to write inference code.
										</div>
									</div>
								</a>
							</div>
							<div class="doc-article-item all os-Android language-java stage-Inference stage-Model-Loading stage-Data-Preparation user-Intermediate user-Expert hidden">
								<a href="https://www.mindspore.cn/lite/docs/en/master/use/runtime_java.html" class="article-link">
									<div>
										<div class="doc-article-head">
											<span class="doc-head-content">Using Java Interface to Perform Inference</span>
										</div>
										<div class="doc-article-desc">
											After model conversion using MindSpore Lite, the model inference process needs to be completed in Runtime. This tutorial introduces how to use Java API to write inference code.
										</div>
									</div>
								</a>
							</div>
							<div class="doc-article-item all os-Linux os-iot language-cpp stage-Model-Code-Generation stage-Inference user-Beginner user-Intermediate hidden">
								<a href="https://www.mindspore.cn/lite/docs/en/master/use/micro.html" class="article-link">
									<div>
										<div class="doc-article-head">
											<span class="doc-head-content">Perform Inference on MCU or Small Systems</span>
										</div>
										<div class="doc-article-desc">
										MindSpore Lite provides a light-weight Micro solution for deploying AI models to IOT devices.
										In this solution, the model is generated into pure C code called by simple operators, and online model parsing and graph compilation are no longer required, which is suitable for the environment with limited memory and computing power.
										</div>
									</div>
								</a>
							</div>
							<div class="doc-article-item all os-Android os-Linux hardware-NPU stage-Environment-Preparation user-Intermediate user-Expert hidden">
								<a href="https://www.mindspore.cn/lite/docs/en/master/use/npu_info.html" class="article-link">
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
							<div class="doc-article-item all hardware-NNIE os-Linux stage-Environment-Preparation user-Intermediate user-Expert hidden">
								<a href="https://www.mindspore.cn/lite/docs/en/master/use/nnie.html" class="article-link">
									<div>
										<div class="doc-article-head">
											<span class="doc-head-content">Usage Description of the Integrated NNIE</span>
										</div>
										<div class="doc-article-desc">
											This tutorial introduces instructions for using integrated NNIE.
										</div>
									</div>
								</a>
							</div>
							<div class="doc-article-item all hardware-TensorRT os-Linux stage-Environment-Preparation user-Intermediate user-Expert hidden">
								<a href="https://www.mindspore.cn/lite/docs/en/master/use/tensorrt_info.html" class="article-link">
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
								<a href="https://www.mindspore.cn/lite/docs/en/master/use/ascend_info.html" class="article-link">
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
								<a href="https://www.mindspore.cn/lite/docs/en/master/quick_start/train_lenet.html" class="article-link">
									<div>
										<div class="doc-article-head">
											<span class="doc-head-content">Implement Device Training Based On C++ Interface</span>
										</div>
							
										<div class="doc-article-desc">
											This tutorial explains the code that trains a LeNet model using Training-on-Device infrastructure.
										</div>
									</div>
								</a>
							</div>
							<div class="doc-article-item all os-Android language-java stage-Whole-Process stage-Model-Loading stage-Model-Training stage-Data-Preparation user-Beginner user-Intermediate user-Expert hidden">
								<a href="https://www.mindspore.cn/lite/docs/en/master/quick_start/train_lenet_java.html" class="article-link">
									<div>
										<div class="doc-article-head">
											<span class="doc-head-content">Implement Device Training Based On Java Interface</span>
										</div>
							
										<div class="doc-article-desc">
											This tutorial demonstrates how to use the Java API on MindSpore Lite by building and deploying LeNet of the Java version. 
										</div>
									</div>
								</a>
							</div>
							<div class="doc-article-item all os-Linux os-Android language-cpp stage-Model-Training stage-Model-Loading stage-Data-Preparation user-Intermediate user-Expert hidden">
								<a href="https://www.mindspore.cn/lite/docs/en/master/use/runtime_train_cpp.html" class="article-link">
									<div>
										<div class="doc-article-head">
											<span class="doc-head-content">Using C++ Interface to Perform Training</span>
										</div>
										<div class="doc-article-desc">
											After model conversion using MindSpore Lite, the model training process needs to be completed in Runtime. This tutorial introduces how to use C++ API to write training code.
										</div>
									</div>
								</a>
							</div>
							<div class="doc-article-item all os-Linux os-Android language-java stage-Model-Training stage-Model-Loading stage-Data-Preparation user-Intermediate user-Expert hidden">
								<a href="https://www.mindspore.cn/lite/docs/en/master/use/runtime_train_java.html" class="article-link">
									<div>
										<div class="doc-article-head">
											<span class="doc-head-content">Using Java Interface to Perform Training</span>
										</div>
										<div class="doc-article-desc">
											After model conversion using MindSpore Lite, the model training process needs to be completed in Runtime. This tutorial introduces how to use Java API to write training code.
										</div>
									</div>
								</a>
							</div>
							<div class="doc-article-item all os-Linux stage-Model-Converting user-Intermediate user-Expert hidden">
								<a href="https://www.mindspore.cn/lite/docs/en/master/use/converter_register.html" class="article-link">
									<div>
										<div class="doc-article-head">
											<span class="doc-head-content">Construct custom kernel by registering conversion tool</span>
										</div>
										<div class="doc-article-desc">
                      MindSpore Lite provides a highly flexible tool for offline model conversion. It supports users to expand such as combining your own specific hardware with MindSpore Lite Inference Engine.
										</div>
									</div>
								</a>
							</div>
							<div class="doc-article-item all os-Windows os-Linux os-Android language-cpp stage-Inference user-Expert hidden">
								<a href="https://www.mindspore.cn/lite/docs/en/master/use/register_kernel.html" class="article-link">
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
							<div class="doc-article-item all os-Linux user-Expert hidden">
								<a href="https://www.mindspore.cn/lite/docs/en/master/use/delegate.html" class="article-link">
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
								<a href="https://www.mindspore.cn/lite/docs/en/master/use/converter_tool.html" class="article-link">
									<div>
										<div class="doc-article-head">
											<span class="doc-head-content">Converting Models for Inference</span>
										</div>
										<div class="doc-article-desc">
											MindSpore Lite provides a tool for offline model conversion. It supports conversion of multiple types of models. The converted models can be used for inference. 
										</div>
									</div>
								</a>
							</div>
							<div class="doc-article-item all os-Linux stage-Environment-Preparation stage-Model-Export stage-Model-Converting user-Intermediate user-Expert hidden">
								<a href="https://www.mindspore.cn/lite/docs/en/master/use/converter_train.html" class="article-link">
									<div>
										<div class="doc-article-head">
											<span class="doc-head-content">Creating MindSpore Lite Models</span>
										</div>
										<div class="doc-article-desc">
											This tutorial introduces that how to convert your MindSpore ToD(Train on Device) model.
										</div>
									</div>
								</a>
							</div>
							<div class="doc-article-item all os-Linux language-python stage-Environment-Preparation stage-Model-Export stage-Model-Converting user-Intermediate user-Expert hidden">
								<a href="https://www.mindspore.cn/lite/docs/en/master/use/converter_python.html" class="article-link">
									<div>
										<div class="doc-article-head">
											<span class="doc-head-content">Using Python Interface for Model Conversion</span>
										</div>
										<div class="doc-article-desc">
											MindSpore Lite supports model conversion via Python interface, supporting multiple types of model conversion, and the converted models can be used for inference.
										</div>
									</div>
								</a>
							</div>
							<div class="doc-article-item all os-Linux os-Windows stage-Environment-Preparation stage-Benchmark-Testing user-Intermediate user-Expert hidden">
								<a href="https://www.mindspore.cn/lite/docs/en/master/use/benchmark_tool.html" class="article-link">
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
								<a href="https://www.mindspore.cn/lite/docs/en/master/use/benchmark_train_tool.html" class="article-link">
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
								<a href="https://www.mindspore.cn/lite/docs/en/master/use/cropper_tool.html" class="article-link">
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
							<div class="doc-article-item all os-Windows os-Linux os-mac stage-Visualization">
								<a href="https://www.mindspore.cn/lite/docs/en/master/use/visual_tool.html" class="article-link">
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
								<a href="https://www.mindspore.cn/lite/docs/en/master/use/obfuscator_tool.html" class="article-link">
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
							<div class="doc-article-item all os-Windows os-Linux language-cpp stage-Whole-Process stage-Inference stage-Data-Preparation user-Beginner hidden">
								<a href="https://www.mindspore.cn/lite/docs/en/master/quick_start/quick_start_server_inference_cpp.html" class="article-link">
									<div>
										<div class="doc-article-head">
											<span class="doc-head-content">Experience C++ Minimalist Concurrent Reasoning Demo</span>
										</div>
							
										<div class="doc-article-desc">
										This tutorial provides a MindSpore Lite parallel inference demo. It demonstrates the basic on-device inference process using C++ by inputting random data, executing inference, and printing the inference result.
										</div>
									</div>
								</a>
							</div>
							<div class="doc-article-item all os-Linux language-java stage-Whole-Process stage-Inference stage-Data-Preparation user-Beginner hidden">
								<a href="https://mindspore.cn/lite/docs/en/master/quick_start/quick_start_server_inference_java.html" class="article-link">
									<div>
										<div class="doc-article-head">
											<span class="doc-head-content">Experience Java Minimalist Concurrent Reasoning Demo</span>
										</div>
							
										<div class="doc-article-desc">
										This tutorial provides an example program for MindSpore Lite to parallel inference.
										</div>
									</div>
								</a>
							</div>
							<div class="doc-article-item all os-Windows os-Linux language-python stage-Whole-Process stage-Inference stage-Data-Preparation user-Beginner hidden">
								<a href="https://mindspore.cn/lite/docs/en/master/quick_start/quick_start_server_inference_python.html" class="article-link">
									<div>
										<div class="doc-article-head">
											<span class="doc-head-content">Experiencing the Python Simplified Concurrent Inference Demo</span>
										</div>
							
										<div class="doc-article-desc">
										This tutorial provides a sample program for MindSpore Lite to perform concurrent inference.
										</div>
									</div>
								</a>
							</div>
							<div class="doc-article-item all os-Windows os-Linux os-Android os-mac os-ios language-cpp stage-Inference stage-Model-Loading stage-Data-Preparation user-Intermediate user-Expert hidden">
								<a href="https://mindspore.cn/lite/docs/en/master/use/runtime_server_inference_cpp.html" class="article-link">
									<div>
										<div class="doc-article-head">
											<span class="doc-head-content">Experience C++ Minimalist Concurrent Reasoning Demo</span>
										</div>
							
										<div class="doc-article-desc">
										This tutorial provides a MindSpore Lite parallel inference demo.
										</div>
									</div>
								</a>
							</div>
							<div class="doc-article-item all os-Android language-java stage-Inference stage-Model-Loading stage-Data-Preparation user-Intermediate user-Expert hidden">
								<a href="https://mindspore.cn/lite/docs/en/master/use/runtime_server_inference_java.html" class="article-link">
									<div>
										<div class="doc-article-head">
											<span class="doc-head-content">Experience Java Minimalist Concurrent Reasoning Demo</span>
										</div>
							
										<div class="doc-article-desc">
										This tutorial provides an example program for MindSpore Lite to parallel inference.
										</div>
									</div>
								</a>
							</div>
							<div class="doc-article-item all os-Android user-Beginner user-Intermediate user-Expert hidden">
								<a href="https://mindspore.cn/lite/docs/en/master/architecture_lite.html" class="article-link">
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
							<div class="doc-article-item all os-Android user-Beginner user-Intermediate user-Expert hidden">
								<a href="https://mindspore.cn/lite/docs/en/master/operator_list_lite.html" class="article-link">
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
							<div class="doc-article-item all os-Android user-Beginner user-Intermediate user-Expert hidden">
								<a href="https://mindspore.cn/lite/docs/en/master/operator_list_codegen.html" class="article-link">
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
								<a href="https://mindspore.cn/lite/docs/en/master/image_classification_lite.html" class="article-link">
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
								<a href="https://mindspore.cn/lite/docs/en/master/object_detection_lite.html" class="article-link">
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
								<a href="https://mindspore.cn/lite/docs/en/master/image_segmentation_lite.html" class="article-link">
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
								<a href="https://mindspore.cn/lite/docs/en/master/style_transfer_lite.html" class="article-link">
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
								<a href="https://mindspore.cn/lite/docs/en/master/scene_detection_lite.html" class="article-link">
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
							<div class="doc-article-item all os-Android user-Beginner user-Intermediate user-Expert hidden">
								<a href="https://mindspore.cn/lite/docs/en/master/troubleshooting_guide.html" class="article-link">
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
							<div class="doc-article-item all os-Android user-Beginner user-Intermediate user-Expert hidden">
								<a href="https://mindspore.cn/lite/docs/en/master/log.html" class="article-link">
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
							<div class="doc-article-item all os-Android user-Beginner user-Intermediate user-Expert hidden">
								<a href="https://mindspore.cn/lite/docs/en/master/RELEASE.html" class="article-link">
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

