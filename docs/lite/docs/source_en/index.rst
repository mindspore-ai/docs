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

.. toctree::
   :glob:
   :maxdepth: 1
   :caption: Quick Start
   :hidden:

   quick_start/quick_start_cpp
   quick_start/quick_start_java
   quick_start/quick_start
   quick_start/image_segmentation
   quick_start/train_lenet
   quick_start/train_lenet_java

.. toctree::
   :glob:
   :maxdepth: 1
   :caption: Inference on Devices
   :hidden:

   use/converter_tool
   use/post_training_quantization
   use/data_preprocessing
   use/runtime
   use/micro
   use/asic
   use/register_kernel
   use/delegate

.. toctree::
   :glob:
   :maxdepth: 1
   :caption: Training on Devices
   :hidden:

   use/converter_train
   use/runtime_train

.. toctree::
   :glob:
   :maxdepth: 1
   :caption: Other Tools
   :hidden:

   use/benchmark
   use/cropper_tool
   use/visual_tool
   use/obfuscator_tool

.. toctree::
   :glob:
   :maxdepth: 1
   :caption: References
   :hidden:

   architecture_lite
   operator_list_lite
   operator_list_codegen
   model_lite

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
							<div class="doc-article-item all os-Windows os-Linux os-Android stage-Environment-Preparation user-Beginner user-Intermediate user-Expert hidden">
								<a href="https://www.mindspore.cn/lite/docs/en/r1.5/use/downloads.html" class="article-link">
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
								<a href="https://www.mindspore.cn/lite/docs/en/r1.5/use/build.html" class="article-link">
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
						    <div class="doc-article-item all os-Windows os-Linux language-cpp stage-Whole-Process stage-Inference stage-Data-Preparation user-Beginner hidden">
								<a href="https://www.mindspore.cn/lite/docs/en/r1.5/quick_start/quick_start_cpp.html" class="article-link">
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
								<a href="https://www.mindspore.cn/lite/docs/en/r1.5/quick_start/quick_start_java.html" class="article-link">
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
							<div class="doc-article-item all os-Linux os-Android language-cpp stage-Whole-Process stage-Model-Converting stage-Model-Loading stage-Inference stage-Data-Preparation user-Beginner user-Intermediate user-Expert hidden">
								<a href="https://www.mindspore.cn/lite/docs/en/r1.5/quick_start/quick_start.html" class="article-link">
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
								<a href="https://www.mindspore.cn/lite/docs/en/r1.5/quick_start/image_segmentation.html" class="article-link">
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
							<div class="doc-article-item all os-Linux os-Android language-cpp stage-Whole-Process stage-Model-Export stage-Model-Converting stage-Model-Training user-Beginner user-Intermediate user-Expert hidden">
								<a href="https://www.mindspore.cn/lite/docs/en/r1.5/quick_start/train_lenet.html" class="article-link">
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
								<a href="https://www.mindspore.cn/lite/docs/en/r1.5/quick_start/train_lenet_java.html" class="article-link">
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
							<div class="doc-article-item all os-Windows os-Linux stage-Model-Converting user-Intermediate user-Expert hidden">
								<a href="https://www.mindspore.cn/lite/docs/en/r1.5/use/converter_tool.html" class="article-link">
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
							<div class="doc-article-item all os-Windows os-Linux stage-Model-Converting stage-Model-Optimization user-Intermediate user-Expert hidden">
								<a href="https://www.mindspore.cn/lite/docs/en/r1.5/use/post_training_quantization.html" class="article-link">
									<div>
										<div class="doc-article-head">
											<span class="doc-head-content">Optimizing the Model (Quantization After Training)</span>
										</div>
										<div class="doc-article-desc">
											Converting a trained `float32` model into an `int8` model through quantization after training can reduce the model size and improve the inference performance.
											This tutorial introduces how to use the function.
										</div>
									</div>
								</a>
							</div>
							<div class="doc-article-item all os-Linux os-iot language-cpp stage-Model-Code-Generation stage-Inference user-Beginner user-Intermediate hidden">
								<a href="https://www.mindspore.cn/lite/docs/en/r1.5/use/micro.html" class="article-link">
									<div>
										<div class="doc-article-head">
											<span class="doc-head-content">Perform Inference on Mini and Small Systems</span>
										</div>
										<div class="doc-article-desc">
										MindSpore Lite provides a code generator tool, namely codegen, which could have runtime compiling and computational graphs building done offline. 
										Only necessary codes and information are kept in the generated program, thereby minimizing the size of the generated inference program.
										</div>
									</div>
								</a>
							</div>
							<div class="doc-article-item all os-Windows os-Linux language-cpp stage-Data-Preparation user-Beginner user-Intermediate user-Expert hidden">
								<a href="https://www.mindspore.cn/lite/docs/en/r1.5/use/image_processing.html" class="article-link">
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
								<a href="https://www.mindspore.cn/lite/docs/en/r1.5/use/runtime_cpp.html" class="article-link">
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
								<a href="https://www.mindspore.cn/lite/docs/en/r1.5/use/runtime_java.html" class="article-link">
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
							<div class="doc-article-item all os-Android os-Linux stage-Environment-Preparation user-Intermediate user-Expert hidden">
								<a href="https://www.mindspore.cn/lite/docs/en/r1.5/use/npu_info.html" class="article-link">
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
								<a href="https://www.mindspore.cn/lite/docs/en/r1.5/use/nnie.html" class="article-link">
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
								<a href="https://www.mindspore.cn/lite/docs/en/r1.5/use/tensorrt_info.html" class="article-link">
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
							<div class="doc-article-item all os-Linux os-Windows stage-Environment-Preparation stage-Benchmark-Testing user-Intermediate user-Expert hidden">
								<a href="https://www.mindspore.cn/lite/docs/en/r1.5/use/benchmark_tool.html" class="article-link">
									<div>
										<div class="doc-article-head">
											<span class="doc-head-content">Performing Benchmark Testing</span>
										</div>
										<div class="doc-article-desc">
											After model conversion and before inference, you can use the Benchmark tool to perform benchmark testing on a MindSpore Lite model. 
											It can not only perform quantitative analysis (performance) on the forward inference execution duration of a MindSpore Lite model, but also perform comparative error analysis (accuracy) based on the output of the specified model.
										</div>
									</div>
								</a>
							</div>
							<div class="doc-article-item all os-Linux stage-Environment-Preparation stage-Static-Library-Cropping user-Intermediate user-Expert hidden">
								<a href="https://www.mindspore.cn/lite/docs/en/r1.5/use/cropper_tool.html" class="article-link">
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
							<div class="doc-article-item all os-Linux stage-Environment-Preparation stage-Model-Export stage-Model-Converting user-Intermediate user-Expert hidden">
								<a href="https://www.mindspore.cn/lite/docs/en/r1.5/use/converter_train.html" class="article-link">
									<div>
										<div class="doc-article-head">
											<span class="doc-head-content">Converting MindSpore ToD Models</span>
										</div>
										<div class="doc-article-desc">
											This tutorial introduces that how to convert your MindSpore ToD(Train on Device) model.
										</div>
									</div>
								</a>
							</div>
							<div class="doc-article-item all os-Linux os-Android language-cpp stage-Model-Training stage-Model-Loading stage-Data-Preparation user-Intermediate user-Expert hidden">
								<a href="https://www.mindspore.cn/lite/docs/en/r1.5/use/runtime_train_cpp.html" class="article-link">
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
							<div class="doc-article-item all os-Linux stage-Data-Preparation stage-Benchmark-Testing user-Intermediate user-Expert hidden">
								<a href="https://www.mindspore.cn/lite/docs/en/r1.5/use/benchmark_train_tool.html" class="article-link">
									<div>
										<div class="doc-article-head">
											<span class="doc-head-content">Performing Benchmark Testing of MindSpore ToD</span>
										</div>
										<div class="doc-article-desc">
											You can use the net_train tool to perform benchmark testing on a MindSpore ToD (Train on Device) model. It can not only perform quantitative analysis (performance) on the execution duration the model, but also perform comparative error analysis (accuracy) based on the output of the specified model.
										</div>
									</div>
								</a>
							</div>
							<div class="doc-article-item all os-Windows os-Linux os-mac stage-Visualization">
								<a href="https://www.mindspore.cn/lite/docs/en/r1.5/use/visual_tool.html" class="article-link">
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
								<a href="https://www.mindspore.cn/lite/docs/en/r1.5/use/obfuscator_tool.html" class="article-link">
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
						</div>
				
					</div>
					
				</div>
			</div>
		</div>
	