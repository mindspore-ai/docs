.. MindSpore documentation master file, created by
   sphinx-quickstart on Thu Aug 17 09:00:00 2020.
   You can adapt this file completely to your liking, but it should at least
   contain the root `toctree` directive.

Using MindSpore on Mobile and IoT
=======================================

.. toctree::
   :glob:
   :maxdepth: 1
   :caption: Quick Start
   :hidden:

   quick_start/quick_start
   quick_start/train_lenet

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
   :caption: Inference on Devices
   :hidden:

   use/converter_tool
   use/post_training_quantization
   use/data_preprocessing
   use/runtime
   use/tools

.. toctree::
   :glob:
   :maxdepth: 1
   :caption: Training on Devices
   :hidden:

   use/converter_train
   use/runtime_train
   use/tools_train

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
										<div class="doc-os">Operation System</div>
									</div>
									<div class="col-sm-10 col-sm-pull-1">
										<button class="doc-filter-btn doc-btn" id="os-Windows">Windows</button>
										<button class="doc-filter-btn doc-btn" id="os-Linux">Linux</button>
										<button class="doc-filter-btn doc-btn" id="os-Android">Android</button>
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
										<button class="doc-filter-btn doc-btn" id="stage-Inference">Inference</button>
										<button class="doc-filter-btn doc-btn" id="stage-Benchmark-Testing">Benchmark Testing</button>
										<button class="doc-filter-btn doc-btn" id="stage-Static-Library-Cropping">Static Library Cropping</button>
									</div>
								</div>
							</div>
							
						</div>
						<hr>
						<div class="doc-article-list">
							<div class="doc-article-item all os-Windows os-Linux os-Android stage-Whole-Process stage-Model-Converting stage-Model-Loading stage-Inference stage-Data-Preparation user-Beginner user-Intermediate user-Expert hidden">
								<a href="https://www.mindspore.cn/tutorial/lite/en/master/quick_start/quick_start.html" class="article-link">
									<div>
										<div class="doc-article-head">
											<span class="doc-head-content">Implementing an Image Classification Application</span>
										</div>
							
										<div class="doc-article-desc">
											It is recommended that you start from the image classification demo on the Android device to understand how to build the MindSpore Lite application project, configure dependencies, and use related APIs.
										</div>
									</div>
								</a>
							</div>
							<div class="doc-article-item all os-Linux os-Android stage-Whole-Process stage-Model-Export stage-Model-Converting stage-Model-Training user-Beginner user-Intermediate user-Expert hidden">
								<a href="https://www.mindspore.cn/tutorial/lite/en/master/quick_start/train_lenet.html" class="article-link">
									<div>
										<div class="doc-article-head">
											<span class="doc-head-content">Training a LeNet Model</span>
										</div>
							
										<div class="doc-article-desc">
											This tutorial explains the code that trains a LeNet model using Training-on-Device infrastructure.
										</div>
									</div>
								</a>
							</div>
							<div class="doc-article-item all os-Windows os-Linux os-Android stage-Environment-Preparation user-Beginner user-Intermediate user-Expert hidden">
								<a href="https://www.mindspore.cn/tutorial/lite/en/master/use/downloads.html" class="article-link">
									<div>
										<div class="doc-article-head">
											<span class="doc-head-content">Downloading MindSpore Lite</span>
										</div>
										<div class="doc-article-desc">
											This tutorial introduces how to download the MindSpore Lite quickly.
										</div>
									</div>
								</a>
							</div>
							
							<div class="doc-article-item all os-Windows os-Linux os-Android stage-Environment-Preparation user-Intermediate user-Expert hidden">
								<a href="https://www.mindspore.cn/tutorial/lite/en/master/use/build.html" class="article-link">
									<div>
										<div class="doc-article-head">
											<span class="doc-head-content">Building MindSpore Lite</span>
										</div>
										<div class="doc-article-desc">
											This tutorial introduces how to build the MindSpore Lite quickly.
												
										</div>
									</div>
								</a>
							</div>
							<div class="doc-article-item all os-Windows os-Linux stage-Model-Converting user-Intermediate user-Expert hidden">
								<a href="https://www.mindspore.cn/tutorial/lite/en/master/use/converter_tool.html" class="article-link">
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
								<a href="https://www.mindspore.cn/tutorial/lite/en/master/use/post_training_quantization.html" class="article-link">
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
							<div class="doc-article-item all os-Windows os-Linux stage-Data-Preparation user-Beginner user-Intermediate user-Expert hidden">
								<a href="https://www.mindspore.cn/tutorial/lite/en/master/use/image_processing.html" class="article-link">
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
							<div class="doc-article-item all os-Windows os-Linux os-Android stage-Inference stage-Model-Loading stage-Data-Preparation user-Intermediate user-Expert hidden">
								<a href="https://www.mindspore.cn/tutorial/lite/en/master/use/runtime_cpp.html" class="article-link">
									<div>
										<div class="doc-article-head">
											<span class="doc-head-content">Using Runtime for Model Inference (C++)</span>
										</div>
										<div class="doc-article-desc">
											After model conversion using MindSpore Lite, the model inference process needs to be completed in Runtime. This tutorial introduces how to use C++ API to write inference code.
										</div>
									</div>
								</a>
							</div>
							<div class="doc-article-item all os-Android stage-Inference stage-Model-Loading stage-Data-Preparation user-Intermediate user-Expert hidden">
								<a href="https://www.mindspore.cn/tutorial/lite/en/master/use/runtime_java.html" class="article-link">
									<div>
										<div class="doc-article-head">
											<span class="doc-head-content">Using Runtime for Model Inference (Java)</span>
										</div>
										<div class="doc-article-desc">
											After model conversion using MindSpore Lite, the model inference process needs to be completed in Runtime. This tutorial introduces how to use Java API to write inference code.
										</div>
									</div>
								</a>
							</div>
							<div class="doc-article-item all os-Linux os-Windows stage-Environment-Preparation stage-Benchmark-Testing user-Intermediate user-Expert hidden">
								<a href="https://www.mindspore.cn/tutorial/lite/en/master/use/benchmark_tool.html" class="article-link">
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
								<a href="https://www.mindspore.cn/tutorial/lite/en/master/use/cropper_tool.html" class="article-link">
									<div>
										<div class="doc-article-head">
											<span class="doc-head-content">Use Cropper Tool To Reduce Library File Size</span>
										</div>
										<div class="doc-article-desc">
											MindSpore Lite provides the libmindspore-lite.a static library cropping tool for runtime, which can filter out the operators in the ms model, crop the static library files, and effectively reduce the size of the library files.
										</div>
									</div>
								</a>
							</div>
							<div class="doc-article-item all os-Linux stage-Environment-Preparation stage-Model-Export stage-Model-Converting user-Intermediate user-Expert hidden">
								<a href="https://www.mindspore.cn/tutorial/lite/en/master/use/converter_train.html" class="article-link">
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
							<div class="doc-article-item all os-Linux os-Android stage-Model-Training stage-Model-Loading stage-Data-Preparation user-Intermediate user-Expert hidden">
								<a href="https://www.mindspore.cn/tutorial/lite/en/master/use/runtime_train_cpp.html" class="article-link">
									<div>
										<div class="doc-article-head">
											<span class="doc-head-content">Using runtime for Model Training and Inference (C++)</span>
										</div>
										<div class="doc-article-desc">
											After model conversion using MindSpore Lite, the model training process needs to be completed in Runtime. This tutorial introduces how to use C++ API to write training code.
										</div>
									</div>
								</a>
							</div>
							<div class="doc-article-item all os-Linux stage-Data-Preparation stage-Benchmark-Testing user-Intermediate user-Expert hidden">
								<a href="https://www.mindspore.cn/tutorial/lite/en/master/use/benchmark_train_tool.html" class="article-link">
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
		