.. MindSpore documentation master file, created by
   sphinx-quickstart on Thu Aug 17 09:00:00 2020.
   You can adapt this file completely to your liking, but it should at least
   contain the root `toctree` directive.

在手机或IoT设备上使用MindSpore
=================================

.. toctree::
   :glob:
   :maxdepth: 1
   :caption: 快速入门
   :hidden:

   quick_start/quick_start

.. toctree::
   :glob:
   :maxdepth: 1
   :caption: 基础使用
   :hidden:

   use/preparation
   use/convert_model
   use/post_training_quantization
   use/data_preprocessing
   use/runtime
   use/tools

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
										<button class="doc-filter-btn doc-btn" id="os-Windows">Windows</button>
										<button class="doc-filter-btn doc-btn" id="os-Linux">Linux</button>
										<button class="doc-filter-btn doc-btn" id="os-Android">Android</button>
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
										<button class="doc-filter-btn doc-btn" id="stage-Model-Converting">模型转换</button>
										<button class="doc-filter-btn doc-btn" id="stage-Model-Optimization">模型调优</button>
										<button class="doc-filter-btn doc-btn" id="stage-Inference">推理应用</button>
									</div>
								</div>
							</div>
							
						</div>
						<hr>
						<div class="doc-article-list">
							<div class="doc-article-item all os-Windows os-Linux os-Android stage-Whole-Process stage-Model-Converting stage-Model-Loading stage-Inference stage-Data-Preparation user-Beginner user-Intermediate user-Expert hidden">
								<a href="https://www.mindspore.cn/tutorial/lite/zh-CN/master/quick_start/quick_start.html" class="article-link">
									<div>
										<div class="doc-article-head">
											<span class="doc-head-content">实现一个图像分类应用</span>
										</div>
							
										<div class="doc-article-desc">
											本教程从端侧Android图像分类demo入手，帮助用户了解MindSpore Lite应用工程的构建、依赖项配置以及相关API的使用。
										</div>
									</div>
								</a>
							</div>
							<div class="doc-article-item all os-Windows os-Linux os-Android stage-Environment-Preparation user-Beginner user-Intermediate user-Expert hidden">
								<a href="https://www.mindspore.cn/tutorial/lite/zh-CN/master/use/downloads.html" class="article-link">
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
								<a href="https://www.mindspore.cn/tutorial/lite/zh-CN/master/use/build.html" class="article-link">
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
								<a href="https://www.mindspore.cn/tutorial/lite/zh-CN/master/use/converter_tool.html" class="article-link">
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
							<div class="doc-article-item all os-Windows os-Linux stage-Model-Converting stage-Model-Optimization user-Intermediate user-Expert hidden">
								<a href="https://www.mindspore.cn/tutorial/lite/zh-CN/master/use/post_training_quantization.html" class="article-link">
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
							<div class="doc-article-item all os-Windows os-Linux stage-Data-Preparation user-Beginner user-Intermediate user-Expert hidden">
								<a href="https://www.mindspore.cn/tutorial/lite/zh-CN/master/use/image_processing.html" class="article-link">
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
							<div class="doc-article-item all os-Windows os-Linux os-Android stage-Inference stage-Model-Loading stage-Data-Preparation user-Intermediate user-Expert hidden">
								<a href="https://www.mindspore.cn/tutorial/lite/zh-CN/master/use/runtime_cpp.html" class="article-link">
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
							<div class="doc-article-item all os-Windows os-Linux os-Android stage-Inference stage-Model-Loading stage-Data-Preparation user-Intermediate user-Expert hidden">
								<a href="https://www.mindspore.cn/tutorial/lite/zh-CN/master/use/runtime_java.html" class="article-link">
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
		