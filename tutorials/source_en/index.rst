.. MindSpore documentation master file, created by
   sphinx-quickstart on Thu Mar 24 11:00:00 2021.
   You can adapt this file completely to your liking, but it should at least
   contain the root `toctree` directive.

MindSpore Tutorial
=========================

.. toctree::
   :glob:
   :maxdepth: 1
   :caption: Quick Start
   :hidden:

   introduction
   quick_start
   tensor
   dataset
   model
   autograd
   optimization
   save_load_model
   inference
   linear_regression

.. raw:: html

    <div class="container">
			<div class="row">
				<div class="col-md-12">
					<div>
						
						
						<div class="doc-condition">
							<span class="doc-filter">Filter</span>
							<button class="doc-delete doc-btn" id="all">Clear All</button>
						</div>
					
						<div class="doc-label-content">
							<div class="doc-label-choice">
								<div class="row">
								<div class="col-sm-2">
									<div class="doc-hardware">Hardware</div>
								</div>
								<div class="col-sm-10 col-sm-pull-1">
									<button class="doc-filter-btn doc-btn" id="hardware-ascend">Ascend</button>
									<button class="doc-filter-btn doc-btn" id="hardware-gpu">GPU</button>
									<button class="doc-filter-btn doc-btn" id="hardware-cpu">CPU</button>
									<button class="doc-filter-btn doc-btn" id="hardware-device">Device</button>
								</div>
							</div>
							</div>
							
							<div class="doc-label-choice">
								<div class="row">
									<div class="col-sm-2 ">
										<div class="doc-stage">Classification</div>
									</div>							
									<div class="col-sm-10 col-sm-pull-1 doc-stage-detail">
										<button class="doc-filter-btn doc-btn" id="stage-Beginner">Beginner</button>
                              <button class="doc-filter-btn doc-btn" id="stage-Whole-Process">Whole Process</button>
										<button class="doc-filter-btn doc-btn" id="stage-Data-Preparation">Data Preparation</button>
										<button class="doc-filter-btn doc-btn" id="stage-Model-Development">Model Development</button>
                              <button class="doc-filter-btn doc-btn" id="stage-Model-Development">Model Optimization</button>
                              <button class="doc-filter-btn doc-btn" id="stage-Model-Running">Model Running</button>
										<button class="doc-filter-btn doc-btn" id="stage-Model-Export">Model Export</button>
                              <button class="doc-filter-btn doc-btn" id="stage-Inference">stage Inference</button>
										<button class="doc-filter-btn doc-btn" id="stage-Model-Loading">Model Loading</button>
										<button class="doc-filter-btn doc-btn" id="stage-Distributed-Parallel">Distributed Parallel</button>
										<button class="doc-filter-btn doc-btn" id="stage-Function-Extension">Function Extension</button>
									</div>
									
								</div>
							</div>							
							<div class="doc-label-choice">
								<div class="row">
									<div class="col-sm-2 ">
										<div class="doc-experience">Experience</div>
									</div>							
									<div class="col-sm-10 col-sm-pull-1 doc-stage-detail">
										<button class="doc-filter-btn doc-btn" id="experience-online">Online</button>
										<button class="doc-filter-btn doc-btn" id="experience-local">Local</button>
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
                     <div class="doc-article-item all hardware-ascend hardware-gpu hardware-cpu stage-Beginner stage-Model-Development experience-local hidden">
                        <a href="https://www.mindspore.cn/tutorials/en/master/autograd.html" class="article-link">
                           <div>
                                 <div class="doc-article-head">
                                    <span class="doc-head-content">Automatic Differentiation</span>
                                 </div>
                                 <div class="doc-article-desc">
                                 Backward propagation is the commonly used algorithm for training neural networks. In this algorithm, parameters (model weights) are adjusted based on a gradient of a loss function for a given parameter.
                                 </div>
                           </div>
                        </a>
                     </div>
                     <div class="doc-article-item all hardware-ascend hardware-gpu hardware-cpu stage-Beginner stage-Data-Preparation experience-local hidden">
                        <a href="https://www.mindspore.cn/tutorials/en/master/dataset.html" class="article-link">
                           <div>
                                 <div class="doc-article-head">
                                    <span class="doc-head-content">Loading and Processing Data</span>
                                 </div>
                                 <div class="doc-article-desc">
                                 MindSpore provides APIs for loading common datasets and datasets in standard formats. You can directly use the corresponding dataset loading class in mindspore.dataset to load data. The dataset class provides common data processing APIs for users to quickly process data.
                                 </div>
                           </div>
                        </a>
                     </div>
                     <div class="doc-article-item all hardware-ascend stage-Beginner experience-local hidden">
                        <a href="https://www.mindspore.cn/tutorials/en/master/inference.html" class="article-link">
                           <div>
                                 <div class="doc-article-head">
                                    <span class="doc-head-content">Inference</span>
                                 </div>
                                 <div class="doc-article-desc">
                                 This is the last tutorial. To better adapt to different inference devices, inference is classified into Ascend AI Processor inference and mobile device inference.
                                 </div>
                           </div>
                        </a>
                     </div>
                     <div class="doc-article-item all hardware-ascend hardware-gpu hardware-cpu stage-Beginner experience-local hidden">
                        <a href="https://www.mindspore.cn/tutorials/en/master/introduction.html" class="article-link">
                           <div>
                                 <div class="doc-article-head">
                                    <span class="doc-head-content">Overview</span>
                                 </div>
                                 <div class="doc-article-desc">
                                 The following describes the Huawei AI full-stack solution and introduces the position of MindSpore in the solution. Developers who are interested in MindSpore can visit the [MindSpore community](https://gitee.com/mindspore/mindspore) and click [Watch, Star, and Fork](https://gitee.com/mindspore/mindspore).
                                 </div>
                           </div>
                        </a>
                     </div>
                     <div class="doc-article-item all experience-local hidden">
                        <a href="https://www.mindspore.cn/tutorials/en/master/linear_regression.html" class="article-link">
                           <div>
                                 <div class="doc-article-head">
                                    <span class="doc-head-content">Simple Linear Function Fitting</span>
                                 </div>
                                 <div class="doc-article-desc">
                                 Regression algorithms usually use a series of properties to predict a value, and the predicted values are consecutive. For example, the price of a house is predicted based on some given feature data of the house, such as area and the number of bedrooms; or future temperature conditions are predicted by using the temperature change data and satellite cloud images in the last week. If the actual price of the house is CNY5 million, and the value predicted through regression analysis is CNY4.99 million, the regression analysis is considered accurate. For machine learning problems, common regression analysis includes linear regression, polynomial regression, and logistic regression. This example describes the linear regression algorithms and how to use MindSpore to perform linear regression AI training.
                                 </div>
                           </div>
                        </a>
                     </div>
                     <div class="doc-article-item all hardware-ascend hardware-gpu hardware-cpu stage-Beginner stage-Model-Development experience-local hidden">
                        <a href="https://www.mindspore.cn/tutorials/en/master/model.html" class="article-link">
                           <div>
                                 <div class="doc-article-head">
                                    <span class="doc-head-content">Building a Neural Network</span>
                                 </div>
                                 <div class="doc-article-desc">
                                 A neural network model consists of multiple data operation layers. `mindspore.nn` provides various basic network modules.
                                 </div>
                           </div>
                        </a>
                     </div>
                     <div class="doc-article-item all hardware-ascend hardware-gpu stage-Beginner stage-Model-Optimization experience-local hidden">
                        <a href="https://www.mindspore.cn/tutorials/en/master/optimization.html" class="article-link">
                           <div>
                                 <div class="doc-article-head">
                                    <span class="doc-head-content">Optimizing Model Parameters</span>
                                 </div>
                                 <div class="doc-article-desc">
                                 After learning how to create a model and build a dataset in the preceding tutorials, you can start to learn how to set hyperparameters and optimize model parameters.
                                 </div>
                           </div>
                        </a>
                     </div>
                     <div class="doc-article-item all hardware-ascend hardware-gpu hardware-cpu stage-Beginner stage-Whole-Process experience-local hidden">
                        <a href="https://www.mindspore.cn/tutorials/en/master/quick_start.html" class="article-link">
                           <div>
                                 <div class="doc-article-head">
                                    <span class="doc-head-content">Quick Start for Beginners</span>
                                 </div>
                                 <div class="doc-article-desc">
                                 The following describes the basic functions of MindSpore to implement common tasks in deep learning. For details, see links in each section.
                                 </div>
                           </div>
                        </a>
                     </div>
                     <div class="doc-article-item all hardware-ascend hardware-gpu hardware-cpu stage-Beginner stage-Model-Export stage-Model-Loading experience-local hidden">
                        <a href="https://www.mindspore.cn/tutorials/en/master/save_load_model.html" class="article-link">
                           <div>
                                 <div class="doc-article-head">
                                    <span class="doc-head-content">Saving and Loading the Model</span>
                                 </div>
                                 <div class="doc-article-desc">
                                 In the previous tutorial, you learn how to train the network. In this tutorial, you will learn how to save and load a model, and how to export a saved model in a specified format to different platforms for inference.
                                 </div>
                           </div>
                        </a>
                     </div>
                     <div class="doc-article-item all hardware-ascend hardware-gpu hardware-cpu stage-Beginner experience-local hidden">
                        <a href="https://www.mindspore.cn/tutorials/en/master/tensor.html" class="article-link">
                           <div>
                                 <div class="doc-article-head">
                                    <span class="doc-head-content">Tensor</span>
                                 </div>
                                 <div class="doc-article-desc">
                                 Tensor is a basic data structure in the MindSpore network computing.
                                 </div>
                           </div>
                        </a>
                     </div>

						</div>
				
					</div>
					
				</div>
			</div>
		</div>
