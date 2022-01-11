.. MindSpore documentation master file, created by
   sphinx-quickstart on Thu Mar 24 11:00:00 2021.
   You can adapt this file completely to your liking, but it should at least
   contain the root `toctree` directive.

MindSpore教程
=====================

.. toctree::
   :glob:
   :maxdepth: 1
   :caption: 入门教程
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

.. toctree::
   :glob:
   :maxdepth: 1
   :caption: 进阶教程
   :hidden:

   intermediate/mid_low_level_api
   intermediate/data
   image_and_video
   text
   intermediate/pynative_mode_and_graph_mode
   distributed_training
   inference_and_deploy

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
                              <button class="doc-filter-btn doc-btn" id="stage-Advanced">进阶</button>
                              <button class="doc-filter-btn doc-btn" id="stage-Whole-Process">全流程</button>
										<button class="doc-filter-btn doc-btn" id="stage-Data-Preparation">数据准备</button>
										<button class="doc-filter-btn doc-btn" id="stage-Model-Development">模型开发</button>
                              <button class="doc-filter-btn doc-btn" id="stage-Model-Running">模型运行</button>
										<button class="doc-filter-btn doc-btn" id="stage-Model-Export">模型导出</button>
                              <button class="doc-filter-btn doc-btn" id="stage-Inference">推理应用</button>
										<button class="doc-filter-btn doc-btn" id="stage-Model-Loading">模型加载</button>
										<button class="doc-filter-btn doc-btn" id="stage-Distributed-Parallel">分布式并行</button>
										<button class="doc-filter-btn doc-btn" id="stage-Function-Extension">功能扩展</button>
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
						<hr>

						<div class="doc-footer">
							<nav aria-label="Page navigation">
								<ul class="pagination" id="pageNav">
									
								</ul>
							</nav>
						</div>

                  <div class="doc-article-list">
                     <div class="doc-article-item all hardware-ascend hardware-gpu hardware-cpu stage-Beginner stage-Model-Development experience-local experience-online hidden">
                        <a href="https://www.mindspore.cn/tutorials/zh-CN/r1.6/autograd.html" class="article-link">
                           <div>
                                 <div class="doc-article-head">
                                    <span class="doc-head-content">自动微分</span>
                                 </div>
                                 <div class="doc-article-desc">
                                 在训练神经网络时，最常用的算法是反向传播，在该算法中，根据损失函数对于给定参数的梯度来调整参数（模型权重）。
                                 </div>
                           </div>
                        </a>
                     </div>
                     <div class="doc-article-item all hardware-ascend hardware-gpu hardware-cpu stage-Beginner stage-Data-Preparation experience-local experience-online hidden">
                        <a href="https://www.mindspore.cn/tutorials/zh-CN/r1.6/dataset.html" class="article-link">
                           <div>
                                 <div class="doc-article-head">
                                    <span class="doc-head-content">数据加载及处理</span>
                                 </div>
                                 <div class="doc-article-desc">
                                 MindSpore提供了部分常用数据集和标准格式数据集的加载接口，用户可以直接使用<em>mindspore.dataset</em>中对应的数据集加载类进行数据加载。
                                 </div>
                           </div>
                        </a>
                     </div>
                     <div class="doc-article-item all hardware-ascend hardware-device stage-Beginner stage-Inference experience-local hidden">
                        <a href="https://www.mindspore.cn/tutorials/zh-CN/r1.6/inference.html" class="article-link">
                           <div>
                                 <div class="doc-article-head">
                                    <span class="doc-head-content">推理</span>
                                 </div>
                                 <div class="doc-article-desc">
                                 本节是初级教程的最后一节，为了更好地适配不同推理设备，因此推理分为 1）昇腾AI处理器推理和 2）移动设备推理。
                                 </div>
                           </div>
                        </a>
                     </div>
                     <div class="doc-article-item all hardware-ascend hardware-gpu hardware-cpu hardware-device stage-Beginner experience-local hidden">
                        <a href="https://www.mindspore.cn/tutorials/zh-CN/r1.6/introduction.html" class="article-link">
                           <div>
                                 <div class="doc-article-head">
                                    <span class="doc-head-content">基本介绍</span>
                                 </div>
                                 <div class="doc-article-desc">
                                 本节将会对华为AI全栈进行整体介绍，并介绍MindSpore在其中的位置，对MindSpore感兴趣的开发者，最后可以参与MindSpore的社区。
                                 </div>
                           </div>
                        </a>
                     </div>
                     <div class="doc-article-item all hardware-ascend hardware-gpu hardware-cpu stage-Beginner stage-Whole-Process experience-local experience-online hidden">
                        <a href="https://www.mindspore.cn/tutorials/zh-CN/r1.6/linear_regression.html" class="article-link">
                           <div>
                                 <div class="doc-article-head">
                                    <span class="doc-head-content">简单线性函数拟合</span>
                                 </div>
                                 <div class="doc-article-desc">
                                 回归问题算法通常是利用一系列属性来预测一个值，预测的值是连续的。本例子介绍线性回归算法，并通过MindSpore进行线性回归AI训练体验。
                                 </div>
                           </div>
                        </a>
                     </div>
                     <div class="doc-article-item all hardware-ascend hardware-gpu hardware-cpu stage-Beginner stage-Model-Development experience-local experience-online hidden">
                        <a href="https://www.mindspore.cn/tutorials/zh-CN/r1.6/model.html" class="article-link">
                           <div>
                                 <div class="doc-article-head">
                                    <span class="doc-head-content">建立神经网络</span>
                                 </div>
                                 <div class="doc-article-desc">
                                 神经网络模型由多个数据操作层组成，<em>mindspore.nn</em>提供了各种网络基础模块。
                                 </div>
                           </div>
                        </a>
                     </div>
                     <div class="doc-article-item all hardware-ascend hardware-gpu stage-Beginner stage-Model-Development experience-local experience-online hidden">
                        <a href="https://www.mindspore.cn/tutorials/zh-CN/r1.6/optimization.html" class="article-link">
                           <div>
                                 <div class="doc-article-head">
                                    <span class="doc-head-content">训练模型</span>
                                 </div>
                                 <div class="doc-article-desc">
                                 通过上面章节的学习，我们已经学会如何创建模型和构建数据集，现在开始学习如何设置超参和优化模型参数。
                                 </div>
                           </div>
                        </a>
                     </div>
                     <div class="doc-article-item all hardware-ascend hardware-gpu hardware-cpu stage-Beginner stage-Whole-Process experience-local experience-online hidden">
                        <a href="https://www.mindspore.cn/tutorials/zh-CN/r1.6/quick_start.html" class="article-link">
                           <div>
                                 <div class="doc-article-head">
                                    <span class="doc-head-content">初学入门</span>
                                 </div>
                                 <div class="doc-article-desc">
                                 本节贯穿MindSpore的基础功能，实现深度学习中的常见任务，请参考各节链接进行更加深入的学习。
                                 </div>
                           </div>
                        </a>
                     </div>
                     <div class="doc-article-item all hardware-ascend hardware-gpu hardware-cpu stage-Beginner stage-Model-Export stage-Model-Loading experience-local hidden">
                        <a href="https://www.mindspore.cn/tutorials/zh-CN/r1.6/save_load_model.html" class="article-link">
                           <div>
                                 <div class="doc-article-head">
                                    <span class="doc-head-content">保存及加载模型</span>
                                 </div>
                                 <div class="doc-article-desc">
                                 上一节我们训练完网络，本节将会学习如何保存模型和加载模型，以及如何将保存的模型导出成特定格式到不同平台进行推理。
                                 </div>
                           </div>
                        </a>
                     </div>
                     <div class="doc-article-item all hardware-ascend hardware-gpu hardware-cpu stage-Beginner experience-local experience-online hidden">
                        <a href="https://www.mindspore.cn/tutorials/zh-CN/r1.6/tensor.html" class="article-link">
                           <div>
                                 <div class="doc-article-head">
                                    <span class="doc-head-content">张量</span>
                                 </div>
                                 <div class="doc-article-desc">
                                 张量（Tensor）是MindSpore网络运算中的基本数据结构。
                                 </div>
                           </div>
                        </a>
                     </div>
                     <div class="doc-article-item all hardware-ascend hardware-gpu hardware-cpu stage-Data-Preparation stage-Advanced experience-local experience-online hidden">
                        <a href="https://www.mindspore.cn/tutorials/zh-CN/r1.6/intermediate/data.html" class="article-link">
                           <div>
                                 <div class="doc-article-head">
                                    <span class="doc-head-content">高级数据集管理</span>
                                 </div>
                                 <div class="doc-article-desc">
                                 MindSpore可以加载常见的数据集或自定义的数据集，这部分功能在初级教程中进行了部分介绍。
                                 </div>
                           </div>
                        </a>
                     </div>
                     <div class="doc-article-item all hardware-ascend hardware-gpu hardware-cpu stage-Whole-Process stage-Advanced experience-local hidden">
                        <a href="https://www.mindspore.cn/tutorials/zh-CN/r1.6/intermediate/mid_low_level_api.html" class="article-link">
                           <div>
                                 <div class="doc-article-head">
                                    <span class="doc-head-content">中低阶API实现深度学习</span>
                                 </div>
                                 <div class="doc-article-desc">
                                 为方便用户控制整网的执行流程，MindSpore提供了高阶的训练和推理接口<em>mindspore.Model</em>，通过指定要训练的神经网络模型和常见的训练设置，调用<em>train</em>和<em>eval</em>方法对网络进行训练和推理。
                                 </div>
                           </div>
                        </a>
                     </div>
                     <div class="doc-article-item all hardware-ascend hardware-gpu hardware-cpu stage-Model-Running stage-Advanced experience-local experience-online hidden">
                        <a href="https://www.mindspore.cn/tutorials/zh-CN/r1.6/intermediate/pynative_mode_and_graph_mode.html" class="article-link">
                           <div>
                                 <div class="doc-article-head">
                                    <span class="doc-head-content">动态图与静态图</span>
                                 </div>
                                 <div class="doc-article-desc">
                                 MindSpore支持两种运行模式：动态图与静态图。
                                 </div>
                           </div>
                        </a>
                     </div>
                     <div class="doc-article-item all hardware-ascend stage-Distributed-Parallel stage-Advanced experience-local hidden">
                        <a href="https://www.mindspore.cn/tutorials/zh-CN/r1.6/intermediate/distributed_training/apply_parameter_server_training.html" class="article-link">
                           <div>
                                 <div class="doc-article-head">
                                    <span class="doc-head-content">使用Parameter Server训练</span>
                                 </div>
                                 <div class="doc-article-desc">
                                 Parameter Server(参数服务器)是分布式训练中一种广泛使用的架构，相较于同步的AllReduce训练方法，Parameter Server具有更好的灵活性、可扩展性以及节点容灾的能力。
                                 </div>
                           </div>
                        </a>
                     </div>
                     <div class="doc-article-item all hardware-ascend stage-Distributed-Parallel stage-Advanced experience-local hidden">
                        <a href="https://www.mindspore.cn/tutorials/zh-CN/r1.6/intermediate/distributed_training/distributed_training_ascend.html" class="article-link">
                           <div>
                                 <div class="doc-article-head">
                                    <span class="doc-head-content">分布式并行训练 （Ascend）</span>
                                 </div>
                                 <div class="doc-article-desc">
                                 在深度学习中，分布式并行训练，可以降低对内存、计算性能等硬件的需求，是进行训练的重要优化手段。
                                 </div>
                           </div>
                        </a>
                     </div>
                     <div class="doc-article-item all hardware-gpu stage-Distributed-Parallel stage-Advanced experience-local hidden">
                        <a href="https://www.mindspore.cn/tutorials/zh-CN/r1.6/intermediate/distributed_training/distributed_training_gpu.html" class="article-link">
                           <div>
                                 <div class="doc-article-head">
                                    <span class="doc-head-content">分布式并行训练 （GPU）</span>
                                 </div>
                                 <div class="doc-article-desc">
                                 本篇教程我们主要讲解，如何在GPU硬件平台上，利用MindSpore的数据并行及自动并行模式训练ResNet-50网络。
                                 </div>
                           </div>
                        </a>
                     </div>
                     <div class="doc-article-item all hardware-ascend hardware-gpu stage-Distributed-Parallel stage-Model-Export stage-Advanced experience-local hidden">
                        <a href="https://www.mindspore.cn/tutorials/zh-CN/r1.6/intermediate/distributed_training/distributed_training_model_parameters_saving_and_loading.html" class="article-link">
                           <div>
                                 <div class="doc-article-head">
                                    <span class="doc-head-content">分布式训练模型参数保存和加载</span>
                                 </div>
                                 <div class="doc-article-desc">
                                 本章将会讲解在Ascend与GPU环境中进行分布式训练时，如何进行参数的保存与加载。
                                 </div>
                           </div>
                        </a>
                     </div>
                     <div class="doc-article-item all hardware-ascend hardware-gpu hardware-cpu stage-Whole-Process stage-Advanced experience-local experience-online hidden">
                        <a href="https://www.mindspore.cn/tutorials/zh-CN/r1.6/intermediate/image_and_video/adversarial_example_generation.html" class="article-link">
                           <div>
                                 <div class="doc-article-head">
                                    <span class="doc-head-content">对抗示例生成</span>
                                 </div>
                                 <div class="doc-article-desc">
                                 在本案例中，我们将以梯度符号攻击FGSM（Fast Gradient Sign Method）为例，演示此类攻击是如何误导模型的。
                                 </div>
                           </div>
                        </a>
                     </div>
                     <div class="doc-article-item all hardware-ascend hardware-gpu stage-Whole-Process stage-Advanced experience-local experience-online hidden">
                        <a href="https://www.mindspore.cn/tutorials/zh-CN/r1.6/intermediate/image_and_video/dcgan.html" class="article-link">
                           <div>
                                 <div class="doc-article-head">
                                    <span class="doc-head-content">深度卷积对抗生成网络</span>
                                 </div>
                                 <div class="doc-article-desc">
                                 生成对抗网络（GAN, Generative Adversarial Networks ）是一种深度学习模型，是近年来复杂分布上无监督学习最具前景的方法之一。
                                 </div>
                           </div>
                        </a>
                     </div>
                     <div class="doc-article-item all hardware-ascend hardware-gpu stage-Whole-Process stage-Advanced experience-local hidden">
                        <a href="https://www.mindspore.cn/tutorials/zh-CN/r1.6/intermediate/image_and_video/transfer_learning.html" class="article-link">
                           <div>
                                 <div class="doc-article-head">
                                    <span class="doc-head-content">图像分类迁移学习</span>
                                 </div>
                                 <div class="doc-article-desc">
                                 在实际场景中，为了减少从头开始训练所带来的时间成本，大多数情况下会基于已有的模型来进行迁移学习。本章将会以狗和狼的图像分类为例，讲解如何在MindSpore中加载预训练模型，并通过固定权重来实现迁移学习的目的。
                                 </div>
                           </div>
                        </a>
                     </div>
                     <div class="doc-article-item all hardware-ascend stage-Inference stage-Advanced experience-local hidden">
                        <a href="https://www.mindspore.cn/tutorials/zh-CN/r1.6/intermediate/inference/ascend310_inference.html" class="article-link">
                           <div>
                                 <div class="doc-article-head">
                                    <span class="doc-head-content">Ascend310处理器上推理MindIR模型</span>
                                 </div>
                                 <div class="doc-article-desc">
                                 本文介绍如何在Ascend310处理器中推理MindIR模型。
                                 </div>
                           </div>
                        </a>
                     </div>
                     <div class="doc-article-item all hardware-ascend hardware-gpu stage-Inference stage-Advanced experience-local experience-online hidden">
                        <a href="https://www.mindspore.cn/tutorials/zh-CN/r1.6/intermediate/inference/ascend910_and_gpu_inference.html" class="article-link">
                           <div>
                                 <div class="doc-article-head">
                                    <span class="doc-head-content">Ascend910与GPU推理</span>
                                 </div>
                                 <div class="doc-article-desc">
                                 本文将介绍如何在Ascend910和GPU硬件环境中，利用MindIR和Checkpoint执行推理。
                                 </div>
                           </div>
                        </a>
                     </div>
                     <div class="doc-article-item all hardware-device stage-Inference stage-Advanced experience-local hidden">
                        <a href="https://www.mindspore.cn/tutorials/zh-CN/r1.6/intermediate/inference/mindspore_lite_inference.html" class="article-link">
                           <div>
                                 <div class="doc-article-head">
                                    <span class="doc-head-content">MindSpore Lite推理流程</span>
                                 </div>
                                 <div class="doc-article-desc">
                                 本文将会以图像分割Demo为例讲解如何使用MindSpore Lite进行推理。
                                 </div>
                           </div>
                        </a>
                     </div>
                     <div class="doc-article-item all hardware-ascend hardware-gpu stage-Inference stage-Advanced experience-local hidden">
                        <a href="https://www.mindspore.cn/tutorials/zh-CN/r1.6/intermediate/inference/mindspore_serving_inference.html" class="article-link">
                           <div>
                                 <div class="doc-article-head">
                                    <span class="doc-head-content">基于MindSpore Serving部署推理服务</span>
                                 </div>
                                 <div class="doc-article-desc">
                                 MindSpore Serving是一个轻量级、高性能的推理服务模块，旨在帮助MindSpore开发者在生产环境中高效部署在线推理服务。
                                 </div>
                           </div>
                        </a>
                     </div>
                     <div class="doc-article-item all hardware-ascend stage-Whole-Process stage-Advanced experience-local hidden">
                        <a href="https://www.mindspore.cn/tutorials/zh-CN/r1.6/intermediate/text/bert_poetry.html" class="article-link">
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
                     <div class="doc-article-item all hardware-ascend hardware-gpu stage-Whole-Process stage-Advanced experience-local experience-online hidden">
                        <a href="https://www.mindspore.cn/tutorials/zh-CN/r1.6/intermediate/text/rnn_classification.html" class="article-link">
                           <div>
                                 <div class="doc-article-head">
                                    <span class="doc-head-content">使用字符级RNN分类名称</span>
                                 </div>
                                 <div class="doc-article-desc">
                                 循环神经网络（Recurrent Neural Network, RNN）是一类以序列（sequence）数据为输入，在序列的演进方向进行递归（recursion）且所有节点（循环单元）按链式连接的递归神经网络（recursive neural network），常用于NLP领域当中来解决序列化数据的建模问题。
                                 </div>
                           </div>
                        </a>
                     </div>
                     <div class="doc-article-item all hardware-ascend hardware-gpu stage-Whole-Process stage-Advanced experience-local experience-online hidden">
                        <a href="https://www.mindspore.cn/tutorials/zh-CN/r1.6/intermediate/text/rnn_generation.html" class="article-link">
                           <div>
                                 <div class="doc-article-head">
                                    <span class="doc-head-content">使用字符级RNN生成名称</span>
                                 </div>
                                 <div class="doc-article-desc">
                                 本教程中，我们将通过反向操作来生成不同语言的名称。这里仍通过编写由线性层结构构建出的小型RNN网络模型来实现目标。
                                 </div>
                           </div>
                        </a>
                     </div>
                     <div class="doc-article-item all hardware-gpu hardware-cpu stage-Whole-Process stage-Advanced experience-local experience-online hidden">
                        <a href="https://www.mindspore.cn/tutorials/zh-CN/r1.6/intermediate/text/sentimentnet.html" class="article-link">
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

						</div>
				
					</div>
					
				</div>
			</div>
		</div>
		
