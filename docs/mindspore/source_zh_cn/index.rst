.. MindSpore documentation master file, created by
   sphinx-quickstart on Thu Mar 24 11:00:00 2022.
   You can adapt this file completely to your liking, but it should at least
   contain the root `toctree` directive.

MindSpore 文档
=========================

.. toctree::
   :glob:
   :maxdepth: 1
   :caption: 设计
   :hidden:

   design/white_paper
   design/all_scenarios
   design/gradient
   design/dynamic_graph_and_static_graph
   design/heterogeneous_training
   design/distributed_training_design
   design/mindir
   design/data_engine
   design/graph_fusion_engine
   design/thor
   可视化调试调优↗ <https://www.mindspore.cn/mindinsight/docs/zh-CN/master/training_visual_design.html>
   安全可信↗ <https://www.mindspore.cn/mindarmour/docs/zh-CN/master/design.html>
   design/glossary
   
.. toctree::
   :glob:
   :maxdepth: 1
   :caption: 参考
   :hidden:

   note/benchmark
   note/network_list
   note/operator_list
   note/env_var_list
   note/syntax_list
   note/api_mapping

.. toctree::
   :glob:
   :maxdepth: 1
   :caption: API
   :hidden:

   api_python/mindspore
   api_python/mindspore.common.initializer
   api_python/mindspore.communication
   api_python/mindspore.context
   api_python/mindspore.dataset
   api_python/mindspore.dataset.audio
   api_python/mindspore.dataset.config
   api_python/mindspore.dataset.text
   api_python/mindspore.dataset.transforms
   api_python/mindspore.dataset.vision
   api_python/mindspore.mindrecord
   api_python/mindspore.nn
   api_python/mindspore.nn.probability
   api_python/mindspore.nn.transformer
   api_python/mindspore.numpy
   api_python/mindspore.ops
   api_python/mindspore.ops.functional
   api_python/mindspore.parallel
   api_python/mindspore.parallel.nn
   api_python/mindspore.profiler
   api_python/mindspore.scipy
   api_python/mindspore.train
   api_python/mindspore.boost
   C++ API↗ <https://www.mindspore.cn/lite/api/zh-CN/master/api_cpp/mindspore.html>

.. toctree::
   :glob:
   :maxdepth: 1
   :caption: 迁移指南
   :hidden:

   migration_guide/overview
   migration_guide/preparation
   migration_guide/script_analysis
   migration_guide/script_development
   migration_guide/neural_network_debug
   migration_guide/accuracy_optimization
   migration_guide/performance_optimization
   migration_guide/inference
   migration_guide/sample_code
   migration_guide/faq
   
.. toctree::
   :glob:
   :maxdepth: 1
   :caption: FAQ
   :hidden:

   faq/installation
   faq/data_processing
   faq/implement_problem
   faq/network_compilation
   faq/operators_compile
   faq/usage_migrate_3rd
   faq/performance_tuning
   faq/precision_tuning
   faq/distributed_configure
   faq/inference
   faq/feature_advice

.. toctree::
   :glob:
   :maxdepth: 1
   :caption: RELEASE NOTES
   :hidden:

   RELEASE

.. raw:: html

   <div class="container">
			<div class="row">
				<div class="col-md-6">
               <div class="doc-article-list">
                  <div class="doc-article-item">
                     <a href="./design/technical_white_paper.html" class="article-link">
                        <div>
                           <div class="doc-article-head">
                              <span class="doc-head-content">设计</span>
                           </div>
                           <div class="doc-article-desc">
                              MindSpore主要功能的设计理念，帮助框架开发者更好地理解整体架构。
                           </div>
                        </div>
                     </a>
                  </div>
					</div>
				</div>
            <div class="col-md-6">
               <div class="doc-article-list">
                  <div class="doc-article-item">
                     <a href="./note/syntax_list.html" class="article-link">
                        <div>
                           <div class="doc-article-head">
                              <span class="doc-head-content">语法支持</span>
                           </div>
                           <div class="doc-article-desc">
                              静态图语法、Tensor索引等的支持情况。
                           </div>
                        </div>
                     </a>
                  </div>
					</div>
				</div>
         </div>
         <div class="row">
            <div class="col-md-6">
               <div class="doc-article-list">
                  <div class="doc-article-item">
                     <a href="./api_python/mindspore.html" class="article-link">
                        <div>
                           <div class="doc-article-head">
                              <span class="doc-head-content">API</span>
                           </div>
                           <div class="doc-article-desc">
                              MindSpore API说明列表。
                           </div>
                        </div>
                     </a>
                  </div>
					</div>
				</div>
				<div class="col-md-6">
               <div class="doc-article-list">
                  <div class="doc-article-item">
                     <a href="./note/api_mapping.html" class="article-link">
                        <div>
                           <div class="doc-article-head">
                              <span class="doc-head-content">API映射</span>
                           </div>
                           <div class="doc-article-desc">
                              由社区提供的PyTorch与MindSpore、TensorFlow与MindSpore之间的API映射。
                           </div>
                        </div>
                     </a>
                  </div>
					</div>
				</div>
         </div>
         <div class="row">
            <div class="col-md-6">
               <div class="doc-article-list">
                  <div class="doc-article-item">
                     <a href="./migration_guide/overview.html" class="article-link">
                        <div>
                           <div class="doc-article-head">
                              <span class="doc-head-content">迁移指南</span>
                           </div>
                           <div class="doc-article-desc">
                              从其他机器学习框架将神经网络迁移到MindSpore的完整步骤和注意事项。
                           </div>
                        </div>
                     </a>
                  </div>
					</div>
				</div>
            <div class="col-md-6">
               <div class="doc-article-list">
                  <div class="doc-article-item">
                     <a href="./faq/installation.html" class="article-link">
                        <div>
                           <div class="doc-article-head">
                              <span class="doc-head-content">FAQ</span>
                           </div>
                           <div class="doc-article-desc">
                              常见问题与解答，包括安装、数据处理、编译执行、调试调优、分布式并行、推理等。
                           </div>
                        </div>
                     </a>
                  </div>
					</div>
				</div>
		   </div>
	</div>