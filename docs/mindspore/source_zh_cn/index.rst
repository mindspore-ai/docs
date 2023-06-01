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

   design/overview
   design/programming_paradigm
   design/auto_gradient
   design/mindir
   design/all_scenarios
   design/dynamic_graph_and_static_graph
   design/pluggable_device
   design/distributed_training_design
   design/graph_fusion_engine
   design/data_engine
   design/glossary

.. toctree::
   :glob:
   :maxdepth: 1
   :caption: 模型库
   :hidden:

   note/official_models

.. toctree::
   :glob:
   :maxdepth: 1
   :caption: 语法支持
   :hidden:

   note/static_graph_syntax_support
   note/index_support

.. toctree::
   :glob:
   :maxdepth: 1
   :caption: API
   :hidden:

   api_python/mindspore
   api_python/mindspore.nn
   api_python/mindspore.ops
   api_python/mindspore.ops.primitive
   api_python/mindspore.amp
   api_python/mindspore.train
   api_python/mindspore.communication
   api_python/mindspore.common.initializer
   api_python/mindspore.dataset
   api_python/mindspore.dataset.transforms
   api_python/mindspore.mindrecord
   api_python/mindspore.nn.probability
   api_python/mindspore.rewrite
   api_python/mindspore.boost
   api_python/mindspore.numpy
   api_python/mindspore.scipy
   C++ API↗ <https://www.mindspore.cn/lite/api/zh-CN/master/api_cpp/mindspore.html>

.. toctree::
   :glob:
   :maxdepth: 1
   :caption: API映射
   :hidden:

   note/api_mapping/pytorch_api_mapping
   note/api_mapping/tensorflow_api_mapping

.. toctree::
   :glob:
   :maxdepth: 1
   :caption: 迁移指南
   :titlesonly:
   :hidden:

   migration_guide/overview
   migration_guide/enveriment_preparation
   migration_guide/analysis_and_preparation
   migration_guide/model_development/model_development
   migration_guide/debug_and_tune
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
   faq/distributed_parallel
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
                     <a href="./design/overview.html" class="article-link">
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
                     <a href="./note/static_graph_syntax_support.html" class="article-link">
                        <div>
                           <div class="doc-article-head">
                              <span class="doc-head-content">规格</span>
                           </div>
                           <div class="doc-article-desc">
                              基准性能、网络支持、API支持和语法支持等规格说明。
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
                     <a href="./note/api_mapping/pytorch_api_mapping.html" class="article-link">
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
