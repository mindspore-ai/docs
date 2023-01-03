.. MindSpore documentation master file, created by
   sphinx-quickstart on Thu Mar 24 11:00:00 2022.
   You can adapt this file completely to your liking, but it should at least
   contain the root `toctree` directive.

MindSpore Documentation
=======================

.. toctree::
   :glob:
   :maxdepth: 1
   :caption: Design
   :hidden:

   design/overview
   design/auto_gradient
   design/distributed_training_design
   design/dynamic_graph_and_static_graph
   design/graph_fusion_engine
   design/mindir
   design/all_scenarios
   design/thor
   Design of Visualization↗ <https://www.mindspore.cn/mindinsight/docs/en/r2.0.0-alpha/training_visual_design.html>
   design/data_engine
   design/glossary

.. toctree::
   :glob:
   :maxdepth: 1
   :caption: Specification
   :hidden:

   note/benchmark
   Network List↗ <https://gitee.com/mindspore/models/blob/r2.0.0-alpha/README.md#table-of-contents>
   note/operator_list
   note/syntax_list

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
   api_python/mindspore.nn.transformer
   api_python/mindspore.rewrite
   api_python/mindspore.boost
   api_python/mindspore.numpy
   api_python/mindspore.scipy
   C++ API↗ <https://www.mindspore.cn/lite/api/en/r2.0.0-alpha/api_cpp/mindspore.html>

.. toctree::
   :glob:
   :maxdepth: 1
   :caption: API Mapping
   :hidden:

   note/api_mapping/pytorch_api_mapping
   note/api_mapping/tensorflow_api_mapping

.. toctree::
   :glob:
   :maxdepth: 1
   :caption: Migration Guide
   :titlesonly:
   :hidden:

   migration_guide/overview
   migration_guide/enveriment_preparation
   migration_guide/analysis_and_preparation
   migration_guide/model_development/model_development
   migration_guide/debug_and_tune
   migration_guide/sample_code
   migration_guide/faq
   migration_guide/typical_api_comparision
   migration_guide/use_third_party_op

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
                     <a href="./design/auto_gradient.html" class="article-link">
                        <div>
                           <div class="doc-article-head">
                              <span class="doc-head-content">Design</span>
                           </div>
                           <div class="doc-article-desc">
                              The design concept of MindSpore's main functions to help framework developers better understand the overall architecture.
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
                              <span class="doc-head-content">Specifications</span>
                           </div>
                           <div class="doc-article-desc">
                              Specifications of benchmark performance, network support, API support and syntax support.
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
                              MindSpore API description list.
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
                              <span class="doc-head-content">API Mapping</span>
                           </div>
                           <div class="doc-article-desc">
                              API mapping between PyTorch and MindSpore, TensorFlow and MindSpore provided by the community.
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
                              <span class="doc-head-content">Migration Guide</span>
                           </div>
                           <div class="doc-article-desc">
                              The complete steps and considerations for migrating neural networks from other machine learning frameworks to MindSpore.
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
                              Frequently asked questions and answers, including installation, data processing, compilation and execution, debugging and tuning, distributed parallelism, inference, etc.
                           </div>
                        </div>
                     </a>
                  </div>
					</div>
				</div>
		   </div>
	</div>
