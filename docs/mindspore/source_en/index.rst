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
   :caption: Models
   :hidden:

   note/official_models

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
   C++ API↗ <https://www.mindspore.cn/lite/api/en/master/api_cpp/mindspore.html>

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

.. toctree::
   :glob:
   :maxdepth: 1
   :caption: Syntax Support
   :hidden:

   note/static_graph_syntax_support
   note/index_support

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
                     <a href="./note/official_models.html" class="article-link">
                        <div>
                           <div class="doc-article-head">
                              <span class="doc-head-content">Model Libraries</span>
                           </div>
                           <div class="doc-article-desc">
                              Contains model examples and performance data for different domains.
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
                     <a href="./note/static_graph_syntax_support.html" class="article-link">
                        <div>
                           <div class="doc-article-head">
                              <span class="doc-head-content">Syntax Support</span>
                           </div>
                           <div class="doc-article-desc">
                              Syntax support for static graphs, Tensor indexes, etc.
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
            <div class="col-md-6">
               <div class="doc-article-list">
                  <div class="doc-article-item">
                     <a href="./RELEASE.html" class="article-link">
                        <div>
                           <div class="doc-article-head">
                              <span class="doc-head-content">RELEASE NOTES</span>
                           </div>
                           <div class="doc-article-desc">
                              Contains information on major features and augments, API changes for the release versions.
                           </div>
                        </div>
                     </a>
                  </div>
					</div>
				</div>
		   </div>
	</div>
