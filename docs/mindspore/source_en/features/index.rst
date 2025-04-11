Feature Description
=================================

.. toctree::
   :glob:
   :maxdepth: 1
   :hidden:
   :caption: Programming Forms

   program_form/overview

.. toctree::
   :glob:
   :maxdepth: 1
   :hidden:
   :caption: Data Processing

   dataset/overview

.. toctree::
   :glob:
   :maxdepth: 1
   :hidden:
   :caption: Distributed Parallelism

   parallel/data_parallel
   parallel/operator_parallel
   parallel/optimizer_parallel
   parallel/pipeline_parallel
   parallel/auto_parallel

.. toctree::
   :glob:
   :maxdepth: 1
   :hidden:
   :caption: Compile

   compile/graph_construction
   compile/graph_optimization

.. toctree::
   :glob:
   :maxdepth: 1
   :hidden:
   :caption: Runtime

   runtime/memory_manager
   runtime/multilevel_pipeline
   runtime/multistream_concurrency
   runtime/pluggable_backend

.. raw:: html

   <div class="container">
      <div class="row">
         <div class="col-md-6">
               <div class="doc-article-list">
                  <div class="doc-article-item">
                     <a href="./program_form/overview.html" class="article-link">
                        <div>
                           <div class="doc-article-head">
                              <span class="doc-head-content">Programming Forms</span>
                           </div>
                           <div class="doc-article-desc">
                              Provide dynamic diagrams, static diagrams, dynamic and static unified programming form, so that developers can take into account the development efficiency and execution performance.
                           </div>
                        </div>
                     </a>
                  </div>
               </div>
         </div>
         <div class="col-md-6">
               <div class="doc-article-list">
                  <div class="doc-article-item">
                     <a href="./dataset/overview.html" class="article-link">
                        <div>
                           <div class="doc-article-head">
                              <span class="doc-head-content">Data Loading and Processing</span>
                           </div>
                           <div class="doc-article-desc">
                              Two data processing models, Data Processing Pipeline and Data Processing Lightweight.
                           </div>
                        </div>
                     </a>
                  </div>
               </div>
         </div>
      </div>
   </div>

.. raw:: html

   <div class="container">
      <div class="row">
         <div class="col-md-6">
            <div class="doc-article-list">
               <div class="doc-article-item">
                  <a href="./parallel/data_parallel.html" class="article-link">
                     <div>
                        <div class="doc-article-head">
                           <span class="doc-head-content">Distributed Parallelism</span>
                        </div>
                        <div class="doc-article-desc">
                           Distributed parallelism reduces the need for hardware such as memory and computational performance, providing powerful computational capabilities and performance advantages for processing large-scale data and complex models.
                        </div>
                     </div>
                  </a>
               </div>
            </div>
         </div>
         <div class="col-md-6">
               <div class="doc-article-list">
                  <div class="doc-article-item">
                     <a href="./compile/graph_construction.html" class="article-link">
                        <div>
                           <div class="doc-article-head">
                              <span class="doc-head-content">Compile</span>
                           </div>
                           <div class="doc-article-desc">
                              Multiple compilation optimization techniques are included.
                           </div>
                        </div>
                     </a>
                  </div>
               </div>
         </div>
      </div>
   </div>
   <div class="container">
      <div class="row">
         <div class="col-md-6">
            <div class="doc-article-list">
               <div class="doc-article-item">
                  <a href="./runtime/memory_manager.html" class="article-link">
                     <div>
                        <div class="doc-article-head">
                           <span class="doc-head-content">Runtime</span>
                        </div>
                        <div class="doc-article-desc">
                           For the efficient execution of the model, providing memory management, multi-level streaming, multi-stream concurrency, and multi-backend access.
                        </div>
                     </div>
                  </a>
               </div>
            </div>
         </div>
      </div>
   </div>