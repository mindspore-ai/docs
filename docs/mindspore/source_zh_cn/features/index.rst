特性介绍
=========================

.. toctree::
   :glob:
   :maxdepth: 1
   :hidden:
   :caption: 编程形态

   program_form/overview

.. toctree::
   :glob:
   :maxdepth: 1
   :hidden:
   :caption: 数据处理

   dataset/overview

.. toctree::
   :glob:
   :maxdepth: 1
   :hidden:
   :caption: 分布式并行

   parallel/data_parallel
   parallel/operator_parallel
   parallel/optimizer_parallel
   parallel/pipeline_parallel
   parallel/auto_parallel

.. toctree::
   :glob:
   :maxdepth: 1
   :hidden:
   :caption: 编译

   compile/graph_construction
   compile/graph_optimization

.. toctree::
   :glob:
   :maxdepth: 1
   :hidden:
   :caption: 运行时

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
                              <span class="doc-head-content">编程形态</span>
                           </div>
                           <div class="doc-article-desc">
                              提供动静统一的编程形态，兼顾开发效率和执行性能。
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
                              <span class="doc-head-content">数据处理</span>
                           </div>
                           <div class="doc-article-desc">
                              提供高性能数据处理引擎。
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
                           <span class="doc-head-content">分布式并行</span>
                        </div>
                        <div class="doc-article-desc">
                           通过分布式并行，降低对内存、计算性能等硬件的需求，为处理大规模数据和复杂模型提供了强大的计算能力和性能优势。
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
                              <span class="doc-head-content">编译</span>
                           </div>
                           <div class="doc-article-desc">
                              介绍编译构图和图优化特性，包括代数化简和冗余消除等。
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
                           <span class="doc-head-content">运行时</span>
                        </div>
                        <div class="doc-article-desc">
                           负责模型的高效执行，提供内存管理、多级流水、多流并发、多后端接入等功能。
                        </div>
                     </div>
                  </a>
               </div>
            </div>
         </div>
      </div>
   </div>
