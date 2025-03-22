模型构建与训练
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

   parallel/startup_method
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


构建
---------

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
                              提供动态图、静态图、动静统一的编程形态，使开发者可以兼顾开发效率和执行性能。
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
                              <span class="doc-head-content">数据加载与处理</span>
                           </div>
                           <div class="doc-article-desc">
                              数据处理Pipeline和数据处理轻量化两种数据处理模式。
                           </div>
                        </div>
                     </a>
                  </div>
               </div>
         </div>
      </div>
   </div>

训练
---------

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
      </div>
   </div>
