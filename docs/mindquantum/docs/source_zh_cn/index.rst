MindSpore Quantum文档
=========================

.. toctree::
   :glob:
   :maxdepth: 1
   :caption: 初级使用教程
   :hidden:

   beginner/beginner
   beginner/parameterized_quantum_circuit
   beginner/quantum_simulator
   beginner/quantum_measurement
   beginner/advanced_operations_of_quantum_circuit
   beginner/bloch_sphere

.. toctree::
   :glob:
   :maxdepth: 1
   :caption: 中级使用教程
   :hidden:

   middle_level/middle_level
   middle_level/noise
   middle_level/noise_simulator
   middle_level/qubit_mapping

.. toctree::
   :glob:
   :maxdepth: 1
   :caption: 高级使用教程
   :hidden:

   advanced/advanced
   advanced/get_gradient_of_PQC_with_mindquantum
   advanced/initial_experience_of_quantum_neural_network
   advanced/equivalence_checking_of_PQC

.. toctree::
   :glob:
   :maxdepth: 1
   :caption: 案例库
   :hidden:

   case_library/case_library
   case_library/grover_search_algorithm
   case_library/shor_algorithm
   case_library/hhl_algorithm
   case_library/quantum_phase_estimation
   case_library/quantum_approximate_optimization_algorithm
   case_library/classification_of_iris_by_qnn
   case_library/vqe_for_quantum_chemistry
   case_library/qnn_for_nlp
   case_library/quantum_annealing_inspired_algorithm
   case_library/qaia_automatic_parameter_adjustment
   case_library/qaia_gpu_tutorial

.. toctree::
   :glob:
   :maxdepth: 1
   :caption: API
   :hidden:

   overview
   mindquantum.dtype
   mindquantum.core
   mindquantum.simulator
   mindquantum.framework
   mindquantum.algorithm
   mindquantum.device
   mindquantum.io
   mindquantum.engine
   mindquantum.utils

.. toctree::
   :glob:
   :maxdepth: 1
   :caption: 论文复现代码
   :hidden:

   paper_with_code

.. toctree::
   :glob:
   :maxdepth: 1
   :caption: RELEASE NOTES
   :hidden:

   RELEASE

.. toctree::
   :glob:
   :maxdepth: 1
   :caption: 安装指南
   :hidden:

   mindquantum_install

.. raw:: html

   <div class="container">
			<div class="row">
            <div class="col-md-6">
               <div class="doc-article-list">
                  <div class="doc-article-item">
                     <a href="./mindquantum_install.html" class="article-link">
                        <div>
                           <div class="doc-article-head">
                              <span class="doc-head-content">安装指南</span>
                           </div>
                           <div class="doc-article-desc">
                              了解如何在不同系统中安装MindSpore Quantum，或者以开发者身份本地化快速编译并调试MindSpore Quantum。
                           </div>
                        </div>
                     </a>
                  </div>
					</div>
				</div>
				<div class="col-md-6">
               <div class="doc-article-list">
                  <div class="doc-article-item">
                     <a href="./beginner/beginner.html" class="article-link">
                        <div>
                           <div class="doc-article-head">
                              <span class="doc-head-content">初级使用教程</span>
                           </div>
                           <div class="doc-article-desc">
                              了解 MindSpore Quantum 的基本组成元素，包括<b style="color: #3366FF">量子门</b>、<b style="color: #3366FF">量子线路</b>、<b style="color: #3366FF">哈密顿量</b>和<b style="color: #3366FF">量子模拟器</b>的生成与使用。
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
                     <a href="./middle_level/middle_level.html" class="article-link">
                        <div>
                           <div class="doc-article-head">
                              <span class="doc-head-content">中级使用教程</span>
                           </div>
                           <div class="doc-article-desc">
                              了解 MindSpore Quantum 在<b style="color: #3366FF">含噪声量子模拟</b>、<b style="color: #3366FF">量子线路编译</b>、<b style="color: #3366FF">比特映射</b>等更贴近真实量子芯片场景的应用。
                           </div>
                        </div>
                     </a>
                  </div>
					</div>
				</div>
            <div class="col-md-6">
               <div class="doc-article-list">
                  <div class="doc-article-item">
                     <a href="./advanced/advanced.html" class="article-link">
                        <div>
                           <div class="doc-article-head">
                              <span class="doc-head-content">高级使用教程</span>
                           </div>
                           <div class="doc-article-desc">
                              了解 MindSpore Quantum 针对 NISQ 算法的设计与使用，特别是如何设计<b style="color: #3366FF">变分量子算法</b>以及与 MindSpore 协同完成<b style="color: #3366FF">量子-经典混合算法</b>的训练。
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
                     <a href="./case_library/case_library.html" class="article-link">
                        <div>
                           <div class="doc-article-head">
                              <span class="doc-head-content">案例库</span>
                           </div>
                           <div class="doc-article-desc">
                              介绍 MindSpore Quantum 在<b style="color: #3366FF">通用量子算法</b>与<b style="color: #3366FF">变分量子算法</b>领域的完整案例教程，快速入门相关研究领域。
                           </div>
                        </div>
                     </a>
                  </div>
					</div>
				</div>
            <div class="col-md-6">
               <div class="doc-article-list">
                  <div class="doc-article-item">
                     <a href="./overview.html" class="article-link">
                        <div>
                           <div class="doc-article-head">
                              <span class="doc-head-content">API</span>
                           </div>
                           <div class="doc-article-desc">
                              MindSpore Quantum API说明列表。
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
                     <a href="./paper_with_code.html" class="article-link">
                        <div>
                           <div class="doc-article-head">
                              <span class="doc-head-content">论文复现代码</span>
                           </div>
                           <div class="doc-article-desc">
                              开源贡献者以及官方基于学术论文的复现代码。
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
                              包含发布版本的主要特性和增强，API变更等信息。
                           </div>
                        </div>
                     </a>
                  </div>
					</div>
				</div>
		   </div>
	</div>
