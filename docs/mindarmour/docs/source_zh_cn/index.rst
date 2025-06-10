MindSpore Armour 文档
=========================

AI作为一种通用技术，在带来巨大机遇和效益的同时也面临着新的安全与隐私保护的挑战。MindSpore Armour是一款安全与隐私保护工具，致力于增强模型的安全可信、保护训推数据隐私安全。

.. raw:: html

   <img src="images/roadmap.png" width="700px" alt="" >

MindSpore Armour当前版本（v2.0）通过对抗鲁棒性、模型安全测试、差分隐私训练、隐私泄露风险评估、数据漂移检测等技术，实现对MindSpore的安全与隐私保护。

.. raw:: html

   <img src="https://mindspore-website.obs.cn-north-4.myhuaweicloud.com/website-images/master/docs/mindarmour/docs/source_zh_cn/images/mindarmour_cn.png" width="700px" alt="" >

代码仓地址：<https://gitee.com/mindspore/mindarmour>

使用MindSpore Armour的典型场景
------------------------------

1. `对抗样本 <https://www.mindspore.cn/mindarmour/docs/zh-CN/master/improve_model_security_nad.html>`_

   涵盖黑白盒对抗攻击、对抗训练、对抗样本检测等能力，帮助安全工作人员快速高效地生成对抗样本，评测AI模型的鲁棒性。

2. `隐私泄漏风险评估 <https://www.mindspore.cn/mindarmour/docs/zh-CN/master/test_model_security_membership_inference.html>`_

   通过成员推理攻击、模型逆向攻击等算法，用于评估模型隐私泄漏的风险。

3. `隐私保护 <https://www.mindspore.cn/mindarmour/docs/zh-CN/master/protect_user_privacy_with_differential_privacy.html>`_

   通过差分隐私训练、抑制隐私保护机制，减少模型隐私泄漏的风险，从而保护用户数据。

4. `可靠性 <https://www.mindspore.cn/mindarmour/docs/zh-CN/master/concept_drift_time_series.html>`_

   通过多种数据漂移检测算法，及时发现数据分布变化，提前预测模型失效征兆，对AI模型的及时调整具有重要意义。

5. `Fuzz <https://www.mindspore.cn/mindarmour/docs/zh-CN/master/test_model_security_fuzzing.html>`_

   基于覆盖率的Fuzzing测试流程，灵活可定制的测试策略和指标；通过神经元覆盖率来指导输入变异，让输入能够激活更多的神经元，神经元值的分布范围更广，从而探索不同类型的模型输出结果、错误行为。

6. `模型加密 <https://www.mindspore.cn/mindarmour/docs/zh-CN/master/model_encrypt_protection.html>`_

   通过加密对模型文件进行保护的功能，使用对称加密算法对参数文件或推理模型进行加密，使用时直接加载密文模型完成推理或增量训练。

.. toctree::
   :glob:
   :maxdepth: 1
   :caption: 安装部署

   mindarmour_install

.. toctree::
   :glob:
   :maxdepth: 1
   :caption: AI安全

   improve_model_security_nad
   test_model_security_fuzzing
   evaluation_of_CNNCTC
   model_encrypt_protection
   dynamic_obfuscation_protection

.. toctree::
   :glob:
   :maxdepth: 1
   :caption: AI隐私

   protect_user_privacy_with_differential_privacy
   protect_user_privacy_with_suppress_privacy
   test_model_security_membership_inference

.. toctree::
   :glob:
   :maxdepth: 1
   :caption: AI可靠性

   concept_drift_time_series
   concept_drift_images
   fault_injection

.. toctree::
   :maxdepth: 1
   :caption: API参考

   mindarmour
   mindarmour.adv_robustness.attacks
   mindarmour.adv_robustness.defenses
   mindarmour.adv_robustness.detectors
   mindarmour.adv_robustness.evaluations
   mindarmour.fuzz_testing
   mindarmour.natural_robustness.transform.image
   mindarmour.privacy.diff_privacy
   mindarmour.privacy.evaluation
   mindarmour.privacy.sup_privacy
   mindarmour.reliability
   mindarmour.utils

.. toctree::
   :glob:
   :maxdepth: 1
   :caption: 参考文档

   design
   differential_privacy_design
   fuzzer_design
   security_and_privacy
   faq

.. toctree::
   :glob:
   :maxdepth: 1
   :caption: RELEASE NOTES

   RELEASE