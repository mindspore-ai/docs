MindArmour Documents
=========================

As a general technology, AI brings great opportunities and benefits, but also faces new security and privacy protection challenges. MindArmour is a subsystem of MindSpore. It provides security and privacy protection for MindSpore, including adversarial robustness, model security test, differential privacy training, privacy risk assessment, and data drift detection.

.. raw:: html

   <img src="https://gitee.com/mindspore/docs/raw/master/docs/mindarmour/docs/source_en/images/mindarmour.png" width="700px" alt="" >

Typical MindArmour Application Scenarios
-----------------------------------------

1. `Adversarial Example <https://www.mindspore.cn/mindarmour/docs/en/master/improve_model_security_nad.html>`_

   Cover capabilities such as black-and-white box adversarial attacks, adversarial training, and adversarial example detection, helping security personnel quickly and efficiently generate adversarial examples and evaluate the robustness of AI models.

2. `Privacy Risk Assessment <https://www.mindspore.cn/mindarmour/docs/en/master/test_model_security_membership_inference.html>`_

   Use algorithms such as membership inference attack and model inversion attack to evaluate the risk of model privacy leakage.

3. `Privacy Protection <https://www.mindspore.cn/mindarmour/docs/en/master/protect_user_privacy_with_differential_privacy.html>`_

   Use differential privacy training and privacy protection suppression mechanisms to reduce the risk of model privacy leakage and protect user data.

4. `Fuzz <https://www.mindspore.cn/mindarmour/docs/en/master/test_model_security_fuzzing.html>`_

   Perform the fuzzing test based on coverage, provide flexible and customizable test policies and indicators. Use the neuron coverage rate to guide input mutation so that the input can activate more neurons and the neuron value distribution range is wider. In this way, different types of model output results and incorrect behaviors can be explored.

5. `Model Encryption <https://www.mindspore.cn/mindarmour/docs/en/master/model_encrypt_protection.html>`_

   Use the symmetric encryption algorithm to encrypt the parameter files or inference models to protect the model files. When the symmetric encryption algorithm is used, the ciphertext model is directly loaded to complete inference or incremental training.

.. toctree::
   :glob:
   :maxdepth: 1
   :caption: Installation

   mindarmour_install

.. toctree::
   :glob:
   :maxdepth: 1
   :caption: AI Security

   improve_model_security_nad
   test_model_security_fuzzing
   test_model_security_membership_inference

.. toctree::
   :glob:
   :maxdepth: 1
   :caption: AI Privacy

   protect_user_privacy_with_differential_privacy
   protect_user_privacy_with_suppress_privacy
   model_encrypt_protection

.. toctree::
   :glob:
   :maxdepth: 1
   :caption: AI Reliability

   concept_drift_time_series
   fault_injection

.. toctree::
   :maxdepth: 1
   :caption: API References

   mindarmour
   mindarmour.adv_robustness.attacks
   mindarmour.adv_robustness.defenses
   mindarmour.adv_robustness.detectors
   mindarmour.adv_robustness.evaluations
   mindarmour.fuzz_testing
   mindarmour.privacy.diff_privacy
   mindarmour.privacy.evaluation
   mindarmour.privacy.sup_privacy
   mindarmour.reliability
   mindarmour.utils

.. toctree::
   :glob:
   :maxdepth: 1
   :caption: References

   differential_privacy_design
   fuzzer_design
   security_and_privacy
   faq
