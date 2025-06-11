MindSpore Armour Documents
============================

AI is the catalyst for change but also faces challengs in security and privacy protection. MindSpore Armour is a security and privacy protection tool that is committed to enhancing the trustworthiness of models and protecting the privacy of training and inference data.

.. raw:: html

   <img src="images/roadmap.png" width="700px" alt="" >

The current version of MindSpore Armour (v2.0) achieves security and privacy protection for MindSpore through technologies such as adversarial robustness, model security testing, differential privacy training, privacy leakage risk assessment, and data drift detection.

.. raw:: html

   <img src="https://mindspore-website.obs.cn-north-4.myhuaweicloud.com/website-images/master/docs/mindarmour/docs/source_en/images/mindarmour.png" width="700px" alt="" >

Code repository address: <https://gitee.com/mindspore/mindarmour>

Typical Application Scenarios
-----------------------------

1. `Adversarial Example <https://www.mindspore.cn/mindarmour/docs/en/master/improve_model_security_nad.html>`_

   Includes capabilities such as white and black box adversarial attacks, adversarial training, and adversarial example detection, to help personnel generate adversarial examples and evaluate the robustness of AI models.

2. `Privacy Risk Assessment <https://www.mindspore.cn/mindarmour/docs/en/master/test_model_security_membership_inference.html>`_

   Uses algorithms such as membership inference attack and model inversion attack to assess the privacy risk for models.

3. `Privacy Protection <https://www.mindspore.cn/mindarmour/docs/en/master/protect_user_privacy_with_differential_privacy.html>`_

   Emhances model privacy and protects user data using differential training and protection suppression mechanisms.

4. `Reliability <https://www.mindspore.cn/mindarmour/docs/en/master/concept_drift_time_series.html>`_

   Detects data distribution changes in time and predicts the symptoms of model failure in advance, which is of great significance for the timely adjustment of the AI model through multiple data drift detection algorithms.

5. `Fuzz <https://www.mindspore.cn/mindarmour/docs/en/master/test_model_security_fuzzing.html>`_

   Provides a coverage-guided fuzzing tool that features flexible, customizable test policies and metrics, and uses neuron coverage to guide input mutation so that the input can activate neurons and distribute neuron values in a wider range. In this way, we can discover different types of model output results and incorrect behaviors.

6. `Model Encryption <https://www.mindspore.cn/mindarmour/docs/en/master/model_encrypt_protection.html>`_

   Uses the symmetric encryption algorithm to encrypt the parameter files or inference models to protect the model files. Directly loads the ciphertext model to implement inference or incremental training when using the algorithm.

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
   evaluation_of_CNNCTC
   dynamic_obfuscation_protection
   model_encrypt_protection

.. toctree::
   :glob:
   :maxdepth: 1
   :caption: AI Privacy

   protect_user_privacy_with_differential_privacy
   protect_user_privacy_with_suppress_privacy
   test_model_security_membership_inference

.. toctree::
   :glob:
   :maxdepth: 1
   :caption: AI Reliability

   concept_drift_time_series
   concept_drift_images
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
   mindarmour.natural_robustness.transform.image
   mindarmour.privacy.diff_privacy
   mindarmour.privacy.evaluation
   mindarmour.privacy.sup_privacy
   mindarmour.reliability
   mindarmour.utils

.. toctree::
   :glob:
   :maxdepth: 1
   :caption: References

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