# MindArmour Module Introduction

<a href="https://gitee.com/mindspore/docs/blob/master/docs/mindarmour/docs/source_en/security_and_privacy.md" target="_blank"><img src="https://mindspore-website.obs.cn-north-4.myhuaweicloud.com/website-images/master/resource/_static/logo_source_en.png"></a>

## Overview

This document describes AI security and privacy protection. As a universal technology, AI brings huge opportunities and benefits, but also faces new security and privacy protection challenges. MindArmour is a sub-project of MindSpore. It provides security and privacy protection capabilities for MindSpore, including technologies such as adversarial robustness, model security test, differential privacy training, and privacy breach risk assessment.

## Adversarial Robustness

### Attack

The `Attack` base class defines the interface for generating adversarial examples. Its subclasses implement various specific generation algorithms and allow security personnel to quickly and efficiently generate adversarial examples for attacking AI models to evaluate the robustness of the models.

### Defense

The `Defense` base class defines the interface for adversarial training. Its subclasses implement various adversarial training algorithms to enhance the adversarial robustness of the models.

### Detector

The `Detector` base class defines the interface for adversarial sample detection. Its subclasses implement various specific detection algorithms to enhance the adversarial robustness of the models.

For details, see [Improving Model Security with NAD Algorithm](https://www.mindspore.cn/mindarmour/docs/en/master/improve_model_security_nad.html).

## Model Security Test

### Fuzzer

The `Fuzzer` class controls the fuzzing process based on the neuron coverage gain. It uses natural perturbation and adversarial sample generation methods as the mutation policy to activate more neurons to explore different types of model output results and error behavior, helping users enhance model robustness.

For details, see [Testing Model Security Using Fuzz Testing](https://www.mindspore.cn/mindarmour/docs/en/master/test_model_security_fuzzing.html).

## Differential Privacy Training

### DPModel

`DPModel` inherits `mindspore.Model` and provides the entry function for differential privacy training.

For details, see [Protecting User Privacy with Differential Privacy Mechanism](https://www.mindspore.cn/mindarmour/docs/en/master/protect_user_privacy_with_differential_privacy.html).

## Suppress Privacy Training

### SuppressModel

`SuppressModel` inherits `mindspore.Model` and provides the entry function for suppress privacy training.

For details, see [Protecting User Privacy with Suppress Privacy Mechanism](https://www.mindspore.cn/mindarmour/docs/zh-CN/master/protect_user_privacy_with_suppress_privacy.html).

## Privacy Breach Risk Assessment

### Membership Inference

The `MembershipInference` class provides a reverse analysis method. It can infer whether a sample is in the training set of a model based on the prediction information of the model on the sample to evaluate the privacy breach risk of the model.

For details, see [Testing Model Security with Membership Inference](https://www.mindspore.cn/mindarmour/docs/zh-CN/master/test_model_security_membership_inference.html).
