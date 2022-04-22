# Local differential privacy perturbation training

<a href="https://gitee.com/mindspore/docs/blob/master/docs/federated/docs/source_en/local_differential_privacy_training_noise.md" target="_blank"><img src="https://mindspore-website.obs.cn-north-4.myhuaweicloud.com/website-images/master/resource/_static/logo_source_en.png"></a>

During federated learning, user data is used only for local device training and does not need to be uploaded to the central server. This prevents personal data leakage.
However, in the conventional federated learning framework, models are migrated to the cloud in plaintext. There is still a risk of indirect disclosure of user privacy.
After obtaining the plaintext model uploaded by a user, the attacker can restore the user's personal training data through attacks such as reconstruction and model inversion. As a result, user privacy is disclosed.

As a federated learning framework, MindSpore Federated provides secure aggregation algorithms based on local differential privacy (LDP).
Noise addition is performed on local models before they are migrated to the cloud. On the premise of ensuring the model availability, the problem of privacy leakage in horizontal federated learning is solved.

## Principles

Differential privacy is a mechanism for protecting user data privacy. **Differential privacy** is defined as follows:

$$
Pr[\mathcal{K}(D)\in S] \le e^{\epsilon} Pr[\mathcal{K}(D') \in S]+\delta​
$$

For datasets $D and D'$ that have only one record difference, the random algorithm $\mathcal{K}$ is used to compute the probability of the $S$ subset, which meets the preceding formula. $\epsilon$ is the differential privacy budget, and $\delta$ is the perturbation. The smaller the values of $\epsilon$ and $\delta$, the closer the data distribution of $\mathcal{K}$ on $D$ and $D'$.

In horizontal federated learning, if the model weight matrix after local training on the client is $W$, the adversary can use $W$ to restore the training dataset[1] of the user because the model "remembers" the features of the training set during the training process.

MindSpore Federated provides a LDP-based secure aggregation algorithm to prevent privacy data leakage when local models are migrated to the cloud.

The MindSpore Federated client generates a differential noise matrix $G$ that has the same dimension as the local model $W$, and then adds the two to obtain a weight $W_p$ that meets the differential privacy definition:

$$
W_p=W+G
$$

The MindSpore Federated client uploads the noise-added model $W_p$ to the cloud server for federated aggregation. The noise matrix $G$ is equivalent to adding a layer of mask to the original model, which reduces the risk of sensitive data leakage from models and affects the convergence of model training. How to achieve a better balance between model privacy and usability is still a question worth studying. Experiments show that when the number of participants $n$ is large enough (generally more than 1000), most of the noises can cancel each other, and the LDP mechanism has no obvious impact on the accuracy and convergence of the aggregation model.

## Usage

Local differential privacy training currently only supports cross device scenarios. Enabling differential privacy training is simple. You only need to perform the following operation during the cloud service startup.
Use `context.set_fl_context()` to set `encrypt_type='DP_ENCRYPT'`.

In addition, to control the effect of privacy protection, three parameters are provided: `dp_eps`, `dp_delta`, and `dp_norm_clip`.
They are also set through `context.set_fl_context()`. The valid value range of `dp_eps` and `dp_norm_clip` is greater than 0.

The value of `dp_delta` ranges between 0 and 1. Generally, the smaller the values of `dp_eps` and `dp_delta`, the better the privacy protection effect.
However, the impact on model convergence is greater. It is recommended that `dp_delta` be set to the reciprocal of the number of clients and the value of `dp_eps` be greater than 50.

`dp_norm_clip` is the adjustment coefficient of the model weight before noise is added to the model weight by the LDP mechanism. It affects the convergence of the model. The recommended value ranges from 0.5 to 2.

## References

[1] Ligeng Zhu, Zhijian Liu, and Song Han. [Deep Leakage from Gradients](http://arxiv.org/pdf/1906.08935.pdf). NeurIPS, 2019.
