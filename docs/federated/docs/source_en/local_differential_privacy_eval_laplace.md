# Horizontal Federated-Local Differential Privacy Inference Result Protection

[![View Source On Gitee](https://mindspore-website.obs.cn-north-4.myhuaweicloud.com/website-images/master/resource/_static/logo_source_en.svg)](https://gitee.com/mindspore/docs/blob/master/docs/federated/docs/source_en/local_differential_privacy_eval_laplace.md)

## Privacy Protection Background

Evaluating the federated unsupervised model training can be judged by the $loss$ of end-side feedback, or the end-side inference results combined with cloud-side clustering and clustering evaluation metrics can be used to further monitor the federated unsupervised model training progress. The latter involves end-side inference data on the cloud, and in order to meet privacy protection requirements, privacy protection processing of end-side inference data is required, while the cloud side can still be evaluated for clustering. This task is a secondary task compared to the training task, so we use lightweight algorithms and cannot introduce privacy protection algorithms with higher computational or communication overhead than the training phase. This paper presents a lightweight scheme for protecting inference results by using the local differential privacy Laplace noise mechanism.

The effective integration of privacy protection technology into the product services will, on the one hand, help enhance the trust of users and the industry in the products and technology, and on the other hand, help to better carry out the federated tasks under the current privacy compliance requirements and create a full lifecycle (training-inference-evaluation) of privacy protection.

## Algorithm Analysis

### $L1$ and $L2$ Paradigm

The $L1$-norm of a vector $V$ with length $k$ is $||V|||_1=\sum^{k}_{i=1}{|V_i|}$, then the $L1$-norm of the difference between two vectors in two-dimensional space is the Manhattan distance.

$L2$-norm is $||V||_2=\sqrt{\sum^{k}_{i=1}{V^2_i}}$.

The inference result is generally a $softmax$ result with a sum of $1$, and each dimension value of the vector indicates the probability of belonging to the corresponding category of that dimension.

### $L1$ and $L2$ Sensitivity

Local differential privacy introduces uncertainty on the data to be uploaded, and the sensitivity describes an upper bound on the uncertainty. Gaussian noise with $L2$ sensitivity can be added to the gradient in the optimizer and federated training, since a cropping operation is performed on the gradient vector before addition. Here the $softmax$ inference result satisfies the sum as $1$, so the Laplace noise of $L1$ is added. For applications where the $L2$ sensitivity is much lower than the $L1$ sensitivity, the Gaussian mechanism allows to add less noise, but the scenario has no $L2$-related constraint limits and uses only the $L1$ sensitivity.

The $L1$-sensitivity is expressed as the maximum distance for any input in the defined domain in local differential privacy:

$\Delta f=max||X-Y||_1$

In this scenario, $X=<x_1, x_2, ..., x_k>, Y=<y_1, y_2, ..., y_k>, \sum X = 1, \sum Y = 1, |x_1-y_1|+|x_2-y_2|+...+|x_k-y_k|\leq1=\Delta f$.

### Laplace Distribution

The Laplace distribution is continuous, and the probability density function of the Laplace with mean value 0 is:

$Lap(x|b)=\frac{1}{2b}exp(-\frac{|x|}{b})$

### Laplace Mechanism

$M(x,\epsilon)=X+Lap(\Delta f/\epsilon)$

where $Lap(\Delta f/\epsilon)$ is a vector of random variables with the same shape as $X$, independently and identically distributed.

In this scenario, $b$ (also called $scale$, $lambda$, $beta$) is $1/\epsilon$.

### Proving that the Laplace Mechanism is Satisfied with the $\epsilon-LDP$

Any two different clients, after being processed by the Laplace mechanism, both output the same result to achieve the confusion indistinguishable and the purpose probability ratio of outputting the same result has upper exact bound. Substituting $b=\Delta f/\epsilon$ yields:

$Lap(\Delta f/\epsilon)=\frac{\epsilon}{2\Delta f}exp(-\frac{\epsilon|x|}{\Delta f})$

$\frac{P(Z|X)}{P(Z|Y)}$

$=\prod^k_{i=1}(\frac{exp(-\frac{\epsilon|x_i-z_i|}{\Delta f})}{exp(-\frac{\epsilon |y_i-z_i|}{\Delta f})})$

$=\prod^k_{i=1}exp(\epsilon\frac{|x_i-z_i|-|y_i-z_i|}{\Delta f})$

$\leq\prod^k_{i=1}(\epsilon\frac{|x_i-y_i|}{\Delta f})$

$=exp(\epsilon\frac{X-Y}{\Delta f})$

$\leq exp(\epsilon)$

#### The Determination of $\epsilon$ with the Corresponding Probability Density Plot

The privacy budget with high availability is calculated by combining the data characteristics, such as the requirement to output noise of the order of $1e-5$ with high probability, otherwise it will directly affect the clustering results. The privacy budget calculation method corresponding to generating the specified amount of noise is given below.

There is the $90\%$ probability to output the magnitude of $1e-5$, and the value of $\epsilon$ is obtained by integrating the probability density curve.

$x>=0, Lap(x|b)=\frac{1}{2b}exp(-\frac{x}{b})$

$\int^ {E^{-5}}_0 {Lap(x|b)dx}$

$=1-\frac{1}{2}exp(-\frac{x}{b})|^{E^{-5}}_{0}$

$=\frac{1}{2}(exp(0)-exp(-\frac{E^{-5}}{b}))$

$=0.5(1-exp(-\frac{E^{-5}}{b})) = 0.45$

i.e.

$exp(-\frac{E^{-5}}{b})=0.1$

$b=-E^{-5}/ln(0.1)=E^{-5}/2.3026=1/\epsilon$

$\epsilon=2.3026E^5$

When the privacy budget takes this value, the Laplace probability density function is as follows:

![laplace](https://mindspore-website.obs.cn-north-4.myhuaweicloud.com/website-images/master/docs/federated/docs/source_zh_cn/images/laplace_pdf.png)

### Impact Analysis of Clustering Evaluation Indicators

Using the **Calinski-Harabasz Index** assessment method as an example, the evaluation indicator is calculated in two steps:

1. Each class calculates the sum of the squares of the distances from all `points` in the class to the `center of the class`;

2. Calculate the sum of squares of distances from each `class` to the `center of the class`;

Source code implementation and impact analysis after noise addition:

```python
# 1.The cloud-side clustering algorithm gets the class ordinal number to which it belongs, with impact
n_labels = argmax(X)

extra_disp, intra_disp = 0.0, 0.0
# 2.Calculate the class center of all points, without impact
mean = np.mean(X, axis=0)
for k in range(n_labels):
    # 3.Get all points in class k, based on the effect of 1
    cluster_k = X[labels == k]
    # 4.Get the class center, based on the impact of 1
    mean_k = np.mean(cluster_k, axis=0)
    # 5.The distance between the class and the center of all classes, based on the impact of 1
    extra_disp += len(cluster_k) * np.sum((mean_k - mean) ** 2)
    # 6.The distance from the point to the center of the class, with impact
    intra_disp += np.sum((cluster_k - mean_k) ** 2)

return (
    1.0
    if intra_disp == 0.0
    else extra_disp * (n_samples - n_labels) / (intra_disp * (n_labels - 1.0))
)
```

In a comprehensive analysis, the main impact is on the clustering algorithm after noise addition, and the error on the distance calculation. When calculating the class center, the error introduced is small because the noise sum is  expected to be $0$.

Taking **SILHOUETTE SCORE** as an example, the process of calculating this evaluation indicator is divided into two steps:

1. Calculate the average distance of a sample point $i$ from all other sample points in the same cluster, which is denoted as $a_i$. The smaller the value is, the more the sample $i$ should be assigned to this cluster.

2. Calculate the average distance $b_{ij}$ of sample $i$ to all samples of some other cluster $C_j$, which is called the dissimilarity of sample $i$ to cluster $C_j$. The inter-cluster dissimilarity of sample $i$ is defined as: $b_i = min(b_{i1}, b_{i2}, ..., b_{ik})$. The larger the value is, the less the sample $i$ should belong to this cluster.

![flow](https://mindspore-website.obs.cn-north-4.myhuaweicloud.com/website-images/master/docs/federated/docs/source_zh_cn/images/two_cluster.png)

$s_i=(b_i-a_i) / max(a_i, b_i)$.

The smaller $a_i$ is, the larger $b_i$ is, and the result is $1-a_i / b_i$. The closer to $1$, the better the clustering effect.

Pseudocode implementation and impact analysis after noise addition:

```c++
// Calculate distance matrix, space for time, upper triangle storage, which has an impact on noise addition
euclidean_distance_matrix(&distance_matrix, group_ids);

// Perform the same calculation for each point, and finally calculate the mean value
for (size_t i = 0; i < n_samples; ++i) {
    std::unordered_map<size_t, std::vector<float>> b_i_map;
    for (size_t j = 0; j < n_samples; ++j) {
        size_t label_j = labels[j];
        float distance = distance_matrix[i][j];
        // Same cluster calculates ai
        if (label_j == label_i) {
        a_distances.push_back(distance);
        } else {
            // Different clusters calculate bi
            b_i_map[label_j].push_back(distance);
        }
    }
    if (a_distances.size() > 0) {
        // Calculate the average distance of the point from other points in the same cluster
        a_i = std::accumulate(a_distances.begin(), a_distances.end(), 0.0) / a_distances.size();
    }
    for (auto &item : b_i_map) {
        auto &b_i_distances = item.second;
        float b_i_distance = std::accumulate(b_i_distances.begin(), b_i_distances.end(), 0.0) / b_i_distances.size();
        b_i = std::min(b_i, b_i_distance);
    }
    if (a_i == 0) {
        s_i[i] = 0;
    } else {
        s_i[i] = (b_i - a_i) / std::max(a_i, b_i);
    }
}
return std::accumulate(s_i.begin(), s_i.end(), 0.0) / n_samples;
```

As above, the main impact is the main impact is on the clustering algorithm after noise addition, and the error on the distance calculation.

### End-side Java Implementation

There is no function in the Java basic library to generate Laplace distributed random numbers. The following combination strategy of random numbers is used to generate.

The source code is as follows:

```java
float genLaplaceNoise(SecureRandom secureRandom, float beta) {
    float u1 = secureRandom.nextFloat();
    float u2 = secureRandom.nextFloat();
    if (u1 <= 0.5f) {
        return (float) (-beta * log(1. - u2));
    } else {
        return (float) (beta * log(u2));
    }
}
```

After obtaining a new round of model on the end-side, the inference calculation is executed immediately. After the training, the inference results after privacy protection are uploaded to the cloud side together with the new model, and the cloud side finally performs operations such as clustering and score calculation. The flow is shown in the following figure, where the red part is the output result of privacy protection processing:

![flow](https://mindspore-website.obs.cn-north-4.myhuaweicloud.com/website-images/master/docs/federated/docs/source_zh_cn/images/eval_flow.png)

## Quick Start

### Preparation

To use this feature, one first needs to successfully complete the training aggregation process for either end-cloud federated scenario. [Implementing an Image Classification Application of Cross-device Federated Learning (x86)](https://www.mindspore.cn/federated/docs/en/master/image_classification_application.html) details the preparation of datasets and network models, as well as simulates the process of initiating multi-client participation in federated learning.

### Configuration Items

The [cloud-side yaml configuration file](https://gitee.com/mindspore/federated/blob/master/tests/st/cross_device_cloud/default_yaml_config.yaml) gives the complete configuration items for opening the end-cloud federated, and the program involves the following additional configuration file items:

```c
encrypt:
    privacy_eval_type: LAPLACE
    laplace_eval:
        laplace_eval_eps: 230260
```

where `privacy_eval_type` currently supports only `NOT_ENCRYPT` and `LAPLACE`, indicating that the inference results are processed without privacy protection methods and with the `LAPLACE` mechanism, respectively.

`laplace_eval_eps` indicates how much of the privacy budget is used if `LAPLACE` processing is used.

## Experimental Results

The basic configuration associated with the inference result evaluation function is used as follows:

```c
unsupervised:
  cluster_client_num: 1000
  eval_type: SILHOUETTE_SCORE
```

We can see that the relationship between $loss$ and the score under the `LAPLACE` mechanism by using `NOT_ENCRYPT` and using `laplace_eval_eps=230260` is shown in the figure:

![flow](https://mindspore-website.obs.cn-north-4.myhuaweicloud.com/website-images/master/docs/federated/docs/source_zh_cn/images/SILHOUETTE.png)

The red dashed line shows the SILHOUETTE scores after the Laplace mechanism is used to protect the inference results. Since the model contains $dropout$ and Gaussian input, the $loss$ of the two trainings are slightly different and the scores obtained based on different models are slightly different. However, the overall trend remains consistent and can assist $loss$ together to detect the model training progress.
