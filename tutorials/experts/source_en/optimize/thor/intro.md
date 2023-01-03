# Introduction to Second-order Optimizer THOR

<a href="https://gitee.com/mindspore/docs/blob/r2.0.0-alpha/tutorials/experts/source_en/optimize/thor/intro.md" target="_blank"><img src="https://mindspore-website.obs.cn-north-4.myhuaweicloud.com/website-images/master/resource/_static/logo_source_en.png"></a>

The deep learning training process can be viewed as a loss function loss value decreasing process, and the right optimizer can make deep learning training time significantly reduced. Optimizers can be divided into first-order optimizers and second-order optimizers. The industry is still the mainstream use of the first-order optimizers, while the second-order optimizers is widely used because the single-step training time is too long. In recent years, there have been theoretical breakthroughs in applying the second-order optimization to deep learning training, and good results have been achieved.

This article will introduce the background of the optimizers, and second-order optimizer THOR self-developed by the MindSpore team.

## Introduction to Background of the Optimizers

Suppose the training sample data set: $D = {(x_1,y_1),... ,(x_i,y_i),... ,(x_N,y_N)},x_i \in X,y_i\in Y$, the deep neural network model with parameter θ formulation is: $\hat{y} = f(x;\theta),x\in{X}$, the loss function defined between the model output and the true label y is: $L(y,\hat y),y \in Y$, the process of network parameter learning is the minimization the loss function: $\min\limits_{\theta}L(y,\hat{y})$. Given the dataset, model, and loss function, the deep learning training problem boils down to the optimization problem. The deep neural network training optimization problem has a huge parameter scale and requires a large amount of computation, making it difficult to compute an analytic solution. Therefore, the process is often compared to descending a mountain. As shown in Figure 1, how can a person find the fastest path down a mountain with limited sight distance while standing at the top?

![The process of deeplearning training](https://gitee.com/mindspore/docs/raw/r2.0.0-alpha/tutorials/experts/source_zh_cn/optimize/thor/images/deeplearning_train_process.png)

*Figure 1 Simulation of Deep Learning Training Process*

The optimizer is doing this, and the optimization algorithms in the industry can be divided into first-order optimization algorithms and second-order optimization algorithms. The following is a brief description of optimizers in the industry.

### First-order Optimizers

Gradient Descent (GD) is the most classic first-order optimization algorithm in machine learning and the most commonly used optimization algorithm among many machine learning algorithms. The following rule is used for updating parameters in common first-order optimization algorithms (e.g., SGD algorithm): $\theta = \theta - \eta \nabla L_\theta$, where $\theta$ is the parameter to be updated, $\eta$ is the learning rate, and $\nabla L_\theta$ is the gradient of the loss function with respect to the parameter.

But the mainstream stochastic gradient descent methods have the following problems: Too small a learning rate will cause the network to converge too slowly, too high a learning rate may affect convergence and cause the loss function to fluctuate on the minimum or even diverge, which is more sensitive to the parameters, and it is easy to converge to the local optimum and difficult to jump out of the saddle point.

Therefore, many improved algorithms for stochastic gradient descent methods have been proposed in the industry, such as Momentum, Nesterov, AdaGrad, RMSprop, Adadelta, and Adam. These improved optimization algorithms can adaptively update the step size by using the historical information of the stochastic gradient, making them easier to tune the reference and convenient to use.

### Second-order Optimizers

The second-order optimization algorithm uses the second-order derivative of the objective function for curvature correction to accelerate the first-order gradient descent. Compared with the first-order optimizer, its convergence is faster, highly approximate the optimal value. Geometrically the descent path is more consistent with the real optimal descent path.

For example, the Newton method of second-order optimization algorithms is to fit a local surface at your current location with a quadratic surface, while the gradient descent method uses a plane to fit the current local surface. Usually, the quadratic surface will be better fitted than the plane, so the descent path chosen by the Newton method will be more consistent with the true optimal descent path. As shown in Figure 2, the left descent path indicates the descent curve of Newton method, and the right indicates the descent curve of the first-order gradient. The second-order algorithm can go to the destination faster than the first-order algorithm, thus accelerating the convergence.

![The different process of deeplearning training](https://gitee.com/mindspore/docs/raw/r2.0.0-alpha/tutorials/experts/source_zh_cn/optimize/thor/images/different_train_process.png)

*Figure 2 Descent Path of Different Optimizers*

Mathematically, in contrast to the first-order optimization algorithm, the second-order optimization algorithm starts by multiplying $\nabla L_{\theta}$ with a matrix $G^{-1}$ to produce the following update rule: $\theta = \theta - \eta G^{-1}\nabla L_{\theta}$, where G is the second-order information matrix. The definition of G in different second-order optimization algorithms is different. The common second-order optimization algorithms are Newton method, natural gradient method, etc., which correspond to the second-order information matrix G as Hessian matrix, Fisher matrix, respectively.

Newton method has a very good local convergence property. When the function L satisfies $\nabla L_{\theta^{*}}=0,\nabla^{2} L_{\theta^{*}}$ is a positive definite matrix at the optimal value point $\theta^{*}$ point, and when the Hessian matrix is Lipschitz continuous near the extreme value point, the Newton method converges quadratically to the optimal value point. The Hessian matrix is a square matrix consisting of all second-order partial derivatives of a multivariate real-valued function. The Hessian matrix can be expressed as $H_{ij} = \frac{\partial^2L}{\partial \theta_i \partial \theta_j}$, where L is the loss function and $\theta$ is the parameter to be updated.

In SGD, Euclidean distance is used for both parameter space and function space metrics, but Euclidean distance cannot be used as an accurate distance metric for function space in some cases. For example, in neural networks, the change in the objective function due to the parameters is a probabilistic change, which does not fit in the Euclidean space metric, and it is not a reasonable characterization of probabilistic property changes. KL scatter is a reasonable measure of the distance between distributions. When using KL divergence as a reasonable measure of the distance between distributions. In this case, the gradient used in the parameter update is the natural gradient. The Fisher matrix in the natural gradient method can be expressed as: $F=\mathrm{E}[\frac{\partial \mathrm {log} p(y|x,\theta)}{\partial \theta}{\frac{\partial \mathrm {log} p(y|x,\theta)}{\ partial \theta}}^T]$, where P(y|x,θ) is the predictive distribution of the network model, p(y|x,θ) is its probability density, and θ is the parameter needed for the network model.

Although the second-order optimization algorithm converges quickly, the time complexity of computing the inverse of the second-order matrix is $\mathrm O(n^3)$. When the number of model parameters is $\n_\theta$, the size of the corresponding second-order information matrix is $\n_\theta \times n_\theta$. In deep learning models, $n_\theta$ is often in the order of millions, and at this time the inverse of the second-order information matrix cannot be computed. Therefore, how to reduce the computational complexity of the inverse of the second-order information matrix becomes a key issue. Next, we introduce the second-order optimizer in deep learning.

## Introduction to THOR

The second-order optimization algorithms currently available in the industry are computationally intensive and have no obvious advantages over first-order or are used in the simple scenarios. MindSpore proposes a self-developed algorithm [THOR (Trace-based Hardware-driven layer-ORiented Natural Gradient Descent Computation)](https://ojs.aaai.org/index.php/AAAI/article/view/16867) accepted by AAAI. THOR has significant gains in several scenarios, such as convergence speed in both BERT and ResNet50. THOR has made two main innovation points:

### Reducing the Frequency of Second-order Information Matrix Updates

By experimentally observing that the F-parameter (Frobenius norm) of the Fisher matrix changes drastically in the early stage and gradually becomes stable in the later stage, it is assumed that $\Big\{{F^k}\Big\}^{n}_{k=1}$ is a Markov process that converges to a steady-state distribution π, where $F^k$ represents the Fisher matrix at the kth iteration. Therefore, gradually increasing the update interval of the Fisher matrix during the training process can reduce the training time without affecting the convergence speed. For example, in ResNet50, the number of update interval steps gets larger and larger as the training proceeds, to the point where only one update of the second-order information matrix is required per epoch in the later stages.

Inspired by KFAC, THOR decouples Fisher matrices by layer to reduce matrix complexity, performs experiments for each layer of Fisher matrices separately. It can be found that some layers of Fisher matrices converge to steady state faster, so the update frequency of each layer is adjusted more fine-grained on a uniform update interval. THOR uses the trace of the matrix as a judgment condition. When the change of the trace is greater than a certain threshold, the second-order information matrix of the layer is updated, otherwise the second-order information matrix of the previous iteration is used, and a stop update mechanism is introduced. Stop updating the second-order information matrix of the layer when the amount of change in the trace is less than a certain threshold. The specific update formula is as follows:

$$
\begin{cases}
update\ F^{k}_{i} , \qquad\qquad\qquad\qquad\qquad\qquad\qquad\qquad\qquad\quad\ \ if \ \Delta^{k} \in (\omega_{1},+\infty)\\
do\ not\ update\ F^{k}_{i}\ and\ set \  F^{k}_{i}=F^{k-1}_{i}, \ \quad\qquad\qquad\qquad\quad if \ \Delta^{k} \in [\omega_{2},\omega_{1}]\\
stop\ update\ F^{k}_{i}\ and\ set \  F^{k+t}_{t}\equiv F^{k-1}_{i}\ for\ all\ t=1,2,...\quad if \ \Delta^{k} \in [0,\omega_{2})
\end{cases}
$$

where

$$\Delta^k=\frac{||tr(F^k_i+\lambda I)|-|tr(F^{k-1}_i+\lambda I)||}{|tr(F^{k-1}_i+\lambda I)|}$$

### Hardware Awareness Matrix Slicing

THOR further assumes that the input and output blocks in each network layer are also independent of each other, based on the decoupling of Fisher matrices by layer. For example, the input and output of each layer of the network is sliced into n blocks, which are independent of each other, and the second-order information matrix is further sliced according to this assumption, thus improving the computational efficiency. THOR combines matrix information loss data and matrix performance data to determine the matrix tiling dimension, thus greatly improving the Fisher matrix inversion time.

So how can we determine the matrix tiling dimension? The specific method is:

1. Determine the matrix slice dimension based on the layer with the largest dimension in the Fisher matrix. Taking ResNet50 as an example, the maximum dimension in the network layer is 2048, and the matrix slice dimensions are determined as [1,16,32,64,128,256,512,1024,2048].

2. Based on the determined matrix dimensions, the matrix loss under each dimension is calculated according to the spectral norm by the following equation:

    $$L=1-\sqrt{\frac{\lambda_{max}\ \ (\hat{A}\hat{A}^T)}{\lambda_{max}\ \ (AA^T)}}$$

    where $\lambda_{max}(X)$ denotes the maximum feature value of the matrix $X$, $A$ denotes the original unpartitioned matrix, and $\hat A$ denotes the partitioned matrix. Then the number of matrices with losses less than 1% in that dimension is counted, and finally the normalized matrix loss information is obtained by dividing by the total number of matrices.

3. According to the determined matrix dimensions, the matrix inversion time under each dimension is calculated, and then the normalized performance data under each dimension is obtained by the formula $normalized_n = \frac{p_1}{p_n}$, where $p_1$ denotes the performance data of the matrix with the smallest dimension and $p_n$ denotes the performance data under the nth dimension.

4. Based on the annotated matrix loss information and the normalized performance data graph, taking the ResNet50 as example, shown in Figure 3, the falling curve in the figure is the performance curve, and the rising curve indicates the matrix loss curve. The intersection point in the figure is 106, which is closest to 128, and finally the matrix slice dimension is determined to be 128.

![The split dimension of matrix](https://gitee.com/mindspore/docs/raw/r2.0.0-alpha/tutorials/experts/source_zh_cn/optimize/thor/images/split_dimension.png)

*Figure 3 Schematic Diagram of Slice Dimension Determination*

### Results

Figure 4 shows the training line graph of THOR on ResNet50+ImageNet with a batchsize of 256 on first and second order, where train loss denotes training error, test accuracy denotes testing accuracy, epoch denotes the number of iterations, and wall-clock time denotes the time. The faster falling curve and the faster rising curve are the curves of this algorithm, and the other curve with more obvious gap is the training curve of momentum.

![The result of ResNet50](https://gitee.com/mindspore/docs/raw/r2.0.0-alpha/tutorials/experts/source_zh_cn/optimize/thor/images/thor_in_resnet.png)

*Figure 4 Results of THOR on ResNet50*

THOR, THOR_stop, and THOR_NT in Figure 4 indicate ($w_1$,$w_2$)=(0.01,0), ($w_1$,$w_2$)=(0.01,0.001), and ($w_1$,$w_2$)=(0,0), respectively. From the figure, we can see that the number of iterations required for THOR convergence is about half of the first order, and the time of the single-step is not significantly different from that of the first order. Compared to the first-order algorithm that takes 117 min, the second-order optimizer speeds up the device-to-device time by about 40%.

THOR also tested the convergence results of ResNet50+ImageNet at different batchsize, and the results are shown in Figure 5 below, where Hardware denotes the hardware platform, Software is the used deep learning framework, Batch size is the number of images per training, Optimizer denotes the used optimizer, Time refers to the overall training time, and Accuracy is the final convergence accuracy. When the batchsize is 8192 and 256 Ascend 910 blocks are used, it only takes 2.7 minutes for the accuracy to converge to 75.9%.

![The large batchsize result of ResNet50](https://gitee.com/mindspore/docs/raw/r2.0.0-alpha/tutorials/experts/source_zh_cn/optimize/thor/images/thor_largebs_in_resnet.png)

*Figure 5 Results of THOR on ResNet50 at Large Batchsize*

In BERT+WIkipedia, THOR also has a good performance effect. Taking the MLPerf as a standard, the accuracy reaches to 71.2%, and an end-to-end improvement of 30% is implemented compared to the first order. The results are shown in Figure 6. The horizontal coordinate in the figure indicates the training time, and the vertical coordinate indicates the test accuracy. The curve that rises faster is the training curve of THOR, and the other one is the training curve of Lamb.

![The result of BERT](https://gitee.com/mindspore/docs/raw/r2.0.0-alpha/tutorials/experts/source_zh_cn/optimize/thor/images/thor_in_bert.png)

*Figure 6 Results of THOR on BERT*
