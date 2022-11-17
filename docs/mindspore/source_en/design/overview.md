# MindSpore Design Overview

<a href="https://gitee.com/mindspore/docs/blob/master/docs/mindspore/source_en/design/overview.md" target="_blank">
<img src="https://mindspore-website.obs.cn-north-4.myhuaweicloud.com/website-images/master/resource/_static/logo_source_en.png">
</a>

## Overview

Artificial intelligence (AI) frameworks have been in development for nearly 10 years, and four main lines drive the evolution and development of AI frameworks:

1. For developers: Balance efficiency and operation performance of algorithm development.
2. For hardware: Fully utilize the performance of the chip and cluster.
3. For algorithm and data: In terms of calculation scale, need to address the increasing challenges of the model; in terms of computational paradigm, need to handle the new computational loads that are constantly emerging.
4. For deployment: AI capabilities need to be deployed to every device, every application, and every industry.

MindSpore is an AI framework designed for "device-edge-cloud" full scenarios, aiming to bridge the gap between AI algorithm research and production deployment.

During the algorithm research phase, provide developers with a unified programming experience to improve the efficiency of algorithm development. During the production phase, automatic parallelism can greatly accelerate the development and debugging efficiency of distributed training, while fully exploiting the computing power of heterogeneous hardware. During the deployment stage, based on the "device-edge-cloud" unified architecture, it addresses the challenges of enterprise-level deployment and security trustworthiness.

## Overall Architecture

The overall MindSpore architecture is divided into four layers:

1. Model layer, providing users with usable-upon-unpacking function. This layer mainly contains repositories for expanding hot research areas, such as pre-built models and development kits, graph neural networks (GNN) and deep probabilistic programming.
2. MindExpression layer, providing users with interfaces for AI model development, training, and inference. Support users to develop and debug neural networks with native Python syntax. Its unique ability to unify dynamic and static graphs allows developers to balance development efficiency and execution performance, while the layer provides a full-scenario unified C++ interface during the production and deployment phases.
3. MindCompiler, as the core of the AI framework, compiles front-end expressions into a more efficient underlying language for execution, using the full-scenario unified MindIR as the medium, and performs simultaneous global performance optimization, including hardware-independent optimization such as automatic differentiation and algebraic simplification, as well as hardware-related optimization such as graph-kernel fusion and operator generation.
4. At runtime, the underlying hardware operator is docked and called according to the compiled and optimized results of the upper layer, while the "device-edge-cloud" unified runtime architecture supports "device-edge-cloud" AI collaboration, including federated learning.

![arch](https://mindspore-website.obs.cn-north-4.myhuaweicloud.com/website-images/master/tutorials/source_en/beginner/images/introduction2.png)

## Design Concept

MindSpore provides users with programming paradigms for languages such as Python. Based on source code conversion, users can use native Python control syntax and other advanced APIs such as Tuple, List, and Lambda expressions.

### Functional Programming Interface

MindSpore offers both object-oriented and function-oriented programming paradigms.

The user can define the AI network or a layer of the network based on the derivation of the nn.Cell class for the desired function, and assemble various layers defined by nested calls of objects to complete the definition of the whole AI network.

Also users can define a Python pure function that can be converted by MindSpore source-to-source compilation, and speed up its execution with the functions or decorators provided by MindSpore. Python pure functions can support subfunction nesting, control logic, and even recursive function expressions while satisfying the requirements of MindSpore static syntax. Therefore, based on this programming paradigm, users have the flexibility to enable a number of functional features.

[Automatic differentiation based on source code conversion](https://www.mindspore.cn/docs/en/master/design/auto_gradient.html): Unlike the automatic differentiation mechanism of common AI frameworks. Based on source code conversion technology, MindSpore gets the Cell object or Python pure function that needs to be derived. The syntax is parsed, function objects that can be differentiated are constructed, and the derivation is performed based on the call chain according to the call relationship.

For conventional automatic differentiation, there are three main types of mainstream:

- Conversion based on graph method: The network is converted to a static data flow graph at compile time, and then the chain rules are converted to a data flow graph for automatic differentiation.
- Conversion based on operator overloading: The operation trajectory of the network during forward execution is recorded in the form of operator overloading, and then the chain rule is applied to the dynamically generated data flow graph to achieve automatic differentiation.
- Conversion based on source code: The technology evolved from a functional programming framework to perform automatic differential transformations of intermediate expressions (the form of an expression of program during compilation) in the form of just-in-time compilation (JIT), supporting complex process control scenarios, higher-order functions, and closures.

The graph method can optimize network performance by using static compilation techniques, but it is very complex to set up or debug the network. Dynamic graphs built on operator overloading techniques are very easy to use, but difficult to optimize to the limit in terms of performance.

The automatic differentiation strategy adopted by MindSpore, based on source code transformations, has an intuitive correspondence with the complex functions in basic algebra. As long as the derivation formula of the basis function is known, the derivation formula of the composite function composed of any basis function can be derived. Thus, programmability and performance are balanced.

On the one hand, it is possible to maintain a consistent programming experience with the programming language, on the other hand, it is a differentiable technique at [Intermediate Representation](https://www.mindspore.cn/docs/en/master/design/mindir.html) (IR) granularity, which can reuse the optimization ability of modern compilers and has better performance.

Also based on the functional programming paradigm, MindSpore provides a wealth of higher-order functions such as vmap, shard, and other built-in higher-order functions. Like the differential derivative function grad, it allows the user to conveniently construct a function or an object that can be used as an argument to a higher-order function. Higher-order functions are internally compiled and optimized to generate optimized versions of user-specific functions, implementing features such as vectorized transformations, distributed parallel slicing, and other functions.

### Device-edge-cloud Full Scenarios

MindSpore is an all-in-one AI framework that supports both training and inference. MindSpore also supports CPU, GPU, NPU and other chips, and provides a unified programming interface on different chips and generates offline models that can be loaded and executed on multiple hardware.

MindSpore provides a variety of versions according to the actual execution environment and business requirements, supports deployment and execution on embedded devices such as cloud, server, and cell phones, and ultra-lightweight devices such as headsets.

### Unified programming experience with Dynamic Graph and Static Graph

Traditional AI frameworks have two main forms of programming execution, static graph mode and dynamic graph mode.

Based on the framework interface called by the user, the static graph mode will be compiled and executed as the graph structure of the neural network before executing the computational operations involved in the graph during compilation execution.

The static graph mode can effectively perceive the relationship situation between the operators of each layer in the neural network and perform effective compilation optimization based on the compilation technique to improve the performance. However, traditional static graphs require user-aware composition interfaces, making it more complicated to set up or debug networks, and difficult to interleave with common Python libraries and custom Python functions.

Dynamic graph mode can effectively solve the more complex problems of programming static graphs. However, because the program is executed in the order in which the code is written, no integral-graph compilation optimization is done, resulting in less room for relative performance optimization, especially for proprietary hardware such as DSA, which is more difficult to enable.

MindSpore builds the graph structure of neural networks based on the source code transformation mechanism, which can be more easily expressed than the traditional static graph model. It is also better compatible with programming interfaces for dynamic and static graphs, such as for control flow, and dynamic graphs can be programmed directly based on Python control flow keyword. Static graphs require programming based on special control flow operators or require user programming to instruct the control flow execution branches, which results in large programming differences between dynamic and static graphs.

MindSpore source code conversion mechanism enables the execution of static graph mode directly based on the Python control flow keyword, making the programming of dynamic and static graphs more uniform. At the same time, users can flexibly control the dynamic and static graph mode of Python code fragments based on interfaces of MindSpore. That is, it is possible to execute the local functions of the program in static graph mode while the other functions are executed in dynamic graph mode. Thus, when interleaved with common Python libraries and custom Python functions, users have the flexibility to specify function fragments for static graph optimization acceleration without sacrificing the ease of programming for interleaved execution.

### [Automatic Parallism](https://www.mindspore.cn/docs/en/master/design/distributed_training_design.html)

MindSpore addresses the problem of increasingly large DL networks that require complex and multiple distributed parallel strategies, and the framework provides a built-in multi-dimensional distributed training strategy that can be flexibly assembled and used by users. It also simplifies the complexity of parallel programming for users by hiding communication operations through parallel abstraction.

Transparent and efficient distributed training capabilities are provided through automatic parallel strategy search. "Transparent" means that users can change one line of configuration, submit one version of Python code, and run that version of Python code on multiple devices for training. "Efficient" means that the algorithm chooses a parallel strategy with minimal cost, reducing computational and communication overhead.

MindSpore introduces Tensor Redistribution (TR) in parallelized strategy search, which enables the device layout of the output tensor to be converted before being input to subsequent operators. MindSpore identifies the output data overlap of the operator under different input data slices, and based on this, it performs slice derivation and automatically generates the corresponding tensor redistribution plan. Based on this plan, multiple parallel strategies such as data parallelism and model parallelism can be expressed uniformly.

At the same time, MindSpore is oriented to distributed training, and also provides various parallel strategies such as pipeline parallelism, optimizer parallelism, and recomputation for users to use.

### High-performance Hardware

MindSpore provides rich hardware-independent optimizations such as IR fusion, algebraic simplification, constant folding, common subexpression elimination based on compilation technology. MindSpore also provides various hardware optimization capabilities for different hardware such as NPU and GPU, so as to better utilize the large-scale computing acceleration capability of hardware.

MindSpore offers some more distinctive techniques in addition to the usual optimizations of traditional AI frameworks:

#### [Graph-kernel Fusion](https://www.mindspore.cn/docs/en/master/design/graph_fusion_engine.html)

Mainstream AI computing frameworks such as MindSpore provide operators to users that is usually defined in terms of understandable and easy use for user. Each operator carries a different amount of computation and varies in computational complexity. However, from the hardware execution point of view, this natural, user perspective-based division of operator computation volume is not efficient and does not fully utilize the computational power of hardware resources, which is mainly reflected in the following aspects:

1. Computationally overloaded and overly complex operators, which usually makes it difficult to generate well-cut high-performance operator, thereby reducing equipment utilization.
2. Operators that are too small in computation may also cause latency in computation and thus reduce equipment utilization, as the computation cannot effectively hide the data moving overhead.
3. Hardware Devices are usually multi-core, many-core architectures. When the operator shape is small or other reasons cause insufficient computational parallelism, it may cause some cores to be idle, thus reducing the device utilization. In particular, chips based on Domain Specific Architecture (DSA for short) are more sensitive to these factors. It has been a big challenge to maximize the performance of hardware operator while making the operator easy to use.

In terms of AI framework design, the current industry mainstream adopts a separate layer implementation approach of graph and operator layers. The graph layer is responsible for fusing or regrouping the computational graph, and the operator layer is responsible for compiling the fused or regrouped operators into high-performance executable operators. The graph layer is usually processed and optimized by using Tensor-based High-Level IR, while the operator layer is analyzed and optimized by using computational instruction-based Low-Level IR. This artificial separate-layer process significantly increases the difficulty of performing collaborative optimization in both graph and computational layers.

MindSpore has adopted the technique of graph kernel fusion to better solve this problem in the past few years. Typical networks in different categories such as NLP and recommendation show significant gains in training speed after enabling graph kernel fusion. One of the main reasons is the presence of a large number of small operator combinations in these networks, which have more opportunities for fusion optimization.

#### Competitive Optimization for Ascend Hardware

The Device in On Device usually refers to the Ascend AI processor. The AICORE, AICPU and CPU are integrated on the Ascend chip, where the AICORE is responsible for large Tensor Vector operations, the AICPU is responsible for scalar operations, and the CPU is responsible for logic control and task distribution.

The CPU on the Host side is responsible for sending the graph or operator down to the Ascend chip. With the functions of computing, logic control and task distribution, the Ascend chip does not need to interact frequently with the CPU on the Host side, but only needs to return the final result after computation to the Host side to realize the whole graph sinking to the Device for execution, avoiding frequent Host-Device interaction and reducing the overhead.

The whole computational graph is sunk to the Device to reduce the Host-Device interaction overhead. It can be combined with cyclic sinking to achieve multiple Step sinking to further reduce the number of Host-Device interactions.

The iteration-offload is an optimization based on the On Device execution to further reduce the number of interactions between the Host side and the Device side. Normally, each step returns a result, and the iteration-offload controls number of steps at which the result is returned.

Data sink means that the data is transferred directly to the Device through the channel.

#### Algorithm Optimization

The algorithm optimization includes second-order optimization, boost optimization, etc.

### [Overall Security and Trustworthiness Design](https://www.mindspore.cn/mindarmour/docs/en/master/design.html)

MindSpore takes into account the rich need for security and trustworthiness when deployed by enterprises. In the continuous evolution and refinement of various technologies geared towards secure and trustworthy directions, with built-in frameworks:

1. Adversarial attack defense

    Adversarial attacks are a growing threat to the security of machine learning models. Attackers can deceive machine learning models by adding small perturbations to the original samples that are not easily perceived by humans.

    To defend against adversarial attacks, MindSpore security component MindArmour provides attack (adversarial sample generation), defense (adversarial sample detection and adversarial training), and evaluation (model robustness evaluation and visualization). Given a model and input data, the attack module provides a simple API that is capable of generating corresponding adversarial samples in both black-box and white-box attack scenarios. These generated adversarial samples are fed into the defense module to improve the generalization ability and robustness of the machine learning model. The defense module also implements multiple detection algorithms that can distinguish between adversarial and normal samples based on malicious content or attack behavior. The evaluation module provides a variety of evaluation metrics that enable developers to easily assess and visualize the robustness of their models.

2. Privacy protection artificial intelligence

    Privacy protection is also an important topic for AI applications. MindArmour considers privacy protection in machine learning and provides corresponding privacy protection features.

    To address the problem that trained models may leak sensitive information in the training dataset, MindArmour implements a series of differential privacy optimizers that automatically add noise to the gradients generated by the inverse computation, thus providing differential privacy guarantees for trained models. In particular, the optimizer adaptively adds noise according to the training process, enabling better model availability with the same privacy budget. A monitoring module is also provided to enable dynamic monitoring of the privacy budget consumption during training. Users can use these differential privacy optimizers just like normal optimizers.

3. Device-side learning and federated learning

    While deep learning models trained on large datasets are somewhat general, in some scenarios these models are still not applicable to the user's own data or personalization tasks. MindSpore offers an device-side training solution that allows users to train their own personalized models or fine-tune existing models on their devices, while avoiding data privacy, bandwidth limitations and network connectivity issues. The device-side will provide a variety of training strategies, such as initialized training strategies, migration learning, and incremental learning. MindSpore supports federated learning where models can learn more general knowledge by sending model updates/gradients to the cloud side to share different data.

