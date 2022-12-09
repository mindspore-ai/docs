# Data Driven

<a href="https://gitee.com/mindspore/docs/blob/master/docs/mindsponge/docs/source_en/intro/data_driven.md" target="_blank"><img src="https://mindspore-website.obs.cn-north-4.myhuaweicloud.com/website-images/master/resource/_static/logo_source_en.png"></a>

The data-driven approach is based on existing physical, chemical and biological data, and applies machine learning methods to achieve molecular learning tasks.

The development of data-driven approaches is the result of a combination of data accumulation and advances in AI technology. With the continuous expansion of data such as protein, DNA, RNA and other biological macromolecule sequences and structure database, small molecule structure, property database, and molecular simulation data, the application of one or more kinds of data can train the AI model to learn the representation, property, correlation, etc., in order to achieve downstream tasks.

In recent years, deep learning technology develops rapidly. AI algorithms such as CNN, Transformer and its derivative architectures, graph neural networks, and AI models such as GAN and VAE are widely used in various molecular learning tasks such as pre-training model, structure prediction, property prediction, molecular design or generation.

Among them, the pre-training model is usually based on the massive size molecular sequences or graph representation, and the medium and large scale models with strong mobility are built. These models can adapt to a variety of downstream tasks through fine tuning:

1. Molecular structure prediction mainly focuses on 3D structure or conformation prediction based on sequence and molecular formula, including protein structure prediction, small molecule conformation prediction, molecular interaction conformation prediction and binding interface prediction. AlphaFold2 is a representative work in this field;
2. Molecular property prediction focuses more on obtaining specific molecular properties directly from the model based on existing data, such as the druggability of small molecules, water solubility, protein stability, activity, etc. In addition to the direct design and training of the AI model, it can often be achieved through the fine-tuning of the pre-training model;
3. Molecular design mainly produces sequences or structural expressions of size molecules conforming to specific distribution or conditions.