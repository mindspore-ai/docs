# Defining the Network

Translator: [huqi](https://gitee.com/hu-qi)

`Linux` `Ascend` `GPU` `CPU` `Model Development` `Beginner` `Intermediate` `Expert`

[![View Source On Gitee](../_static/logo_source.png)](https://gitee.com/mindspore/docs/blob/r1.1/tutorials/training/source_en/use/defining_the_network.md)

A neural network model composed of multiple layers is an important part of the training process. You can build a network model based on the base class of `nn.Cell` in MindSpore by initializing the `__init__` method and constructing the `construct` method. There are several ways to define the network model:

- Use the official network model directly.

  It is recommended to consult the current [Network Support List](https://www.mindspore.cn/doc/note/en/r1.1/network_list_ms.html) provided by MindSpore to directly use the corresponding network model. In the network support list, the platforms supported by each network are provided. Click the corresponding network name to view the definition of the network. Users can customize the network initialization parameters according to their needs.

- Build your own network.

    - If the built-in operators in the network are not enough to meet your needs, you can use MindSpore to customize the operators quickly and easily and add them to the network.

      Go to [Custom Operators](https://www.mindspore.cn/tutorial/training/en/r1.1/advanced_use/custom_operator.html) for detailed help information.

    - MindSpore provides scripts for migrating third-party training frameworks, and supports the migration of existing TensorFlow, PyTorch, etc. networks to MindSpore to help you quickly migrate the network.

      Go to [Migrating Training Scripts from Third Party Frameworks](https://www.mindspore.cn/tutorial/training/en/r1.1/advanced_use/migrate_script.html) for detailed help information.

    - MindSpore supports probabilistic programming using the logic of developing deep learning models, and also provides a toolbox for deep probabilistic learning to build Bayesian neural networks.

      Go to [Deep Probability Programming](https://www.mindspore.cn/tutorial/training/en/r1.1/advanced_use/apply_deep_probability_programming.html) for detailed help information.
