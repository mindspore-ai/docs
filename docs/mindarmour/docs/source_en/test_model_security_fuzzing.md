# Testing Model Security Using Fuzz Testing

<a href="https://gitee.com/mindspore/docs/blob/master/docs/mindarmour/docs/source_en/test_model_security_fuzzing.md" target="_blank"><img src="https://gitee.com/mindspore/docs/raw/master/resource/_static/logo_source_en.png"></a>&nbsp;&nbsp;

## Overview

The decision logic of traditional software is determined by the code logic. Traditional software determines whether the test is adequate based on the code line coverage rate. Ideally, the higher the coverage rate is, the more adequate the code test is. However, for deep neural network, the decision logic of the program is determined by the training data, network structure, and parameters through a black box mechanism. The code line coverage fails to evaluate the test adequacy. A more suitable test evaluation criterion needs to be selected according to the deep network features to guide the neural network to perform a more adequate test and find more corner error cases, thereby ensuring universality and robustness of a model.

The fuzz testing module of MindArmour uses the neuron coverage rate as the test evaluation criterion. Neuron coverage is the range of the number of neurons observed and activated and the range of the neuron output value through a set of inputs. The neuron coverage is used to guide input mutation so that the input can activate more neurons and neuron values can be distributed in a wider range. In this way, we can explore different types of model output results and incorrect behaviors.

The LeNet model and MNIST dataset are used as an example to describe how to use Fuzz testing.

> This example is for CPUs, GPUs, and Ascend 910 AI processors. Currently only supports GRAPH_MODE. You can download the complete sample code at <https://gitee.com/mindspore/mindarmour/blob/master/examples/ai_fuzzer/lenet5_mnist_fuzzing.py>.

## Implementation

### Importing Library Files

The following lists the required common modules, MindSpore-related modules, Fuzz testing feature modules, and configuration log labels and log levels.

Here, we use `KMultisectionNeuronCoverage` as fuzzing guide, you can also choose other supported neuron coverage metrics: `NeuronCoverage`, `TopKNeuronCoverage`, `NeuronBoundsCoverage`, `SuperNeuronActivateCoverage`.

```python
import numpy as np
from mindspore import Model
from mindspore import context
from mindspore import load_checkpoint, load_param_into_net

from mindarmour.fuzz_testing import Fuzzer
from mindarmour.fuzz_testing import KMultisectionNeuronCoverage
from mindarmour.utils import LogUtil

from examples.common.dataset.data_processing import generate_mnist_dataset
from examples.common.networks.lenet5.lenet5_net_for_fuzzing import LeNet5

LOGGER = LogUtil.get_instance()
TAG = 'Fuzz_testing'
LOGGER.set_level('INFO')
```

### Parameter Configuration

Configure necessary information, including the environment information and execution mode.

```python
context.set_context(mode=context.GRAPH_MODE, device_target="Ascend")
```

For details about the API configuration, see the `context.set_context`.

### Fuzz Testing Application

1. Create a LeNet model and load the MNIST dataset. The operation is the same as that for [Model Security](https://www.mindspore.cn/mindarmour/docs/en/master/improve_model_security_nad.html).

   ```python
   ...
   # Lenet model
   model = Model(net)
   # get training data
   mnist_path = "../common/dataset/MNIST/"
   batch_size = 32
   ds = generate_mnist_dataset(os.path.join(mnist_path, "train"), batch_size, sparse=False)
   train_images = []
   for data in ds.create_tuple_iterator():
       images = data[0].asnumpy().astype(np.float32)
       train_images.append(images)
   train_images = np.concatenate(train_images, axis=0)

   # get test data
   batch_size = 32
   ds = generate_mnist_dataset(os.path.join(mnist_path, "test"), batch_size, sparse=False)
   test_images = []
   test_labels = []
   for data in ds.create_tuple_iterator():
       images = data[0].asnumpy().astype(np.float32)
       labels = data[1].asnumpy()
       test_images.append(images)
       test_labels.append(labels)
   test_images = np.concatenate(test_images, axis=0)
   test_labels = np.concatenate(test_labels, axis=0)
   ```

2. Configure Fuzzer parameters.

   Set the data mutation method and parameters. Multiple methods can be configured at the same time. Currently, the following data mutation methods are supported:

   - Natural Robustness Methods:
       - Image affine transformation methods: Translate, Scale, Shear, Rotate, Perspective, Curve;
       - Image blur methods: GaussianBlur, MotionBlur, GradientBlur;
       - Luminance adjustment methods: Contrast, GradientLuminance;
       - Add noise methods: UniformNoise, GaussianNoise, SaltAndPepperNoise, NaturalNoise.
   - Methods for generating adversarial examples based on white-box and black-box attacks: FGSM(FastGradientSignMethod), PGD(ProjectedGradientDescent), and MDIIM(MomentumDiverseInputIterativeMethod).

   The data mutation method must include the method based on the image pixel value changes.

   The first two image transform methods support user-defined configuration parameters and randomly generated parameters by algorithms. For user-defined configuration parameters see the class methods corresponding to <https://gitee.com/mindspore/mindarmour/tree/master/mindarmour/natural_robustness/transform/image>. For randomly generated parameters by algorithms you can set method's params to `'auto_param': [True]`. The mutation parameters are randomly generated within the recommended range.

   For details about how to set parameters based on the attack defense method, see the corresponding attack method class.

   The following is an example for configure Fuzzer parameters.

   ```python
   mutate_config = [
        {'method': 'GaussianBlur',
         'params': {'ksize': [1, 2, 3, 5],
                    'auto_param': [True, False]}},
        {'method': 'MotionBlur',
         'params': {'degree': [1, 2, 5], 'angle': [45, 10, 100, 140, 210, 270, 300], 'auto_param': [True]}},
        {'method': 'GradientBlur',
         'params': {'point': [[10, 10]], 'auto_param': [True]}},
        {'method': 'UniformNoise',
         'params': {'factor': [0.1, 0.2, 0.3], 'auto_param': [False, True]}},
        {'method': 'GaussianNoise',
         'params': {'factor': [0.1, 0.2, 0.3], 'auto_param': [False, True]}},
        {'method': 'SaltAndPepperNoise',
         'params': {'factor': [0.1, 0.2, 0.3], 'auto_param': [False, True]}},
        {'method': 'NaturalNoise',
         'params': {'ratio': [0.1, 0.2, 0.3], 'k_x_range': [(1, 3), (1, 5)], 'k_y_range': [(1, 5)],
                    'auto_param': [False, True]}},
        {'method': 'Contrast',
         'params': {'alpha': [0.5, 1, 1.5], 'beta': [-10, 0, 10], 'auto_param': [False, True]}},
        {'method': 'GradientLuminance',
         'params': {'color_start': [(0, 0, 0)], 'color_end': [(255, 255, 255)], 'start_point': [(10, 10)],
                    'scope': [0.5], 'pattern': ['light'], 'bright_rate': [0.3], 'mode': ['circle'],
                    'auto_param': [False, True]}},
        {'method': 'Translate',
         'params': {'x_bias': [0, 0.05, -0.05], 'y_bias': [0, -0.05, 0.05], 'auto_param': [False, True]}},
        {'method': 'Scale',
         'params': {'factor_x': [1, 0.9], 'factor_y': [1, 0.9], 'auto_param': [False, True]}},
        {'method': 'Shear',
         'params': {'factor': [0.2, 0.1], 'direction': ['horizontal', 'vertical'], 'auto_param': [False, True]}},
        {'method': 'Rotate',
         'params': {'angle': [20, 90], 'auto_param': [False, True]}},
        {'method': 'Perspective',
         'params': {'ori_pos': [[[0, 0], [0, 800], [800, 0], [800, 800]]],
                    'dst_pos': [[[50, 0], [0, 800], [780, 0], [800, 800]]], 'auto_param': [False, True]}},
        {'method': 'Curve',
         'params': {'curves': [5], 'depth': [2], 'mode': ['vertical'], 'auto_param': [False, True]}},
        {'method': 'FGSM',
         'params': {'eps': [0.3, 0.2, 0.4], 'alpha': [0.1], 'bounds': [(0, 1)]}},
        {'method': 'PGD',
         'params': {'eps': [0.1, 0.2, 0.4], 'eps_iter': [0.05, 0.1], 'nb_iter': [1, 3]}},
        {'method': 'MDIIM',
         'params': {'eps': [0.1, 0.2, 0.4], 'prob': [0.5, 0.1],
                    'norm_level': [1, 2, '1', '2', 'l1', 'l2', 'inf', 'np.inf', 'linf']}}
       ]
   ```

   Initialize the seed queue. Each seed in the seed queue has two values: original image and image label. Here we select 100 samples as initial seed queue.

   ```python
   # make initial seeds
   initial_seeds = []
   for img, label in zip(test_images, test_labels):
    initial_seeds.append([img, label])
   initial_seeds = initial_seeds[:100]
   ```

4. Instantiate class `KMultisectionNeuronCoverage` and test the k-multisection neuron coverage rate before the fuzz testing.

   ```python
   coverage = KMultisectionNeuronCoverage(model, train_images, segmented_num=100, incremental=True)
   kmnc = coverage.get_metrics(test_images[:100])
   print('KMNC of initial seeds is: ', kmnc)
   ```

   Result:

   ```text
   KMNC of initial seeds is:  0.3152149321266968
   ```

5. Perform the fuzz testing.

   ```python
   model_fuzz_test = Fuzzer(model)
   fuzz_samples, true_labels, fuzz_preds, fuzz_strategies, metrics_report = model_fuzz_test.fuzzing(mutate_config, initial_seeds, coverage, evaluate=True, max_iters=10,mutate_num_per_seed=20)
   ```

6. Experiment results.

   The results of fuzz testing contains five aspect data:

   - fuzz_samples: mutated samples in fuzz testing.
   - true_labels: the ground truth labels of fuzz_samples.
   - fuzz_pred: predictions of tested model about fuzz_samples.
   - fuzz_strategies: the methods used to mutate fuzz_samples.
   - metrics_report: metrics report of fuzz testing.

   The first 4 returns can be used to further calculated complex metrics and analyze the robustness of the model.

   Run the following command to view the result:

   ```python
   if metrics:
       for key in metrics:
           LOGGER.info(TAG, key + ': %s', metrics[key])
   ```

   The fuzz testing result is as follows:

   ```text
   Accuracy:  0.445
   Attack_success_rate:  0.375
   coverage_metrics:  0.43835972850678734
   ```

   Before the fuzzing test, the KMNC neuron coverage rate of the seed is 31.5%. After the fuzzing test, the KMNC neuron coverage rate is 43.8%, and the neuron coverage rate and sample diversity increase. After the fuzzing test, the accuracy rate of the model to generate samples is 44.5%, and the attack success rate is 37.5% for samples using the adversarial attack method. Since the initial seed, the mutation method and the corresponding parameters are all randomly selected, it is normal that the result floats to some extent.

   Original image:

   ![fuzz_seed](./images/fuzz_seed.png)

   Mutation images generated by fuzzing:

   ![fuzz_res](./images/fuzz_res.png)
