# File Method for Converting TensorFlow Models to MindSpore Models

<a href="https://gitee.com/mindspore/docs/blob/master/docs/mindspore/source_en/migration_guide/tensorflow2mindspore.md" target="_blank"><img src="https://mindspore-website.obs.cn-north-4.myhuaweicloud.com/website-images/master/resource/_static/logo_source_en.png"></a>

In this chapter, we will take LeNet5 network structure as an example and introduce the model weight conversion method in detail with [code](https://gitee.com/mindspore/docs/tree/master/docs/sample_code/convert_tf2ms_code).

First we need to clarify the trained TensorFlow model into a checkpoint that MindSpore can use, which basically requires the following processes:

1. Print the parameter names and shapes of all parameters inside the parameter file of TensorFlow, and print the parameter names and shapes of all parameters in the MindSpore Cell where the parameters need to be loaded.
2. Compare parameter name and shape to construct parameter mapping relationship.
3. Map the TensorFlow parameters to MindSpore Parameter according to the parameter mapping and form a Parameter List and then save it as a checkpoint.
4. Unit test: MindSpore loads the transformed parameters, fixes the input, and compares the results of MindSpore with TensorFlow.

## Printing the Parameter Information

```python
# Read the name of the model parameter and the shape of the corresponding parameter according to the TensorFlow parameter file
def tensorflow_param(ckpt_path):
    """Get TensorFlow parameter and shape"""
    tf_params = {}
    reader = tf.train.load_checkpoint(ckpt_path)
    for name in reader.get_variable_to_shape_map():
        try:
            print(name, reader.get_tensor(name).shape)
            tf_params[name] = reader.get_tensor(name)
        except Exception as e:
            pass
    return tf_params

# Print the parameter names and shapes of all the parameters in the Cell through MindSpore Cell, and return the parameter dictionary
def mindspore_params(network):
    """Get MindSpore parameter and shape"""
    ms_params = {}
    for param in network.get_parameters():
        name = param.name
        value = param.data.asnumpy()
        print(name, value.shape)
        ms_params[name] = value
    return ms_params
```

Execute the following code:

```python
from ms_lenet import LeNet5
tf_ckpt_path = './checkpoint_dir'
tensorflow_param(tf_ckpt_path)
print("*"*20)
network = LeNet5()
mindspore_params(network)
```

The outputs are as follows:

```text
fc3/dense/kernel (84, 1)
fc3/dense/bias (1,)
conv1/weight (5, 5, 1, 6)
fc1/dense/bias (120,)
fc1/dense/kernel (400, 120)
fc2/dense/bias (84,)
conv2/weight (5, 5, 6, 16)
fc2/dense/kernel (120, 84)
******************************
conv1.weight (6, 1, 5, 5)
conv2.weight (16, 6, 5, 5)
fc1.weight (120, 400)
fc1.bias (120,)
fc2.weight (84, 120)
fc2.bias (84,)
fc3.weight (1, 84)
fc3.bias (1,)

```

## Parameter Mapping and Checkpoint Saving

By comparing the above parameter names with the output of the shape, we can find that there is a certain regularity that can be matched with the network structure between the two parameter names. For the parameter shape, we can find that the shape dimension of convolutional and fully connected layers are different.
A shape of weight is [out_channel, in_channel, kernel_height, kernel_weight] in MindSpore convolutional layer, while a shape of weight is [kernel_height, kernel_weight, in_channel, out_channel] in TensorFlow convolutional layer. A shape of weight is [out_channel, in_channel] in the MindSpore fully-connected layer, while a  shape of weight is [out_channel, in_channel] in the TensorFlow fully-connected layer, so here we need to perform transposition when dealing with convolution and fully-connected layer weight conversion.

```python
def tensorflow2mindspore(tf_ckpt_dir, param_mapping_dict, ms_ckpt_path):

    reader = tf.train.load_checkpoint(tf_ckpt_dir)
    new_params_list = []
    for name in param_mapping_dict:
        param_dict = {}
        parameter = reader.get_tensor(name)
        if 'conv' in name and 'weight' in name:
            # Transpose the convolution weights
            parameter = np.transpose(parameter, axes=[3, 2, 0, 1])
        if 'fc' in name and 'kernel' in name:
            parameter = np.transpose(parameter, axes=[1, 0])
        param_dict['name'] = param_mapping_dict[name]
        param_dict['data'] = Tensor(parameter)
        new_params_list.append(param_dict)
    save_checkpoint(new_params_list, os.path.join(ms_ckpt_path, 'tf2mindspore.ckpt'))
```

Because the parameter name mapping of the current network is very complex, it is difficult to find the mapping relationship by parameter name, so we need to map dictionary through a parameter. When encountering simpler parameter name mappings, the
conversion method can be found in [Convert PyTorch model file to MindSpore model file](https://www.mindspore.cn/docs/en/master/migration_guide/sample_code.html#model-validation).

```python
params_mapping = {
    "conv1/weight":"conv1.weight",
    "conv2/weight":"conv2.weight",
    "fc1/dense/kernel":"fc1.weight",
    "fc1/dense/bias":"fc1.bias",
    "fc2/dense/kernel":"fc2.weight",
    "fc2/dense/bias":"fc2.bias",
    "fc3/dense/kernel":"fc3.weight",
    "fc3/dense/bias":"fc3.bias",
}
ms_ckpt_path='./model'
tf_ckpt_dir = './model'
tensorflow2mindspore(tf_ckpt_dir, param_mapping_dir, ms_ckpt_path)
```

After execution, you can find the model files that MindSpore can use after conversion in the corresponding path.

## Unit Test

After obtaining the corresponding parameter files, we need to perform a unit test on the entire model to ensure the consistency of the model:

```python
from ms_lenet import mindspore_running
from tf_lenet import tf_running

tf_model_path = './model'
tf_outputs = tf_running(tf_model_path)
ms_outputs = mindspore_running('./tf2mindspore.ckpt')
diff = mean_relative_error(tf_outputs, ms_outputs)
print("************tensorflow outputs**************")
print(tf_outputs)
print("************mindspore outputs**************")
print(ms_outputs)
print("Diff: ", diff)
```

Output:

```text
************tensorflow outputs**************
[[56.040612]
 [56.040612]
 [56.040612]
 [56.040612]
 [56.040612]
 [56.040612]
 [56.04064 ]
 [56.04064 ]]
************mindspore outputs**************
[[56.04065]
 [56.04065]
 [56.04065]
 [56.04065]
 [56.04065]
 [56.04065]
 [56.04065]
 [56.04065]]
Diff:  5.4456143e-07

```

You can see that the final results vary greatly and basically meet the expectations.
