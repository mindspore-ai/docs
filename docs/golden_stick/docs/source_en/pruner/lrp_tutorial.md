# LRP Deployment tutorial

[![View Source On Gitee](https://mindspore-website.obs.cn-north-4.myhuaweicloud.com/website-images/master/resource/_static/logo_source_en.svg)](https://gitee.com/mindspore/docs/blob/master/docs/golden_stick/docs/source_en/pruner/lrp_tutorial.md)

## Introduction

The LRP algorithm performs head pruning by:

1. Adding gate modules to a model
2. Fine-tuning the model where a penalty term is added to the original loss term
3. At the end of the training, hardening the gates or pruning the heads (according to the learned values of the gates)

To support this scheme, we define two new classes and insert few additions to the original model.

## Provided Classes

We introduce two main new classes. The first class modules the gates and is planted into the model itself. The second is in fact a main class that uses several other classes to prune heads of a gated fine-tuned model, according to the gates values.

### Concrete-Gate Class

***The Concrete–Gate class is a class that we provide and needs no adjustments when the model to be pruned is replaced. The information below is provided only for the purpose of understanding the changes and additions to be made elsewhere in the code.***

Conceptually a gate is a simple scalar that determines whether to let an input keep flowing in the neural network, or not. Yet, the list of features we want the gate to have (e.g., being distinctly either closed or open during evaluation mode, being differentiable during training mode, encouraged to be zero during training), requires building a whole module around these scalars. The name of the gate module class is “ConcreteGate”,

A single Concrete-Gate class is in charge of all the gates within a single Attention mechanism. Besides the initialization method, it has the following three methods:

* get_gates
    * Returns the gates’ scalar values, given the operation mode (training/evaluation)
* construct
    * Given an input and, optionally, the operating mode, apply the gates on the input (using the method get_gates)
* get_penalty
    * calculating a penalty term to be added to the final loss

ConcreteGate code from [utils.py](https://gitee.com/mindspore/golden-stick/blob/master/mindspore_gs/pruner/heads/lrp/utils.py) file as follows.

```python
 class ConcreteGate(nn.Cell):
    def __init__(self, shape, temperature=0.33, stretch_limits=(-0.1, 1.1),
                 l0_penalty=1.0, eps=1e-6):
        super(ConcreteGate, self).__init__()
        self.temperature, self.stretch_limits, self.eps = temperature, stretch_limits, eps
        self.l0_penalty = l0_penalty
        self.log_a = Parameter(initializer('xavier_uniform', shape), name="log_a")
        self.sigmoid = ops.Sigmoid()
        self.log = ops.Log()
        self.op = ops.ReduceSum()

    def construct(self, values, is_train=True):
        is_train = self.training if is_train is None else is_train
        gates = self.get_gates(is_train)
        return values * gates

    def get_gates(self, is_train):
        low, high = self.stretch_limits

        if is_train:
            shape = self.log_a.shape
            uniformReal = ops.UniformReal()
            noise = (1 - 2 * self.eps) * uniformReal(shape) + self.eps
            concrete = self.sigmoid((self.log(noise) - self.log(1 - noise) + self.log_a) / self.temperature)
        else:
            concrete = self.sigmoid(self.log_a)

        stretched_concrete = concrete * (high - low) + low
        clipped_concrete = ops.clip_by_value(stretched_concrete, 0, 1)
        return clipped_concrete

    def get_penalty(self):
        low, high = self.stretch_limits
        assert low < 0.0, "p_gate_closed can be computed only if lower stretch limit is negative"
        # compute p(gate_is_closed) = cdf(stretched_sigmoid < 0)
        p_open = self.sigmoid(self.log_a - self.temperature * self.log(Tensor(-low) / Tensor(high)))
        p_open = ops.clip_by_value(p_open, self.eps, 1.0 - self.eps)

        total_reg = self.l0_penalty * self.op(p_open)
        return total_reg
 ```

### Pruner class(es)

***Unlike the Concrete-Gate class, which is agnostic to the model, the pruner class is defined slightly different for each model.***

The pruning is executed as follows:

1. A model to prune and pruning configuration are received
2. A gated version of the model (that is a model that includes and supports gates) is generated, and the variables values of the original model are copied to the gated model
3. The gated model is trained
4. The variables values of the gated model are copied to the original model, and some heads of the original model are pruned according to the values of the trained gates.

The pruner class purpose is to supports these actions. These actions require reaching different blocks in the model, and since different models concatenating blocks differently and named them differently, we need to write different pruner class for each model.

A pruner class inherits the class “AbstructHeadPrunerLRP”, which is a class that we provide. Besides the initialization method (that is performed by the “AbstractHeadPrunerLRP”), the pruner has three more methods that are written differently depending on the model one wants to prune:

* init_head
    * The pruner is initialized with a model to prune and stored it in the attribute “original_model”. The method init_head rearrange the attribute “original_model” and adds a “head” attribute, such that “head” contains the full model and “original_model” contains only the backbone part. Additionally, the method set the value of the Boolean attribute “has_head” to be True if the received model is a full model, and False if the received model is only the backbone part - both cases are supported.
* decorate_model
    * The method creates and returns the gated version of the model (with weights copied from the received model).
* get_penalty
    * The method receives a trained gated model and use it to correct and output the model stored in the class. The correction is done by coping gated model variables values, and pruning heads that are needed to be pruned.

We provide two examples of this class - “HeadPrunerBertLRP” that is designed to prune Bert model, and “HeadPrunerGPTLRP” that is designed to prune GPT model.

## Additions to the Original Model

As implied by the description of the pruner class, we need to form a class that is the gated version of the model to prune class. This is done by taking the classes that compose the original model, and insert some additions.

To ease the review of the added code lines to the original model, we divide the additions into two groups – additions related to the normal flow of the data in the model, and additions that enables controlling and changing the model.

### Data Flow Within the Model

The abilities we want to the add to the original model are:

* Applying gates and calculate contribution to the penalty term at the Attention level
* Transferring and accumulating the penalty terms from the Attention level to the level where the loss is calculated, through the concatenated and nested classes that are the building blocks of the model.
* Adding the accumulated penalty term to the loss

To enable these abilities, we distinguish between three types of classes – the class that calculate the Attention, the class that calculate the loss, and classes that connect between these two above-mentioned classes

#### Attention Level Class

We introduce two attributes to the class - a “ConcreteGate” class type attribute that is named “gate”, and a Boolean type attribute named “has_gate”. The first is obviously the added gate, and the second is an indicator of whether to address or ignore the gate. These two attributes are defined in the __init__ method of the class

Code example from [gated_bert_model.py](https://gitee.com/mindspore/golden-stick/blob/master/mindspore_gs/pruner/heads/lrp/bert/archs/gated_bert_model.py):

```python
BertAttention.__init__()
    self.has_gate = False
    '''Creation of the gate calculation object'''
    self.gate = ConcreteGate(shape=[1, self.num_attention_heads, 1, 1])
```

Another example is given for GPT model in [gated_transformer.py](https://gitee.com/mindspore/golden-stick/blob/master/mindspore_gs/pruner/heads/lrp/gpt/archs/gated_transformer.py):

```python
MultiHeadAttention.__init__()
    '''Creation of the gate calculation object'''
    self.gate = ConcreteGate(shape=[1, self.n_head, 1, 1])
    self.has_gates = False
```

While performing the forward path of the class, if the gates are enabled, we apply the gates and calculate the contribution of the Attention to the penalty term. These two actions are performed just before we multiply the Attention probabilities by the “values vectors” of the Attention mechanism

Code example from BertAttention forward function in [gated_bert_model.py](https://gitee.com/mindspore/golden-stick/blob/master/mindspore_gs/pruner/heads/lrp/bert/archs/gated_bert_model.py):

```python
BertAttention.construct()
    .
    .
    reg = 0.0
    if self.has_gates:
        attention_probs = self.gate(attention_probs)
        reg = self.gate.get_penalty()
    .
    .
```

Finally, we need to output the calculated the penalty term so it could climb up to the loss level

Code example from BertAttention forward function in [gated_bert_model.py](https://gitee.com/mindspore/golden-stick/blob/master/mindspore_gs/pruner/heads/lrp/bert/archs/gated_bert_model.py):

```python
BertAttention.construct()
    .
    .
    if self.has_gates:
        outputs += (reg,)
    .
    .
```

## Intermediate Level Classes

Intermediate level class are responsible for transferring and accumulating penalty terms. We distinguish between two types of such classes – classes that receive a single penalty term and only need to transfer it, and classes that receive more than one penalty term and need to accumulate them into a single term before the transfer.

Generally, in the forward path each class –

1. Receives a package of output variables from the classes to which it calls
2. Extract from the package variable that it needs to perform its forward path
3. Update and add variables to the package
4. Output the package

Thus, a class that only transfers the penalty onward needs no additional code lines. On the other hand, class that accumulate the penalties needs additional code lines that execute actions 2 and 3, if the gated are enabled.

Overall, the second type classes need an attribute “has_gate”, similar to the Attention level class, and some additions to the forward path

Code example from BertTransformer in [gated_bert_model.py](https://gitee.com/mindspore/golden-stick/blob/master/mindspore_gs/pruner/heads/lrp/bert/archs/gated_bert_model.py):

```python
BertTransformer.__init__()
    .
    .
    selg.has_gates = False
    .
    .

BertTransformer.construct()
    .
    .
    total_reg = 0.0
    for layer_module in self.layers:
        if self.has_gates
            total_reg += layer_output[1]
    .
    .
    return all_encoder_layers, total_reg
```

### Loss Level Class

The total_reg that comes back from the model is added to Loss calculation.

From Bert code example in [bert_pretrain_gates_sample.py](https://gitee.com/mindspore/golden-stick/blob/master/mindspore_gs/pruner/heads/lrp/bert/samples/bert_pretrain_gates_sample.py)

```python
BertPreTrainingForGates.construct()
    .
    .
    _, pooled_output, _, total_reg = self.bert(input_ids, token_type_id, input_mask)
    .
    .
    if self.has_gates:
        loss += total_reg
```

### Model Control

The pruning of the model is done according to the values of the trained gates. The values of the gates are stored at the Attention level, and to get them we need to transfer this request from the external class of the gated model down to the attention class. This is done using the functions get_gate_values
Another two functions that transfer instruction from the external class of the model down to its components are “apply_gates” and “remove_gates”, that enable and disable the gate using the attributes “has_gate” in the internal classes.

Code additions to BertAttention as can be seen in [gated_bert_model.py](https://gitee.com/mindspore/golden-stick/blob/master/mindspore_gs/pruner/heads/lrp/bert/archs/gated_bert_model.py):

```text
class BertAttention
    .
    .
    def get_gate_value (self):
        gate_values = None
        if self.gate is not None:
            gate_values = self.gate.get_gates(False).flatten()
        return gate_values
    def apply_gates(self, l0_penalty):
        if not self.has_gates:
            self.has_gates = True
            self.gate.l0_penalty = l0_penalty
    def remove_gates(self):
        self.has_gates = False
```
