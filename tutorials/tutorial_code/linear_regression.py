import numpy as np
import mindspore as ms
from mindspore.ops import composite as C
from mindspore.ops import operations as P
from mindspore import Tensor
from mindspore import context
from mindspore.common.initializer import TruncatedNormal
from mindspore import nn
 
context.set_context(mode=context.PYNATIVE_MODE, device_target="GPU")

# Generating training data sets
def get_data(num, w=2.0, b=3.0):
    np_x = np.ones([num, 1])
    np_y = np.ones([num, 1])
    for i in range(num):
        x  = np.random.uniform(-10.0, 10.0)
        np_x[i] = x
        noise = np.random.normal(0, 1)
        y  = x * w + b + noise
        np_y[i]=y
    return Tensor(np_x,ms.float32), Tensor(np_y,ms.float32)

# Define the form of loss function: 1/2 * (y - y')^2
class MyLoss(nn.loss.loss._Loss):
    def __init__(self, reduction='mean'):
        super().__init__(reduction)
        self.square = P.Square()
    def construct(self, data, label):
        x = self.square(data - label) * 0.5
        return self.get_loss(x)

# Gradient function
class GradWrap(nn.Cell):
    """ GradWrap definition """
    def __init__(self, network):
        super().__init__(auto_prefix=False)
        self.network = network
        self.weights = ms.ParameterTuple(filter(lambda x: x.requires_grad,
            network.get_parameters()))

    def construct(self, data, label):
        weights = self.weights
        return C.GradOperation('get_by_list', get_by_list=True) \
            (self.network, weights)(data, label)

# Initializing model functions
net = nn.Dense(1, 1, TruncatedNormal(0.02), TruncatedNormal(0.02))

# Loss function
criterion = MyLoss()
loss_opeartion = nn.WithLossCell(net, criterion)
train_network = GradWrap(loss_opeartion) 
train_network.set_train()

# Defining optimization
optim = nn.RMSProp(params=net.trainable_params(), learning_rate=0.02)

# Executive Training
step_size = 200
batch_size = 16
for i in range(step_size):
    data_x, data_y = get_data(batch_size)
    grads = train_network(data_x, data_y) 
    optim(grads)   

    # Print loss value per 10 step
    if i%10 == 0:
        output = net(data_x)
        loss_output = criterion(output, data_y)
        print(loss_output.asnumpy())

# Print final weight parameters
print("weight:", net.weight.default_input[0][0], "bias:", net.bias.default_input[0])