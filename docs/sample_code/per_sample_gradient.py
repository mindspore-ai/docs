"""
This sample code is about Per-sample-gradient computation, which is computing the gradient for each and every
sample in a batch of data. It is a useful in differential privacy, meta-learning, and optimization research.
This sample code is applicable to CPU and GPU.
"""

import os
import time
import argparse
import mindspore.ops as ops
import mindspore.numpy as mnp
import mindspore.dataset as ds
import mindspore.dataset.vision as vision
import mindspore.dataset.transforms as trans
from mindspore import context, Tensor, ms_function, nn, vmap, dtype as mstype
from mindspore.dataset.vision import Inter
from mindspore.common.initializer import TruncatedNormal
from download import download


def parse_args():
    """
    You can execute the program with the following arguments, such as python per_sample_gradient.py --vmap.
    """
    parser = argparse.ArgumentParser(description="MindSpore calculate per-example gradients!")
    parser.add_argument('--vmap', dest='vmap', action='store_true')
    parser.add_argument("--data_dir", default="MNIST_Data/", type=str, help="Where dataset is be stored")
    parser.add_argument("--epochs", default=1, type=int, help="number of total epochs to run")
    parser.add_argument("--lr", default=0.001, type=float, help="the learning rate of model's optimizer")
    parser.add_argument("--momentum", default=0.9, type=float, help="the momentum value of model's optimizer")
    parser.add_argument("--batch_size", default=256, type=int, help="mini-batch size for dataset")
    parser.add_argument("--micro_batches", default=64, type=int,
                        help="the number of small batches split from an original batch")
    parser.add_argument("--norm_bound", default=2.0, type=float,
                        help="the clip bound of the gradients of model's training parameters")
    parser.add_argument("--noise_multiplier", default=0.5, type=float,
                        help="the multiplication coefficient of the noise added to training")
    return parser.parse_args()


# Data Generation
def generate_mnist_dataset(data_path, batch_size=32, repeat_size=1, num_parallel_workers=1, sparse=True):
    """
    create dataset for training or testing
    """
    # Download data from open datasets
    url = "https://mindspore-website.obs.cn-north-4.myhuaweicloud.com/" \
          "notebook/datasets/MNIST_Data.zip"
    download(url, "./", kind="zip", replace=True)

    # define dataset
    ds1 = ds.MnistDataset(data_path)

    # define operation parameters
    resize_height, resize_width = 32, 32
    rescale = 1.0 / 255.0
    shift = 0.0

    # define map operations
    resize_op = vision.Resize((resize_height, resize_width), interpolation=Inter.LINEAR)
    rescale_op = vision.Rescale(rescale, shift)
    hwc2chw_op = vision.HWC2CHW()
    type_cast_op = trans.TypeCast(mstype.int32)

    # apply map operations on images
    if not sparse:
        one_hot_enco = trans.OneHot(10)
        ds1 = ds1.map(input_columns="label", operations=one_hot_enco,
                      num_parallel_workers=num_parallel_workers)
        type_cast_op = trans.TypeCast(mstype.float32)
    ds1 = ds1.map(input_columns="label", operations=type_cast_op,
                  num_parallel_workers=num_parallel_workers)
    ds1 = ds1.map(input_columns="image", operations=resize_op,
                  num_parallel_workers=num_parallel_workers)
    ds1 = ds1.map(input_columns="image", operations=rescale_op,
                  num_parallel_workers=num_parallel_workers)
    ds1 = ds1.map(input_columns="image", operations=hwc2chw_op,
                  num_parallel_workers=num_parallel_workers)

    # apply DatasetOps
    buffer_size = 10000
    ds1 = ds1.shuffle(buffer_size=buffer_size)
    ds1 = ds1.batch(batch_size, drop_remainder=True)
    ds1 = ds1.repeat(repeat_size)

    return ds1


# Model Definition
def weight_variable():
    return TruncatedNormal(0.05)


def conv(in_channels, out_channels, kernel_size, stride=1, padding=0):
    weight = weight_variable()
    return nn.Conv2d(in_channels, out_channels,
                     kernel_size=kernel_size, stride=stride, padding=padding,
                     weight_init=weight, has_bias=False, pad_mode="valid")


def fc_with_initialize(input_channels, out_channels):
    weight = weight_variable()
    bias = weight_variable()
    return nn.Dense(input_channels, out_channels, weight, bias)


class LeNet5(nn.Cell):
    """
    Lenet network
    """
    def __init__(self):
        super(LeNet5, self).__init__()
        self.conv1 = conv(1, 6, 5)
        self.conv2 = conv(6, 16, 5)
        self.fc1 = fc_with_initialize(16 * 5 * 5, 120)
        self.fc2 = fc_with_initialize(120, 84)
        self.fc3 = fc_with_initialize(84, 10)
        self.relu = nn.ReLU()
        self.max_pool2d = nn.MaxPool2d(kernel_size=2, stride=2)
        self.flatten = nn.Flatten()

    def construct(self, x):
        """construct"""
        x = self.conv1(x)
        x = self.relu(x)
        x = self.max_pool2d(x)
        x = self.conv2(x)
        x = self.relu(x)
        x = self.max_pool2d(x)
        x = self.flatten(x)
        x = self.fc1(x)
        x = self.relu(x)
        x = self.fc2(x)
        x = self.relu(x)
        x = self.fc3(x)
        return x


def main():
    context.set_context(mode=context.GRAPH_MODE)
    args = parse_args()
    if args.micro_batches and args.batch_size % args.micro_batches != 0:
        raise ValueError("Number of micro_batches should divide evenly batch_size")
    micro_batches = args.micro_batches
    norm_bound = Tensor(args.norm_bound, mstype.float32)
    noise_multiplier = Tensor(args.noise_multiplier, mstype.float32)

    hype_map_op = ops.HyperMap()

    train_dataset = generate_mnist_dataset(os.path.join(args.data_dir, "train"), args.batch_size)

    print("Model initialization.")
    net = LeNet5()
    weights = net.trainable_params()
    opt = nn.Momentum(weights, learning_rate=args.lr, momentum=args.momentum)
    loss_fn = nn.SoftmaxCrossEntropyWithLogits(sparse=True, reduction="mean")
    model = nn.WithLossCell(net, loss_fn)

    def clip_grad(data, labels):
        # calculate loss and grad
        loss, record_grad = ops.value_and_grad(model, grad_position=None, weights=weights)(data, labels)

        # calculate the norm of the gradient
        square_sum = Tensor(0, mstype.float32)
        for grad in record_grad:
            cur_square_sum = mnp.sum(mnp.square(grad))
            square_sum = mnp.add(square_sum, cur_square_sum)
        cur_norm_grad = mnp.sqrt(square_sum)

        # clip grad
        clip_grads = ()
        cur_norm_grad = mnp.where((cur_norm_grad <= norm_bound), x=norm_bound, y=cur_norm_grad)
        for grad in record_grad:
            clipped_grad = grad * (norm_bound / cur_norm_grad)
            clip_grads = clip_grads + (clipped_grad,)
        return clip_grads, loss

    def add_noise(grads, seed=0):
        mean = Tensor(0, mstype.float32)
        stddev = norm_bound * noise_multiplier

        grad_noise_tuple = ()
        for grad_item in grads:
            shape = ops.shape(grad_item)
            noise = ops.normal(shape, mean, stddev, seed)
            grad_noise_tuple = grad_noise_tuple + (noise,)

        grads = hype_map_op(mnp.add, grads, grad_noise_tuple)
        return grads

    @ms_function
    def private_grad_with_forloop(data, labels):
        record_datas = mnp.split(data, micro_batches)
        record_labels = mnp.split(labels, micro_batches)

        # step 1: calculate per sample gradients with forloop
        grads, total_loss = clip_grad(record_datas[0], record_labels[0])

        for i in range(1, micro_batches):
            grad, loss = clip_grad(record_datas[i], record_labels[i])
            grads = hype_map_op(mnp.add, grads, grad)
            total_loss = total_loss + loss
        loss = total_loss / micro_batches

        # step 2: add gaussian noise
        noise_grads = add_noise(grads)

        # step 3: update param
        loss = ops.depend(loss, opt(noise_grads))
        return loss

    @ms_function
    def private_grad_with_vmap(data, labels):
        batch_datas = ops.reshape(data, (micro_batches, -1,) + data.shape[1:])
        batch_labels = ops.reshape(labels, (micro_batches, -1,) + labels.shape[1:])

        # step 1: calculate per-sample gradients with vmap
        batch_grads, batch_loss = vmap(clip_grad)(batch_datas, batch_labels)
        grads = hype_map_op(mnp.sum, batch_grads, (0,) * len(batch_grads))
        loss = mnp.sum(batch_loss) / micro_batches

        # step 2: add gaussian noise
        noise_grads = add_noise(grads)

        # step 3: update param
        loss = ops.depend(loss, opt(noise_grads))
        return loss

    if args.vmap:
        train_net = private_grad_with_vmap
    else:
        train_net = private_grad_with_forloop

    steps = train_dataset.get_dataset_size()
    train_begin_time = time.time()
    for epoch in range(args.epochs):
        step = 0
        for d in train_dataset.create_dict_iterator():
            step_begin_time = time.time()
            result = train_net(d["image"], d["label"])
            step_time = time.time() - step_begin_time
            print(f"Epoch: [{epoch} / {args.epochs}], " f"step: [{step} / {steps}], "
                  f"loss: {result}, " f"step time: {step_time}")
            step = step + 1
    train_time = time.time() - train_begin_time
    print(f"Total time: {train_time} ms.")

if __name__ == "__main__":
    main()
