"""Evaluation"""
import numpy as np
import mindspore as ms
from mindspore import nn
from src.dataset import create_dataset
from src.model_utils.moxing_adapter import moxing_wrapper
from src.model_utils.config import config
from src.utils import init_env
from src.resnet import resnet50


def test_epoch(model, data_loader, loss_func):
    """Evaluation once"""
    model.set_train(False)
    test_loss = 0
    correct = 0
    for data, target in data_loader:
        output = model(data)
        test_loss += float(loss_func(output, target).asnumpy())
        pred = np.argmax(output.asnumpy(), axis=1)
        correct += (pred == target.asnumpy()).sum()
    dataset_size = data_loader.get_dataset_size()
    test_loss /= dataset_size
    print('\nTest set: Average loss: {:.4f}, Accuracy: {:.0f}%\n'.format(
        test_loss, 100. * correct / dataset_size))


@moxing_wrapper()
def test_net():
    init_env(config)
    eval_dataset = create_dataset(config.dataset_name, config.data_path, False, batch_size=1,
                                  image_size=(int(config.image_height), int(config.image_width)))
    resnet = resnet50(num_classes=config.class_num)
    ms.load_checkpoint(config.checkpoint_path, resnet)
    loss = nn.SoftmaxCrossEntropyWithLogits(sparse=True, reduction='mean')
    test_epoch(resnet, eval_dataset, loss)


if __name__ == '__main__':
    test_net()
