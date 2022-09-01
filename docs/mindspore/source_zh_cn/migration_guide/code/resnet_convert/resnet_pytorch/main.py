"""PyTorch training"""
import torch
import torch.nn.functional as F
import torchvision
import torchvision.transforms as trans

from resnet import resnet50


def train_epoch(epoch, model, loss_fun, device, data_loader, optimizer):
    """Single train one epoch"""
    model.train()
    for batch_idx, (data, target) in enumerate(data_loader):
        optimizer.zero_grad()
        output = model(data.to(device))
        loss = loss_fun(output, target.to(device))
        loss.backward()
        optimizer.step()
        if batch_idx % 100 == 0:
            print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                epoch, batch_idx, len(data_loader),
                100. * batch_idx / len(data_loader), loss.item()))


def test_epoch(model, device, data_loader):
    """Single evaluation once"""
    model.eval()
    test_loss = 0
    correct = 0
    with torch.no_grad():
        for data, target in data_loader:
            output = model(data.to(device))
            test_loss += F.nll_loss(output, target.to(device), reduction='sum').item() # sum up batch loss
            pred = output.max(1)[1] # get the index of the max log-probability
            correct += pred.eq(target.to(device)).sum().item()

    test_loss /= len(data_loader.dataset)
    print('\nTest set: Average loss: {:.4f}, Accuracy: {:.0f}%\n'.format(
        test_loss, 100. * correct / len(data_loader.dataset)))


if __name__ == '__main__':
    use_cuda = torch.cuda.is_available()
    device_pt = torch.device("cuda" if use_cuda else "cpu")

    train_transform = trans.Compose([
        trans.RandomCrop(32, padding=4),
        trans.RandomHorizontalFlip(0.5),
        trans.Resize(224),
        trans.ToTensor(),
        trans.Normalize([0.4914, 0.4822, 0.4465], [0.2023, 0.1994, 0.2010]),
    ])
    test_transform = trans.Compose([
        trans.Resize(224),
        trans.RandomHorizontalFlip(0.5),
        trans.ToTensor(),
        trans.Normalize([0.4914, 0.4822, 0.4465], [0.2023, 0.1994, 0.2010]),
    ])
    train_set = torchvision.datasets.CIFAR10(root='./data', train=True, transform=train_transform)
    train_loader = torch.utils.data.DataLoader(train_set, batch_size=32, shuffle=True)
    test_set = torchvision.datasets.CIFAR10(root='./data', train=False, transform=test_transform)
    test_loader = torch.utils.data.DataLoader(test_set, batch_size=1, shuffle=False)
    # 2. define forward network
    net = resnet50(num_classes=10).cuda() if use_cuda else resnet50(num_classes=10)
    # 3. define loss
    net_loss = torch.nn.CrossEntropyLoss()
    # 4. define optimizer
    net_opt = torch.optim.Adam(net.parameters(), 0.001, weight_decay=1e-5)
    for i in range(90):
        train_epoch(i, net, net_loss, device_pt, train_loader, net_opt)
        test_epoch(net, device_pt, test_loader)

    print('Finished Training')
    save_path = './resnet.pth'
    torch.save(net.state_dict(), save_path)
