from model.cnn import CNN4, CNN4_MNIST 
from model.resnet import ResNet10
from model.bat import bat_replace_modules


def get_model(model, dataset):
    if "mnist" in dataset:
        net = CNN4_MNIST()
    elif "svhn" in dataset:
        net = CNN4()
    elif "cifar10" in dataset and "cifar100" not in dataset :
        net = ResNet10()
    elif "cifar100" in dataset:
        net = ResNet10(100)

    if "bat" in model:
        bat_replace_modules(net)
    return net


if __name__ == "__main__":
    print("done")
