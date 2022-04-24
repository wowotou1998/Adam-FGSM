'''Train CIFAR10 with PyTorch.'''
from __future__ import print_function

import os

import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
import torch
from utils import training, save_model_accuracy_rate
from MNIST_models import lenet5, FC_256_128


def get_train_data(data_set_name, batch_size):
    transform_train = transforms.Compose([
        # transforms.RandomCrop(32, padding=4),
        # transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
    ])

    transform_test = transforms.Compose([
        transforms.ToTensor(),
    ])

    train_set, test_set = None, None
    if data_set_name == 'CIFAR10':
        train_set = torchvision.datasets.CIFAR10(root='../DataSet/' + data_set_name, train=True, download=True,
                                                 transform=transform_train)
        test_set = torchvision.datasets.CIFAR10(root='../DataSet/' + data_set_name, train=False, download=True,
                                                transform=transform_test)
    if data_set_name == 'MNIST':
        train_set = torchvision.datasets.MNIST(root='../DataSet/' + data_set_name, train=True, download=True,
                                               transform=transform_train)
        test_set = torchvision.datasets.MNIST(root='../DataSet/' + data_set_name, train=False, download=True,
                                              transform=transform_test)
    train_loader = torch.utils.data.DataLoader(train_set, batch_size=batch_size, shuffle=True, )
    test_loader = torch.utils.data.DataLoader(test_set, batch_size=batch_size, shuffle=False, )
    return [train_loader, test_loader]


if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser(description='training arguments with PyTorch')
    parser.add_argument('--gpu_id', default=0, type=int, help='The GPU id.')
    parser.add_argument('--batch_size', default=128, type=int, help='The batch_size.')
    parser.add_argument('--epochs', default=50, type=int, help='The epochs.')
    parser.add_argument('--lr', default=1e-3, type=float, help='The learning rate.')
    parser.add_argument('--load_model_args', default=True, type=bool, help='Load_model_args.')
    parser.add_argument('--model', default='VGG16', type=bool, help='model.')
    parser.add_argument('--dataset', default='CIFAR10', type=bool, help='dataset.')
    args = parser.parse_args()
    # preparing data
    print('--> Preparing data..')
    batch_size = args.batch_size
    # from os import path
    # data_path = os.path.dirname(path.dirname(__file__)) + '/DataSet/CIFAR10'
    # print('train_set_path',data_path)

    classes = ('plane', 'car', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck')
    import torchvision.models as models

    # VGG16 ResNet50 DenseNet161 InceptionV3,
    LeNet5 = lenet5()
    FC_256_128 = FC_256_128()
    VGG16 = models.vgg16(num_classes=10)
    VGG19 = models.vgg19(num_classes=10)
    ResNet50 = models.resnet50(num_classes=10)
    ResNet101 = models.resnet101(num_classes=10)
    DenseNet121 = models.densenet121(num_classes=10)

    print('--> Building model..')
    model_set = [FC_256_128]
    model_names = ['FC_256_128']
    model_accuracy_rate = {}
    data_set = [get_train_data('MNIST', batch_size)]
    # lr =1e-3 100
    # lr =1e-4 200
    # first train
    for lr in [1e-3]:
        for data in data_set:
            for i, model_i in enumerate(model_set):
                train_loader = data[0]
                test_loader = data[1]
                model = model_i
                # model = torch.nn.DataParallel(model, device_ids=range(torch.cuda.device_count()))
                # cudnn.benchmark = True
                start_epoch = 0  # start from epoch 0 or last checkpoint epoch
                model_data, best_test_acc = training(model=model,
                                                     train_data_loader=train_loader,
                                                     test_data_loader=test_loader,
                                                     epochs=args.epochs,
                                                     criterion=None,
                                                     optimizer=optim.SGD(model.parameters(),
                                                                         lr=0.1,
                                                                         momentum=0.9,
                                                                         weight_decay=5e-4),
                                                     enable_cuda=True,
                                                     gpu_id=args.gpu_id,
                                                     load_model_args=args.load_model_args,
                                                     model_name=model_names[i])

                # show_model_performance(model_data)
                model_accuracy_rate[model_names[i]] = best_test_acc
                # torch.cuda.empty_cache()
                print('save model acc')
                save_model_accuracy_rate(model_accuracy_rate, model_name=model_names[i])
    # second train
    print('training is over!')
