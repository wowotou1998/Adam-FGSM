import os

import torch
from torch import nn
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from tqdm import tqdm
from GeneralTools import ModelSet, AttackMethods
from attack_methods import fgsm_attack, ssi_fgsm_attack, i_fgsm_attack, adam_fgsm_attack
from lab_2_MNIST.CommonOperations import show_many_imgs


def do_fsgm_for_model(model, test_loader, criterion, epsilon):
    test_count = 0
    sample_attacked = 0
    acc_before_attack = 0
    acc_after_fgsm, acc_after_i_fgsm, acc_after_ssi_fgsm_random, \
    acc_after_ssi_fgsm_max_g, acc_after_adam_fgsm = 0, 0, 0, 0, 0
    # every epoch has 64 images ,every images has 1 channel and the channel size is 28*28
    pbar = tqdm(total=len(test_loader))
    for data in test_loader:
        # train_loader is a class, DataSet is a list(length is 2,2 tensors) ,img is a tensor,label is a tensor
        # img is consisted by 64 tensor, so we will get the 64 * 10 matrix. label is a 64*1 matrix, like a vector.
        img, label = data
        _, predict = torch.max(model(img), 1)
        # epoch_correct_before_perturbation_list.append((label == predict).sum().item() / img.shape[0])
        acc_before_attack += (label == predict).sum().item()
        # --------------------
        # fgsm
        img_under_fgsm = fgsm_attack(model, img.clone().detach(), label, criterion, epsilon)
        _, predict = torch.max(model(img_under_fgsm), 1)
        acc_after_fgsm += (label == predict).sum().item()
        # i-fgsm
        img_under_i_fgsm = i_fgsm_attack(model, img.clone().detach(), label,
                                         criterion, epsilon, 10)
        _, predict = torch.max(model(img_under_i_fgsm), 1)
        acc_after_i_fgsm += (label == predict).sum().item()

        # adam-fgsm
        img_under_adam_fgsm = adam_fgsm_attack(model, img.clone().detach(), label,
                                               criterion, epsilon, 20, 0.5, 'random')
        _, predict = torch.max(model(img_under_adam_fgsm), 1)
        acc_after_adam_fgsm += (label == predict).sum().item()

        # # ssi-fgsm-random
        # img_under_ssi_fgsm_random = ssi_fgsm_attack(model, img.clone().detach(), label,
        #                                             criterion, epsilon, 20, 0.5, 'random')
        # _, predict = torch.max(model(img_under_ssi_fgsm_random), 1)
        # acc_after_ssi_fgsm_random += (label == predict).sum().item()

        # # ssi-fgsm-max-gradient
        # img_under_ssi_fgsm_max_g = ssi_fgsm_attack(model, img.clone().detach(), label,
        #                                            criterion, epsilon, 20, 0.5, 'max_gradient')
        # _, predict = torch.max(model(img_under_ssi_fgsm_max_g), 1)
        # acc_after_ssi_fgsm_max_g += (label == predict).sum().item()

        # -----------------
        sample_attacked += label.shape[0]
        pbar.update(label.shape[0])
        test_count += 1
        if test_count == 1:
            show_many_imgs(img, 'clean_img')
            show_many_imgs(img_under_fgsm, 'fgsm_img')
            show_many_imgs(img_under_i_fgsm, 'i-fgsm_img')
            show_many_imgs(img_under_adam_fgsm, 'adam-fgsm_img')
            # show_many_imgs(img_under_ssi_fgsm_random, 'ssi-fgsm_img')
            # break

    pbar.close()

    print(
        'acc_before_attack %.2f%% \n'
        'acc_after_fgsm = %.2f%% \n'
        'acc_after_i_fgsm = %.2f%% \n'
        'acc_after_ssi_fgsm_random = %.2f%% \n'
        'acc_after_ssi_fgsm_max_g = %.2f%% \n'
        'acc_after_adam_fgsm = %.2f%% \n'
        % (
            acc_before_attack / sample_attacked * 100.0,
            acc_after_fgsm / sample_attacked * 100.0,
            acc_after_i_fgsm / sample_attacked * 100.0,
            acc_after_ssi_fgsm_random / sample_attacked * 100.0,
            acc_after_ssi_fgsm_max_g / sample_attacked * 100.0,
            acc_after_adam_fgsm / sample_attacked * 100.0

        ))


if __name__ == '__main__':
    data_tf = transforms.Compose(
        [
            transforms.ToTensor(),
        ]
    )

    # os.path.dirname(path.dirname(__file__))+ '/DataSet/MNIST'
    from os import path

    data_path = os.path.dirname(path.dirname(__file__)) + '/DataSet/MNIST'
    train_dataset = datasets.MNIST(root=data_path, train=True, transform=data_tf, download=True)
    test_dataset = datasets.MNIST(root=data_path, train=False, transform=data_tf)

    # 设置batch_size=64后，加载器中的基本单为是一个batch的数据所以train_loader 的长度是60000/64 = 938 个batch
    batch_size = 1024
    train_loader = DataLoader(dataset=train_dataset, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(dataset=test_dataset, batch_size=batch_size, shuffle=False)

    # 选择模型
    criterion = nn.CrossEntropyLoss()

    model = ModelSet.CNN3_sigmoid()
    model.load_state_dict(torch.load('../SaveModel/MNIST_params.pth', map_location=torch.device('cpu')))
    do_fsgm_for_model(model=model, test_loader=test_loader, criterion=criterion, epsilon=0.2)
    print("all things have been done")
