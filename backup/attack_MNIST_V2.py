import os

import torch
from torch import nn
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from tqdm import tqdm
from .attack_methods import adam_fgsm_attack, fgsm_attack, i_fgsm_attack, mi_fgsm_attack
from .utils import show_img

def do_fsgm_for_model(model, test_loader, criterion, epsilon):
    test_count = 0
    sample_attacked = 0
    acc_before_attack = 0
    acc_after_fgsm, acc_after_i_fgsm, acc_after_ssi_fgsm_random, \
    acc_after_ssi_fgsm_max_g, acc_after_adam_fgsm, acc_after_mi_fgsm = 0, 0, 0, 0, 0, 0
    # every epoch has 64 images ,every images has 1 channel and the channel size is 28*28
    pbar = tqdm(total=len(test_loader))
    for data in test_loader:
        # train_loader is a class, DataSet is a list(length is 2,2 tensors) ,img is a tensor,label is a tensor
        # img is consisted by 64 tensor, so we will get the 64 * 10 matrix. label is a 64*1 matrix, like a vector.
        img, label = data
        _, predict = torch.max(model(img), 1)
        acc_before_attack += (label == predict).sum().item()
        iterations_N = 20
        norm_p_N = 2
        # fgsm--------------------
        img_under_fgsm = fgsm_attack(model, img.clone().detach(), label, criterion, epsilon, norm_p=norm_p_N)
        _, predict = torch.max(model(img_under_fgsm), 1)
        acc_after_fgsm += (label == predict).sum().item()
        # i-fgsm--------------------
        img_under_i_fgsm = i_fgsm_attack(model, img.clone().detach(), label,
                                         criterion, epsilon, iterations=iterations_N, norm_p=norm_p_N)
        _, predict = torch.max(model(img_under_i_fgsm), 1)
        acc_after_i_fgsm += (label == predict).sum().item()
        # mi-fgsm--------------------
        img_under_mi_fgsm = mi_fgsm_attack(model, img.clone().detach(), label,
                                           criterion, epsilon, iterations=iterations_N, norm_p=norm_p_N)
        _, predict = torch.max(model(img_under_mi_fgsm), 1)
        acc_after_mi_fgsm += (label == predict).sum().item()

        # adam-fgsm--------------------
        img_under_adam_fgsm = adam_fgsm_attack(model, img.clone().detach(), label,
                                               criterion, epsilon, iterations=iterations_N, norm_p=norm_p_N)
        _, predict = torch.max(model(img_under_adam_fgsm), 1)
        acc_after_adam_fgsm += (label == predict).sum().item()

        # -----------------
        sample_attacked += label.shape[0]
        pbar.update(label.shape[0])
        test_count += 1
        if test_count == 1:
            show_many_imgs(img, 'clean_img')
            show_many_imgs(img_under_fgsm, 'fgsm_img')
            show_many_imgs(img_under_i_fgsm, 'i-fgsm_img')
            show_many_imgs(img_under_mi_fgsm, 'mi-fgsm_img')
            show_many_imgs(img_under_adam_fgsm, 'adam-fgsm_img')
            # break

    pbar.close()

    print(
        'acc_before_attack %.2f%% \n'
        'acc_after_fgsm = %.2f%% \n'
        'acc_after_i_fgsm = %.2f%% \n'
        'acc_after_mi_fgsm = %.2f%% \n'
        'acc_after_adam_fgsm = %.2f%% \n'

        % (
            acc_before_attack / sample_attacked * 100.0,
            acc_after_fgsm / sample_attacked * 100.0,
            acc_after_i_fgsm / sample_attacked * 100.0,
            acc_after_mi_fgsm / sample_attacked * 100.0,
            acc_after_adam_fgsm / sample_attacked * 100.0,

        ))


if __name__ == '__main__':
    from pylab import mpl

    # 指定默认字体
    mpl.rcParams['font.sans-serif'] = ['Arial']
    # 解决保存图像是负号'-'显示为方块的问题
    mpl.rcParams['axes.unicode_minus'] = False
    mpl.rcParams['savefig.dpi'] = 200  # 保存图片分辨率
    mpl.rcParams['figure.dpi'] = 200  # 分辨率
    mpl.rcParams["figure.subplot.left"], mpl.rcParams["figure.subplot.right"] = 0.0, 1.0
    mpl.rcParams["figure.subplot.bottom"], mpl.rcParams["figure.subplot.top"] = 0.0, 0.95
    mpl.rcParams["figure.subplot.wspace"], mpl.rcParams["figure.subplot.hspace"] = 0.1, 0.1
    data_tf = transforms.Compose(
        [
            transforms.ToTensor(),
        ]
    )

    train_dataset = datasets.MNIST(root='../DataSet/MNIST', train=True, transform=data_tf, download=True)
    test_dataset = datasets.MNIST(root='../DataSet/MNIST', train=False, transform=data_tf)

    # 设置batch_size=64后，加载器中的基本单为是一个batch的数据所以train_loader 的长度是60000/64 = 938 个batch
    batch_size = 1024
    train_loader = DataLoader(dataset=train_dataset, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(dataset=test_dataset, batch_size=batch_size, shuffle=False)

    # 选择模型
    criterion = nn.CrossEntropyLoss()
    model = ModelSet.CNN3_sigmoid()
    model.load_state_dict(torch.load('../Checkpoint/MNIST_params.pth', map_location=torch.device('cpu')))
    do_fsgm_for_model(model=model, test_loader=test_loader, criterion=criterion, epsilon=5)
    print("all things have been done")
