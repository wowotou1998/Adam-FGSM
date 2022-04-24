import os

import torch
from torch import nn
import torchvision.models as models
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from tqdm import tqdm
from attack_methods import adam_fgsm_attack, fgsm_attack, i_fgsm_attack, mi_fgsm_attack, rmsprop_fgsm_attack
import matplotlib.pyplot as plt

'''
所有的tensor都有.requires_grad属性,可以设置这个属性.

　　　　x = tensor.ones(2,4,requires_grad=True)

2.如果想改变这个属性，就调用tensor.requires_grad_()方法：

　　 x.requires_grad_(False)

3.自动求导注意点:

　　(1)  要想使x支持求导，必须让x为浮点类型;

　　(2) 求导，只能是【标量】对标量，或者【标量】对向量/矩阵求导;

　　(3) 不是标量也可以用backward()函数来求导;

　　(4) 　一般来说，我是对标量求导，比如在神经网络里面，我们的loss会是一个标量，那么我们让loss对神经网络的参数w求导，直接通过loss.backward()即可。

　　　　　　但是，有时候我们可能会有多个输出值，比如loss=[loss1,loss2,loss3]，那么我们可以让loss的各个分量分别对x求导，这个时候就采用：
　　　　　　　　loss.backward(torch.tensor([[1.0,1.0,1.0,1.0]]))

　　　　　　如果你想让不同的分量有不同的权重，那么就赋予gradients不一样的值即可，比如：
　　　　　　　　　　loss.backward(torch.tensor([[0.1,1.0,10.0,0.001]]))

　　　　　　这样，我们使用起来就更加灵活了，虽然也许多数时候，我们都是直接使用.backward()就完事儿了。

　　(5)一个计算图只能backward一次,改善方法:retain_graph=True

　　　　但是这样会吃内存！，尤其是，你在大量迭代进行参数更新的时候，很快就会内存不足，memory out了。

引自:

　　https://www.jianshu.com/p/a105858567df


'''


def show_many_imgs(img, title):
    plt.figure()
    print(img.shape)
    img = img.cpu().detach().numpy()[0].transpose(1, 2, 0)
    # print(img.detach().numpy()[0].shape)
    plt.imshow(img)
    plt.title(title)
    plt.show()


def attack_model(model, test_loader, criterion):
    test_count = 0
    sample_attacked = 0
    acc_before_attack = 0
    acc_after_fgsm, acc_after_i_fgsm, acc_after_mi_fgsm, \
    acc_after_ssi_fgsm_max_g, acc_after_adam_fgsm, acc_after_rmsprop_fgsm = 0, 0, 0, 0, 0, 0
    device = torch.device("cuda:%d" % (0) if torch.cuda.is_available() else "cpu")
    # device = torch.device("cpu")
    # every epoch has 64 images ,every images has 1 channel and the channel size is 28*28
    pbar = tqdm(total=10000)
    model.eval()
    model.to(device)

    epsilon = 1.
    iterations_N = 20
    norm_p_N = 1
    momentum = 0.9

    for data in test_loader:
        # train_loader is a class, DataSet is a list(length is 2,2 tensors) ,img is a tensor,label is a tensor
        # img is consisted by 64 tensor, so we will get the 64 * 10 matrix. label is a 64*1 matrix, like a vector.
        img, label = data
        img = img.to(device)
        img.requires_grad_(True)
        # label.requires_grad_(True)
        label = label.to(device)
        _, predict = torch.max(model(img), 1)
        # 选择预测正确的img和label，剔除预测不正确的img和label
        predict_answer = (label == predict)
        # print('predict_answer', predict_answer)
        # 我们要确保predict correct是一个一维向量,因此使用flatten
        predict_correct_index = torch.flatten(torch.nonzero(predict_answer))
        # print('predict_correct_index', predict_correct_index)
        img = torch.index_select(img, 0, predict_correct_index)
        label = torch.index_select(label, 0, predict_correct_index)

        acc_before_attack += predict_answer.sum().item()

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
                                           criterion, momentum, epsilon, iterations=iterations_N, norm_p=norm_p_N)
        _, predict = torch.max(model(img_under_mi_fgsm), 1)
        acc_after_mi_fgsm += (label == predict).sum().item()
        # rmsprop-fgsm--------------------
        img_under_rmsprop_fgsm = rmsprop_fgsm_attack(model, img.clone().detach(), label,
                                                     criterion, momentum, epsilon, iterations=iterations_N,
                                                     norm_p=norm_p_N)
        _, predict = torch.max(model(img_under_rmsprop_fgsm), 1)
        acc_after_rmsprop_fgsm += (label == predict).sum().item()

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
            print('')
            print('predict_correct_index.nelement: ', predict_correct_index.nelement())
            show_many_imgs(img, 'clean_img')
            # show_many_imgs(img_under_fgsm, 'fgsm_img')
            show_many_imgs(img_under_i_fgsm, 'i-fgsm_img')
            show_many_imgs(img_under_mi_fgsm, 'mi-fgsm_img')
            show_many_imgs(img_under_adam_fgsm, 'adam-fgsm_img')
            show_many_imgs(img_under_rmsprop_fgsm, 'rmsprop-fgsm_img')
        if test_count > 600:
            break
    print(
        'epsilon %.2f\n'
        'iterations_N %d\n'
        'norm_p_N %d \n'
        'momentum %.2f \n'
        'acc_before_attack %.2f%% \n'
        'acc_after_fgsm = %.2f%% \n'
        'acc_after_i_fgsm = %.2f%% \n'
        'acc_after_mi_fgsm = %.2f%% \n'
        'acc_after_adam_fgsm = %.2f%% \n'
        'acc_after_rmsprop_fgsm = %.2f%% \n'

        % (
            epsilon, iterations_N, norm_p_N, momentum,
            acc_before_attack / sample_attacked * 100.0,
            acc_after_fgsm / sample_attacked * 100.0,
            acc_after_i_fgsm / sample_attacked * 100.0,
            acc_after_mi_fgsm / sample_attacked * 100.0,
            acc_after_adam_fgsm / sample_attacked * 100.0,
            acc_after_rmsprop_fgsm / sample_attacked * 100.0,

        ))

    pbar.close()


if __name__ == '__main__':
    from pylab import mpl
    import os

    os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'
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

    train_dataset = datasets.CIFAR10(root='../DataSet/CIFAR10', train=True, transform=data_tf, download=True)
    test_dataset = datasets.CIFAR10(root='../DataSet/CIFAR10', train=False, transform=data_tf)

    # 设置batch_size=64后，加载器中的基本单为是一个batch的数据所以train_loader 的长度是60000/64 = 938 个batch
    batch_size = 700
    train_loader = DataLoader(dataset=train_dataset, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(dataset=test_dataset, batch_size=batch_size, shuffle=False)

    # 选择模型
    criterion = nn.CrossEntropyLoss()
    model = models.vgg16(num_classes=10)
    checkpoint = torch.load('../Checkpoint/%s.pth' % ('VGG16'))
    model.load_state_dict(checkpoint['model'])
    attack_model(model=model, test_loader=test_loader, criterion=criterion)
    print("all things have been done")
