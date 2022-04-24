import torch
import torchvision.models as models
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from tqdm import tqdm
from torchattacks import FGSM  # atk_1
from torchattacks import BIM as I_FGSM  # atk_2
from torchattacks import PGD  # atk_3
from torchattacks import MIFGSM as MI_FGSM  # atk_4
from attack_method_self_defined import Adam_FGSM  # atk_5
from attack_models_on_datasets import load_dataset, load_model_args
import matplotlib.pyplot as plt
import numpy
from pylab import mpl
import torch.nn.functional as F
from ImageNet_labels import imagenet_labels
import datetime


# 相似性范围从-1到1：
# -1意味着两个向量指向的方向正好截然相反，1表示它们的指向是完全相同的，
# 0通常表示它们之间是独立的，而在这之间的值则表示中间的相似性或相异性。
def get_label(dataset, idx):
    if dataset == 'MNIST':
        return str(idx)
    elif dataset == 'CIFAR10':
        cifar10_labels = ('plane', 'car', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck')
        return cifar10_labels[idx]
    elif dataset == 'ImageNet':
        return imagenet_labels[str(idx)][0]
    else:
        raise Exception('unknown index')


def show_one_image(images, title_set):
    # plt.rcParams['font.sans-serif'] = [u'SimHei']
    mpl.rcParams['font.sans-serif'] = ['Times New Roman']
    # 解决保存图像是负号'-'显示为方块的问题
    mpl.rcParams['axes.unicode_minus'] = False
    mpl.rcParams['savefig.dpi'] = 200  # 保存图片分辨率
    mpl.rcParams['figure.dpi'] = 200  # 分辨率
    mpl.rcParams["figure.subplot.left"], mpl.rcParams["figure.subplot.right"] = 0.002, 0.998
    mpl.rcParams["figure.subplot.bottom"], mpl.rcParams["figure.subplot.top"] = 0.004, 0.824
    mpl.rcParams["figure.subplot.wspace"], mpl.rcParams["figure.subplot.hspace"] = 0.05, 0.0
    plt.rcParams['xtick.direction'] = 'in'  # 将x周的刻度线方向设置向内
    plt.rcParams['ytick.direction'] = 'in'  # 将y轴的刻度方向设置向内
    # 宽和高
    plt.figure(figsize=(len(images) * 1.5, 1.6))
    # C*H*W-->H*W*C
    for idx, image in enumerate(images):
        b = image.shape[0]
        image = numpy.transpose(image[0].cpu().detach().numpy(), (1, 2, 0))
        plt.subplot(1, len(images), idx + 1)
        if b == 1:
            plt.imshow(image, 'gray')
        else:
            plt.imshow(image)
        plt.axis('off')  # 关掉坐标轴为 off
        plt.title(title_set[idx], )  # 图像题目

    fig = plt.gcf()
    fig.savefig('../Checkpoint/%s_%s.png' % (title_set[0], datetime.datetime.now().strftime("%Y_%m_%d_%H_%M_%S")))
    plt.show()
    fig.savefig('../Checkpoint/%s_%s.pdf' % (title_set[0], datetime.datetime.now().strftime("%Y_%m_%d_%H_%M_%S")))
    print('result has been saved')


def get_hard_to_classify(mode_name, dataset, Epsilon, Iterations, Momentum):
    test_loader, test_loader_size = load_dataset(dataset, batch_size=1, is_shuffle=True)

    model, model_acc = load_model_args(mode_name)
    device = torch.device("cuda:%d" % (0) if torch.cuda.is_available() else "cpu")
    model.to(device)
    model.eval()

    # label_id_set = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]
    # label_name_set = ('0', '1', '2', '3', '4', '5', '6', '7', '8', '9')

    test_count = 0
    bar = tqdm(total=test_loader_size)
    print('len(test_loader)', test_loader_size)
    for data in test_loader:

        # train_loader is a class, DataSet is a list(length is 2,2 tensors) ,images is a tensor,labels is a tensor
        # images is consisted by 64 tensor, so we will get the 64 * 10 matrix. labels is a 64*1 matrix, like a vector.
        original_images, original_labels = data
        original_images = original_images.to(device)
        original_labels = original_labels.to(device)
        _, predict = torch.max(F.softmax(model(original_images), dim=1), 1)
        # 选择预测正确的original_images和original_labels，剔除预测不正确的original_images和original_labels
        # predict_answer为一维向量，大小为batch_size
        predict_answer = (original_labels == predict)
        # torch.nonzero会返回一个二维矩阵，大小为（nozero的个数）*（1）
        no_zero_predict_answer = torch.nonzero(predict_answer)
        # 我们要确保 predict_correct_index 是一个一维向量,因此使用flatten,其中的元素内容为下标
        predict_correct_index = torch.flatten(no_zero_predict_answer)
        # print('predict_correct_index', predict_correct_index)
        images = torch.index_select(original_images, 0, predict_correct_index)
        labels = torch.index_select(original_labels, 0, predict_correct_index)
        if labels.shape[0] == 0:
            bar.update(labels.shape[0])
            continue
        # 查看本次图片的label 是不是我们想要的.
        # if labels[0].item() not in label_id_set:
        #     bar.update(labels.shape[0])
        #     continue
        # ------------------------------
        atk_1 = FGSM(model, eps=Epsilon)
        atk_2 = I_FGSM(model, eps=Epsilon, alpha=Epsilon / Iterations, steps=Iterations)
        atk_3 = PGD(model, eps=Epsilon, alpha=Epsilon / Iterations, steps=Iterations)
        atk_4 = MI_FGSM(model, eps=Epsilon, steps=Iterations, decay=Momentum)
        atk_5 = Adam_FGSM(model, eps=Epsilon, steps=Iterations, decay=Momentum)
        # -----------------------------

        # 选择FGSM攻击不成功的样本
        images_under_attack_1 = atk_1(images, labels)
        p_1, p_idx_1 = torch.max(F.softmax(model(images_under_attack_1), dim=1), 1)
        # 如果FGSM攻击成功则终止本次图片保存， 测试下一张图片
        if (labels != p_idx_1)[0]:
            bar.update(labels.shape[0])
            continue

        # 选择I-FGSM攻击不成功的样本
        images_under_attack_2 = atk_2(images, labels)
        p_2, p_idx_2 = torch.max(F.softmax(model(images_under_attack_2), dim=1), 1)
        # 如果I-FGSM攻击成功则终止本次图片保存， 测试下一张图片
        if (labels != p_idx_2)[0]:
            bar.update(labels.shape[0])
            continue

        # 选择PGD攻击不成功的样本
        images_under_attack_3 = atk_3(images, labels)
        p_3, p_idx_3 = torch.max(F.softmax(model(images_under_attack_3), dim=1), 1)
        # 如果FGSM攻击成功则终止本次图片保存， 测试下一张图片
        if (labels != p_idx_3)[0]:
            bar.update(labels.shape[0])
            continue

        # 选择MI-FGSM对于集合A攻击未成功的样本
        images_under_attack_4 = atk_4(images, labels)
        p_4, p_idx_4 = torch.max(F.softmax(model(images_under_attack_4), dim=1), 1)
        # 如果MI-FGSM攻击成功则终止本次图片保存， 测试下一张图片
        if (labels != p_idx_4)[0]:
            bar.update(labels.shape[0])
            continue

        # 选择Adam-FGSM对于集合A_4攻击成功的样本
        images_under_attack_5 = atk_5(images, labels)
        p_5, p_idx_5 = torch.max(F.softmax(model(images_under_attack_5), dim=1), 1)
        # 如果Adam-FGSM攻击失败则终止本次图片保存， 测试下一张图片
        if (labels == p_idx_5)[0]:
            bar.update(labels.shape[0])
            continue

        special_images_set = [images,
                              images_under_attack_1,
                              images_under_attack_2,
                              images_under_attack_3,
                              images_under_attack_4,
                              images_under_attack_5]
        # title_set = [
        #     label_name_set[labels[0].item()],
        #     label_name_set[p_idx_1[0].item()] + ': %.2f%%' % (p_1[0].item() * 100),
        #     label_name_set[p_idx_2[0].item()] + ': %.2f%%' % (p_2[0].item() * 100),
        #     label_name_set[p_idx_3[0].item()] + ': %.2f%%' % (p_3[0].item() * 100),
        #     label_name_set[p_idx_4[0].item()] + ': %.2f%%' % (p_4[0].item() * 100),
        #     label_name_set[p_idx_5[0].item()] + ': %.2f%%' % (p_5[0].item() * 100), ]
        # print('I am here')
        title_set = [
            get_label(dataset, labels[0].item()),
            get_label(dataset, p_idx_1[0].item()) + ' (%.2f)' % (p_1[0].item()),
            get_label(dataset, p_idx_2[0].item()) + ' (%.2f)' % (p_2[0].item()),
            get_label(dataset, p_idx_3[0].item()) + ' (%.2f)' % (p_3[0].item()),
            get_label(dataset, p_idx_4[0].item()) + ' (%.2f)' % (p_4[0].item()),
            get_label(dataset, p_idx_5[0].item()) + ' (%.2f)' % (p_5[0].item()), ]
        show_one_image(special_images_set, title_set)
        # 从label_id_set剔除已经不需要的label
        # label_id_set.remove(labels[0].item())
        # print('the label_id_set： ', label_id_set)

        bar.update(labels.shape[0])
        # if len(label_id_set) == 0:
        #     break
        test_count += 1
        if test_count > 5:
            break


if __name__ == '__main__':
    # model_name_set = ['VGG16', 'VGG19', 'ResNet50', 'ResNet101', 'DenseNet121']
    get_hard_to_classify('ResNet50_ImageNet',
                         'ImageNet',
                         Epsilon=5 / 255,
                         Iterations=10,
                         Momentum=1.0)

    # get_hard_to_classify('VGG19',
    #                      'CIFAR10',
    #                      Epsilon=5 / 255,
    #                      Iterations=10,
    #                      Momentum=1.0)

    # get_hard_to_classify('LeNet5',
    #                      'MNIST',
    #                      Epsilon=45 / 255,
    #                      Iterations=10,
    #                      Momentum=1.0)

    print()
    print("----ALL WORK HAVE BEEN DONE!!!----")
