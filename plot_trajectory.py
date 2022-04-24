import numpy as np
import torch
import torchvision.models as models
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from tqdm import tqdm
from torchattacks import FGSM  # atk_1
# from torchattacks import BIM as I_FGSM  # atk_2
from torchattacks import PGD  # atk_3
# from torchattacks import MIFGSM as MI_FGSM  # atk_4
# from attack_methods import Adam_FGSM  # atk_5
from attack_methods_to_plot_trajectory import I_FGSM, MI_FGSM, Adam_FGSM
import matplotlib.pyplot as plt
import numpy
from pylab import mpl
import torch.nn.functional as F
from MNIST_models import lenet5, FC_256_128
from pytorchcv.model_provider import get_model as ptcv_get_model
from sklearn.decomposition import PCA
from mpl_toolkits.mplot3d import Axes3D
from torch import nn
from attack_models_on_datasets import load_model_args, load_dataset


# 相似性范围从-1到1：
# -1意味着两个向量指向的方向正好截然相反，1表示它们的指向是完全相同的，
# 0通常表示它们之间是独立的，而在这之间的值则表示中间的相似性或相异性。

def show_one_image(subplot, images, title, color):
    # C*H*W-->H*W*C
    image = numpy.transpose(images[0].cpu().detach().numpy(), (1, 2, 0))
    c, h, w = image.shape
    if c == 1:
        subplot.imshow(image, 'gray')
    else:
        subplot.imshow(image)
    # subplot.axis('off')  # 关掉坐标轴为 off
    # 显示坐标轴但是无刻度
    subplot.set_xticks([])
    subplot.set_yticks([])
    # 设定图片边框粗细
    subplot.spines['top'].set_linewidth('2.0')  # 设置边框线宽为2.0
    subplot.spines['bottom'].set_linewidth('2.0')  # 设置边框线宽为2.0
    subplot.spines['left'].set_linewidth('2.0')  # 设置边框线宽为2.0
    subplot.spines['right'].set_linewidth('2.0')  # 设置边框线宽为2.0
    # 设定边框颜色
    subplot.spines['top'].set_color(color)
    subplot.spines['bottom'].set_color(color)
    subplot.spines['left'].set_color(color)
    subplot.spines['right'].set_color(color)
    # subplot.set_title(title, y=-0.25, color=color, fontsize=8)  # 图像题目


def obtain_one_loss_value(sample, label, model):
    from torch import nn
    criterion = nn.CrossEntropyLoss()
    loss = criterion(model(sample), label).sum().item()
    return loss


def loss_function(u, v, sample_a, sample_b, sample_offset, label, model):
    """
    计算高度的函数
    :param x: 向量
    :param y: 向量
    :return: dim(x)*dim(y)维的矩阵
    """
    result = np.zeros_like(u)
    r, c = u.shape
    for i in range(r):
        for j in range(c):
            sample = u[i][j] * sample_a + v[i][j] * sample_b + sample_offset
            # sample = v[i][j] * sample_b + sample_offset
            # sample = u[i][j] * sample_b + sample_offset
            result[i][j] = obtain_one_loss_value(sample, label, model)
    return result


def region_split(u, v, sample_a, sample_b, label, model):
    """
        计算高度的函数
        :param x: 向量
        :param y: 向量
        :return: dim(x)*dim(y)维的矩阵
        """
    from torch import nn
    criterion = nn.CrossEntropyLoss()

    result = np.zeros_like(u)
    r, c = u.shape
    for i in range(r):
        for j in range(c):
            sample = u[i][j] * sample_a + v[i][j] * sample_b
            _, predict = torch.max(F.softmax(model(sample), dim=1), 1)
            result[i][j] = predict[0].item()
    return result


def obtain_predict_label(image_size, vectors, factors, model):
    n_vector = len(vectors)
    batch, channel, h, w = image_size
    sample = np.zeros((batch * channel * h * w), dtype=float)
    for i in range(n_vector):
        sample += vectors[i] * factors[i]
    device = torch.device("cuda:%d" % (0) if torch.cuda.is_available() else "cpu")
    sample_ = torch.from_numpy(sample).view(batch, channel, h, w).type(torch.FloatTensor).to(device)
    _, predict_label = torch.max(F.softmax(model(sample_), dim=1), 1)
    result = predict_label[0].item()
    return result


def pca_contour_2d(sample_a, sample_b, label, model, many_sample_list):
    # plt.show()
    # plt.figure(figsize=(8, 8))
    # 进行颜色填充
    # plt.contourf(ii, jj, kk, 8, cmap='rainbow')
    # plt.contourf(ii, jj, kk, 8, cmap='coolwarm')
    # 进行等高线绘制
    # contour = plt.contour(ii, jj, loss_function(ii, jj, sample_a, sample_b, label, model), 8, colors='black')
    # # 线条标注的绘制
    # plt.clabel(c, inline=True, fontsize=10)

    # --------------------------数据处理---------------------------------

    list_size_1 = len(many_sample_list)
    list_size_2 = len(many_sample_list[0])
    n_samples = list_size_1 * list_size_2
    n_features = many_sample_list[0][0].nelement()
    data = np.zeros((n_samples, n_features))
    for i in range(list_size_1):
        for j in range(list_size_2):
            data[i * list_size_2 + j] = torch.flatten(many_sample_list[i][j][0]).cpu().numpy()

    # --------------------------数据压缩---------------------------------
    # 我们将数据投影到二维平面上, 二维平面上主成分最大者为x轴, 二维平面上主成分次大者为y轴
    pca = PCA(n_components=2)
    # X: ndarray, array-like of shape (n_samples, n_features)
    pca.fit(data)
    #  new_data_coordinate 表示数据在新坐标系下前k个坐标,这里只有两个坐标
    new_data_coordinate = pca.transform(data)
    print('explained variance ratio (first two components): %s'
          % str(pca.explained_variance_ratio_))

    # --------------------------准备基向量,确定坐标轴的大致形状---------------------------------
    x_min, x_max = new_data_coordinate[:, 0].min(), new_data_coordinate[:, 0].max()
    y_min, y_max = new_data_coordinate[:, 1].min(), new_data_coordinate[:, 1].max()
    x_i = np.linspace(x_min * 1.4, x_max * 1.4, 80)
    y_i = np.linspace(y_min * 1.4, y_max * 1.4, 80)
    ii, jj = np.meshgrid(x_i, y_i)  # 获得网格坐标矩阵

    a, b = pca.components_[0].copy(), pca.components_[1].copy()  # pca.components_ 表示新坐标系下前k个基向量
    batch, channel, h, w = many_sample_list[0][0].shape
    sample_a = torch.from_numpy(a).view(batch, channel, h, w).type(torch.FloatTensor)
    sample_b = torch.from_numpy(b).view(batch, channel, h, w).type(torch.FloatTensor)
    sample_offset = torch.from_numpy(pca.mean_).view(batch, channel, h, w).type(torch.FloatTensor)
    device = torch.device("cuda:%d" % (0) if torch.cuda.is_available() else "cpu")
    sample_a = sample_a.to(device)
    sample_b = sample_b.to(device)
    sample_offset = sample_offset.to(device)

    # --------------------------绘制loss平面---------------------------------
    kk = loss_function(ii, jj, sample_a, sample_b, sample_offset, label, model)
    # kk = region_split(ii, jj, sample_a, sample_b, label, model)

    # 绘制曲面
    # fig_1 = plt.figure()
    # ax = plt.axes(projection='3d')
    # ax.plot_surface(ii, jj, kk, cmap='coolwarm')
    # plt.show()

    # -------------------显示原样本和对抗样本以及迭代轨迹-------------------------
    fig_2 = plt.figure(figsize=(6, 3))
    # subplot2grid, 总的长宽, 块起始点坐标
    grid_size = (2, 4)
    ax_1 = plt.subplot2grid(grid_size, (0, 0))
    sample_1 = new_data_coordinate[0][0] * sample_a + \
               new_data_coordinate[0][1] * sample_b + sample_offset
    sample_1 = sample_offset
    loss_1 = obtain_one_loss_value(sample_1, label, model)
    show_one_image(ax_1, sample_1, 'Original %.2f' % (loss_1), 'k')

    ax_2 = plt.subplot2grid(grid_size, (1, 0))
    sample_2 = new_data_coordinate[list_size_2 - 1][0] * sample_a + \
               new_data_coordinate[list_size_2 - 1][1] * sample_b + sample_offset
    loss_2 = obtain_one_loss_value(sample_2, label, model)
    show_one_image(ax_2, sample_2, 'I-FGSM %.2f' % (loss_2), 'r')

    ax_3 = plt.subplot2grid(grid_size, (0, 3))
    sample_3 = new_data_coordinate[2 * list_size_2 - 1][0] * sample_a + \
               new_data_coordinate[2 * list_size_2 - 1][1] * sample_b + sample_offset
    loss_3 = obtain_one_loss_value(sample_3, label, model)
    show_one_image(ax_3, sample_3, 'MI-FGSM %.2f' % (loss_3), 'g')

    ax_4 = plt.subplot2grid(grid_size, (1, 3))
    sample_4 = new_data_coordinate[3 * list_size_2 - 1][0] * sample_a + \
               new_data_coordinate[3 * list_size_2 - 1][1] * sample_b + sample_offset
    loss_4 = obtain_one_loss_value(sample_4, label, model)
    show_one_image(ax_4, sample_4, 'Adam-FGSM %.2f' % (loss_4), 'b')
    # 显示等高线和攻击轨迹
    plt.subplot2grid(grid_size, (0, 1), colspan=2, rowspan=2)
    plt.contourf(ii, jj, kk, 8, cmap='rainbow')  # 'coolwarm'
    label_set = ['I-FGSM', 'MI-FGSM', 'Adam-FGSM']
    color_set = ['r', 'g', 'b']
    loss_list = [loss_1, loss_2, loss_3, loss_4]
    ax_5 = plt.gca()
    for i in range(list_size_1):
        for j in range(list_size_2):
            idx = i * list_size_2 + j
            x = new_data_coordinate[idx, 0]
            y = new_data_coordinate[idx, 1]
            # 每个列表中初始第一个点不绘制箭头
            if i == 0 and j == 0:
                ax_5.text(x, y - 0.11,
                          r'$x_{Original}$' + '\n    %.2f' % (loss_list[0]),
                          fontsize=8, family='Times New Roman')
                ax_5.scatter(x, y,
                             facecolors='none', edgecolors='black',
                             s=20, marker='o')  # 把 corlor 设置为空，通过edgecolors来控制颜色
            if j == 0:
                previous_xy = [new_data_coordinate[idx, 0], new_data_coordinate[idx, 1]]
            else:
                ax_5.quiver(previous_xy[0], previous_xy[1],
                            x - previous_xy[0], y - previous_xy[1],
                            color=color_set[i],
                            width=0.003,
                            headwidth=10,
                            headlength=10,
                            angles='xy', scale_units='xy', scale=1,
                            label=label_set[i] if j == 1 else None)
                previous_xy[0] = x
                previous_xy[1] = y
            if j == (list_size_2 - 1):
                # 显示一些注释
                ax_5.text(x, y,
                          r'$x^{\prime}_{%s}$' % (label_set[i]) + '\n    %.2f' % (loss_list[i + 1]),
                          fontsize=6,
                          family='Times New Roman')
                # 显示空心圆点
                ax_5.scatter(x, y,
                             facecolors='none', edgecolors=color_set[i],
                             s=20, marker='o')  # 把 corlor 设置为空，通过edgecolors来控制颜色

    ax_5.legend()
    ax_5.axis('off')
    # plt.tight_layout()
    import datetime
    fig = plt.gcf()
    fig.savefig('../Checkpoint/%s_%s.png' % ('trajectory', datetime.datetime.now().strftime("%Y_%m_%d_%H_%M_%S")))
    plt.show()
    fig.savefig('../Checkpoint/%s_%s.pdf' % ('trajectory', datetime.datetime.now().strftime("%Y_%m_%d_%H_%M_%S")))
    print('result has been saved')


def pca_contour_3d(sample_a, sample_b, label, model, many_sample_list):
    # --------------------------数据处理---------------------------------
    # fig_1 = plt.figure()
    list_size_1 = len(many_sample_list)
    list_size_2 = len(many_sample_list[0])
    n_samples = list_size_1 * list_size_2
    n_features = many_sample_list[0][0].nelement()
    data = np.zeros((n_samples, n_features))
    for i in range(list_size_1):
        for j in range(list_size_2):
            data[i * list_size_2 + j] = torch.flatten(many_sample_list[i][j][0]).cpu().numpy()

    # --------------------------数据压缩---------------------------------
    # 我们将数据投影到3维平面上, 3维平面上主成分最大者为x轴, 次大者为y轴, 最小的为z轴
    pca = PCA(n_components=3)
    # X: ndarray, array-like of shape (n_samples, n_features)
    pca.fit(data)
    #  new_data_coordinate 表示数据在新坐标系下前k个坐标,这里只有两个坐标
    new_data_coordinate = pca.transform(data)
    print('explained variance ratio (first 3 components): %s'
          % str(pca.explained_variance_ratio_))

    # --------------------------绘制轨迹---------------------------------
    # pca.components_ 表示新坐标系下前k个基向量
    a, b, c = pca.components_[0].copy(), pca.components_[1].copy(), pca.components_[2].copy()
    x_min, x_max = new_data_coordinate[:, 0].min(), new_data_coordinate[:, 0].max()
    y_min, y_max = new_data_coordinate[:, 1].min(), new_data_coordinate[:, 1].max()
    z_min, z_max = new_data_coordinate[:, 2].min(), new_data_coordinate[:, 2].max()

    batch, channel, h, w = many_sample_list[0][0].shape
    sample_a = torch.from_numpy(a).view(batch, channel, h, w).type(torch.FloatTensor)
    sample_b = torch.from_numpy(b).view(batch, channel, h, w).type(torch.FloatTensor)
    sample_c = torch.from_numpy(c).view(batch, channel, h, w).type(torch.FloatTensor)
    device = torch.device("cuda:%d" % (0) if torch.cuda.is_available() else "cpu")
    sample_a = sample_a.to(device)
    sample_b = sample_b.to(device)
    sample_c = sample_c.to(device)

    # -------------------显示原样本和对抗样本以及迭代轨迹-------------------------
    fig_2 = plt.figure(figsize=(6, 3))
    grid_size = (2, 4)
    ax_1 = plt.subplot2grid(grid_size, (0, 0))
    sample_1 = new_data_coordinate[0][0] * sample_a + \
               new_data_coordinate[0][1] * sample_b
    loss_1 = obtain_one_loss_value(sample_1, label, model)
    show_one_image(ax_1, sample_1, 'Original %.2f' % (loss_1), 'k')

    ax_2 = plt.subplot2grid(grid_size, (1, 0))
    sample_2 = new_data_coordinate[list_size_2 - 1][0] * sample_a + \
               new_data_coordinate[list_size_2 - 1][1] * sample_b
    loss_2 = obtain_one_loss_value(sample_2, label, model)
    show_one_image(ax_2, sample_2, 'I-FGSM %.2f' % (loss_2), 'r')

    ax_3 = plt.subplot2grid(grid_size, (0, 3))
    sample_3 = new_data_coordinate[2 * list_size_2 - 1][0] * sample_a + \
               new_data_coordinate[2 * list_size_2 - 1][1] * sample_b
    loss_3 = obtain_one_loss_value(sample_3, label, model)
    show_one_image(ax_3, sample_3, 'MI-FGSM %.2f' % (loss_3), 'g')

    ax_4 = plt.subplot2grid(grid_size, (1, 3))
    sample_4 = new_data_coordinate[3 * list_size_2 - 1][0] * sample_a + \
               new_data_coordinate[3 * list_size_2 - 1][1] * sample_b
    loss_4 = obtain_one_loss_value(sample_4, label, model)
    show_one_image(ax_4, sample_4, 'Adam-FGSM %.2f' % (loss_4), 'b')

    # 显示等高线和攻击轨迹
    color_set = ['r', 'g', 'b']
    label_set = ['I-FGSM', 'MI-FGSM', 'Adam-FGSM']

    plt.subplot2grid(grid_size, (0, 1), colspan=2, rowspan=2, projection='3d')
    ax_5 = plt.gca()
    for i in range(list_size_1):
        for j in range(list_size_2):
            idx = i * list_size_2 + j
            x = new_data_coordinate[idx, 0]
            y = new_data_coordinate[idx, 1]
            z = new_data_coordinate[idx, 2]
            # ax.scatter(x, y, z, color=color_set[i])
            label = obtain_predict_label(image_size=sample_a.shape,
                                         vectors=pca.components_.copy(),
                                         factors=(x, y, z),
                                         model=model)
            ax_5.text(x, y, z + 0.02, str(label), fontsize=12)
            # 每个列表中初始第一个点不绘制箭头
            if j == 0:
                previous_xy = [new_data_coordinate[idx, 0],
                               new_data_coordinate[idx, 1],
                               new_data_coordinate[idx, 2]]
            else:
                ax_5.quiver(previous_xy[0], previous_xy[1], previous_xy[2],
                            x - previous_xy[0], y - previous_xy[1], z - previous_xy[2],
                            color=color_set[i],
                            arrow_length_ratio=0.15,
                            label=label_set[i] if j == 1 else None
                            )
                previous_xy[0] = x
                previous_xy[1] = y
                previous_xy[2] = z
    ax_5.legend()
    ax_5.set_xlim(x_min, x_max)
    ax_5.set_ylim(y_min, y_max)
    ax_5.set_zlim(z_min, z_max)
    plt.show()


def get_hard_to_classify(dataset, mode_name, Epsilon, Iterations, Momentum):
    # data_tf = transforms.Compose(
    #     [
    #         transforms.ToTensor(),
    #     ]
    # )
    # if dataset == 'MNIST':
    #     test_dataset = datasets.MNIST(root='../DataSet/MNIST', train=False, transform=data_tf, download=True)
    #     test_loader = DataLoader(dataset=test_dataset, batch_size=1, shuffle=False)
    # else:
    #     test_dataset = datasets.CIFAR10(root='../DataSet/CIFAR10', train=False, transform=data_tf)
    #     test_loader = DataLoader(dataset=test_dataset, batch_size=1, shuffle=False)
    test_loader, test_dataset_size = load_dataset(dataset, batch_size=1, is_shuffle=True)
    model, model_acc = load_model_args(mode_name)
    device = torch.device("cuda:%d" % (0) if torch.cuda.is_available() else "cpu")
    model.to(device)
    model.eval()

    label_id_set = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]
    label_name_set = ('0', '1', '2', '3', '4', '5', '6', '7', '8', '9')

    test_count = 0
    bar = tqdm(total=test_dataset_size)
    print('len(test_loader)', len(test_loader))
    for data in test_loader:
        title_set = []

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
        atk_2 = I_FGSM(model,
                       eps=Epsilon,
                       alpha=Epsilon / Iterations,
                       steps=Iterations)
        atk_3 = PGD(model,
                    eps=Epsilon,
                    alpha=Epsilon / Iterations,
                    steps=Iterations)
        atk_4 = MI_FGSM(model,
                        eps=Epsilon,
                        alpha=Epsilon / Iterations,
                        steps=Iterations,
                        decay=Momentum)
        atk_5 = Adam_FGSM(model,
                          eps=Epsilon,
                          steps=Iterations,
                          decay=Momentum)
        # -----------------------------

        # # 选择FGSM攻击不成功的样本
        # images_under_attack_1 = atk_1(images, labels)
        # p_1, p_idx_1 = torch.max(F.softmax(model(images_under_attack_1), dim=1), 1)
        # # 如果FGSM攻击成功则终止本次图片保存， 测试下一张图片
        # if (labels != p_idx_1)[0]:
        #     bar.update(labels.shape[0])
        #     continue

        # 选择I-FGSM攻击不成功的样本
        images_under_attack_2, i_fgsm_adv_images = atk_2(images, labels), atk_2.get_iterations()
        p_2, p_idx_2 = torch.max(F.softmax(model(images_under_attack_2), dim=1), 1)
        # 如果I-FGSM攻击成功则终止本次图片保存， 测试下一张图片
        if (labels != p_idx_2)[0]:
            bar.update(labels.shape[0])
            continue

        # # 选择PGD攻击不成功的样本
        # images_under_attack_3 = atk_3(images, labels)
        # p_3, p_idx_3 = torch.max(F.softmax(model(images_under_attack_3), dim=1), 1)
        # # 如果FGSM攻击成功则终止本次图片保存， 测试下一张图片
        # if (labels != p_idx_3)[0]:
        #     bar.update(labels.shape[0])
        #     continue
        # 选择MI-FGSM对于集合A攻击未成功的样本
        images_under_attack_4, mi_fgsm_adv_images = atk_4(images, labels), atk_4.get_iterations()
        p_4, p_idx_4 = torch.max(F.softmax(model(images_under_attack_4), dim=1), 1)
        # 如果MI-FGSM攻击成功则终止本次图片保存， 测试下一张图片
        if (labels != p_idx_4)[0]:
            bar.update(labels.shape[0])
            continue
        # 选择Adam-FGSM对于集合A_4攻击成功的样本
        images_under_attack_5, adam_fgsm_adv_images = atk_5(images, labels), atk_5.get_iterations()
        p_5, p_idx_5 = torch.max(F.softmax(model(images_under_attack_5), dim=1), 1)
        # 如果Adam-FGSM攻击失败则终止本次图片保存， 测试下一张图片
        if (labels == p_idx_5)[0]:
            bar.update(labels.shape[0])
            continue

        # 从label_id_set剔除已经不需要的label
        # label_id_set.remove(labels[0].item())
        print()
        print('the label: %d is selected' % (labels[0].item()))
        # print('rest label is ： ', label_id_set)
        bar.update(labels.shape[0])

        # 准备工作
        sample_a = adam_fgsm_adv_images[0]
        sample_b = adam_fgsm_adv_images[Iterations]
        # 绘制超平面和轨迹
        many_adv_images_list = [i_fgsm_adv_images, mi_fgsm_adv_images, adam_fgsm_adv_images]
        pca_contour_2d(sample_a, sample_b, labels, model, many_adv_images_list)
        pca_contour_3d(sample_a, sample_b, labels, model, many_adv_images_list)
        test_count += 1
        if test_count > 10:
            break


if __name__ == '__main__':
    mpl.rcParams['font.sans-serif'] = ['Times New Roman']
    # 解决保存图像是负号'-'显示为方块的问题
    mpl.rcParams['axes.unicode_minus'] = False
    mpl.rcParams['savefig.dpi'] = 200  # 保存图片分辨率
    mpl.rcParams['figure.dpi'] = 200  # 分辨率
    mpl.rcParams["figure.subplot.left"], mpl.rcParams["figure.subplot.right"] = 0.05, 0.99
    mpl.rcParams["figure.subplot.bottom"], mpl.rcParams["figure.subplot.top"] = 0.07, 0.99
    mpl.rcParams["figure.subplot.wspace"], mpl.rcParams["figure.subplot.hspace"] = 0.1005, 0.1005
    plt.rcParams['xtick.direction'] = 'in'  # 将x周的刻度线方向设置向内
    plt.rcParams['ytick.direction'] = 'in'  # 将y轴的刻度方向设置向内

    # model_name_set = ['VGG16', 'VGG19', 'ResNet50', 'ResNet101', 'DenseNet121']
    get_hard_to_classify('MNIST',
                         'FC_256_128',
                         Epsilon=35 / 255,
                         Iterations=10,
                         Momentum=1.0)

    # get_hard_to_classify('CIFAR10',
    #                      'VGG19',
    #                      Epsilon=5 / 255,
    #                      Iterations=8,
    #                      Momentum=1.0)

    # get_hard_to_classify('ImageNet',
    #                      'ResNet50_ImageNet',
    #                      Epsilon=3 / 255,
    #                      Iterations=8,
    #                      Momentum=1.0)
    print()
    print("----ALL WORK HAVE BEEN DONE!!!----")
