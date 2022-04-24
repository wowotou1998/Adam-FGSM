import csv

import numpy
import numpy as np
import matplotlib.pyplot as plt
import torch
from pylab import mpl
import os
from matplotlib.pyplot import MultipleLocator


# 从pyplot导入MultipleLocator类，这个类用于设置刻度间隔
def string_to_float(str_list):
    a, b = len(str_list), len(str_list[0])
    number_list = numpy.zeros((a, b), dtype=np.float)
    for i in range(a):
        for j in range(b):
            number_list[i][j] = str_list[i][j]
    return number_list


os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'
# 指定默认字体
# 显示中文
# plt.rcParams['font.sans-serif'] = [u'SimHei']
mpl.rcParams['font.sans-serif'] = ['Times New Roman']
# 解决保存图像是负号'-'显示为方块的问题
mpl.rcParams['axes.unicode_minus'] = False
mpl.rcParams['savefig.dpi'] = 400  # 保存图片分辨率
mpl.rcParams['figure.dpi'] = 400  # 分辨率
mpl.rcParams["figure.subplot.left"], mpl.rcParams["figure.subplot.right"] = 0.16, 0.99
mpl.rcParams["figure.subplot.bottom"], mpl.rcParams["figure.subplot.top"] = 0.13, 0.98
mpl.rcParams["figure.subplot.wspace"], mpl.rcParams["figure.subplot.hspace"] = 0.1, 0.1
plt.rcParams['xtick.direction'] = 'in'  # 将x周的刻度线方向设置向内
plt.rcParams['ytick.direction'] = 'in'  # 将y轴的刻度方向设置向内

file_name = os.path.basename(__file__).split(".")[0]

model_name_set = ['VGG-19', 'Res-101', 'Dense-121']
line_style = ['-.', '--', '-', ]
marker = ['o', 'v', '+']
attack_method_set = ['I-FGSM', 'MI-FGSM', 'Adam-FGSM']
color_set = ['r', 'b', 'g', ]
data_type_set = [r'$L_{1}$ distance', r'$L_{\infty}$ distance', r'Cosine Similarity']
legend_set = []
iterations = [str(i) for i in range(1, 16)]


def get_norm_1_norm_inf_similarity():
    with open('../Checkpoint/explain_best_one.csv', 'r') as f:
        content = list(csv.reader(f))
        # content = content[1:-1][6:21]
        content = numpy.array(content)[1:, 6:21]
        content = string_to_float(content)
        print(content)
        VGG19 = content[0 * 9:1 * 9, 6:21]
        Res101 = content[1 * 9:2 * 9, 6:21]
        Dense121 = content[2 * 9:3 * 9, 6:21]
        # ----L_1 norm----
        # I-FGSM-VGG16
        # I-FGSM-Res101
        # I-FGSM-Dense121

        # MI-FGSM-VGG16
        # MI-FGSM-Res101
        # MI-FGSM-Dense121

        # Adam-FGSM-VGG16
        # Adam-FGSM-Res101
        # Adam-FGSM-Dense121

        # L_inf norm
        # I-FGSM-VGG16
        # I-FGSM-Res101
        # I-FGSM-Dense121

        # MI-FGSM-VGG16
        # MI-FGSM-Res101
        # MI-FGSM-Dense121

        # Adam-FGSM-VGG16
        # Adam-FGSM-Res101
        # Adam-FGSM-Dense121
        L_1_norm = []
        L_inf_norm = []
        Similarity = []
        for i, attack_method in enumerate(attack_method_set):
            for j, model in enumerate(model_name_set):
                tmp_L1 = content[j * 9 + i * 3 + 0]
                L_1_norm.append(tmp_L1)
                tmp_L_inf = content[j * 9 + i * 3 + 1]
                L_inf_norm.append(tmp_L_inf)
                tmp_similarity = content[j * 9 + i * 3 + 2]
                Similarity.append(tmp_similarity)
        return L_1_norm, L_inf_norm, Similarity


def plot_data(data_type_i, data):
    # 宽度,高度
    plt.figure(figsize=(4, 3))
    # plt.title("epsilon compare", fontsize=18)
    plt.xlabel('Iterations')
    plt.ylabel(data_type_set[data_type_i])
    plt.grid(axis='y')
    # plt.grid(b=True, axis='y') #只显示x轴网格线
    # 调整x轴和y轴的范围
    # plt.xlim(xmin=0, xmax=15)
    # plt.ylim(ymin=70, ymax=100)
    # y_major_locator = MultipleLocator(5)
    # 把y轴的刻度间隔设置为10，并存在变量里
    # ax = plt.gca()
    # #把x轴的主刻度设置为1的倍数
    # y_major_locator = MultipleLocator(1)
    # ax.yaxis.set_major_locator(y_major_locator)
    # 把y轴的主刻度设置为10的倍数
    # 添加一个坐标轴，默认0到1
    # plt.twinx()
    # plt.style.use('bmh')
    for i, attack in enumerate(attack_method_set):
        for j, model in enumerate(model_name_set):
            plt.plot(iterations, data[i * 3 + j],
                     linestyle=line_style[i],
                     # linestyle='',
                     marker=marker[j], markersize=2,
                     color=color_set[i], linewidth=0.5, )
            legend_set.append(attack + ' vs. ' + model)

    # 直接传入legend
    plt.legend(legend_set, loc='best', fontsize=6.5)

    '''
    切记下面的顺序不能更改
    '''
    fig = plt.gcf()
    fig.savefig('./output_pictures/%s_%s.png' % (file_name, data_type_i))
    plt.show()
    fig.savefig('./output_pdfs/%s_%s.pdf' % (file_name, data_type_i))
    print('result has been saved')


# L_1_norm, L_inf_norm, Similarity
# 上述三者都是每一个都是(attack_method * models) * iterations 大小
L_1_norm, L_inf_norm, Similarity = get_norm_1_norm_inf_similarity()
for idx, data in enumerate([L_1_norm, L_inf_norm, Similarity]):
    # if idx == 0:
    plot_data(idx, data)
