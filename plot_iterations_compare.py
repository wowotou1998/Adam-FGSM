import csv

import numpy
import matplotlib.pyplot as plt
import numpy as numpy
import torch
from pylab import mpl
import os
from matplotlib.pyplot import MultipleLocator


# 从pyplot导入MultipleLocator类，这个类用于设置刻度间隔
def string_to_float(str_list):
    a, b = len(str_list), len(str_list[0])
    number_list = numpy.zeros((a, b), dtype=float)
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
mpl.rcParams['savefig.dpi'] = 200  # 保存图片分辨率
mpl.rcParams['figure.dpi'] = 200  # 分辨率
mpl.rcParams["figure.subplot.left"], mpl.rcParams["figure.subplot.right"] = 0.17, 0.995
mpl.rcParams["figure.subplot.bottom"], mpl.rcParams["figure.subplot.top"] = 0.12, 0.995
mpl.rcParams["figure.subplot.wspace"], mpl.rcParams["figure.subplot.hspace"] = 0.22, 0.20
plt.rcParams['xtick.direction'] = 'in'  # 将x周的刻度线方向设置向内
plt.rcParams['ytick.direction'] = 'in'  # 将y轴的刻度方向设置向内

file_name = 'iteration_compare_1_20_2'

# 宽度,高度
# plt.figure(figsize=(4, 3))


# model_name_set = ['VGG-16', 'Res-121', 'Dense-121']
# model_name_set = ['Models Average']
model_name_set = ['FC-256-128', 'LeNet-5', 'VGG-16', 'VGG-19', 'Res-50', 'Res-101', 'Dense-121', 'Model-average']
line_style_marker = ['-o', '-o', '-o', ':+', '-.o', '-.+', '-o', '-+', ]
marker = ['o', 's', 'D', '*', 'v', 'x', '>', '^']
attack_method_set = ['FGSM', 'I-FGSM', 'PGD', 'MI-FGSM', 'Adam-FGSM']
# color_set = ['c', 'b', 'y', 'r', 'g', ]
light_blue = '#03a9f4'
teal = '#009688'
light_green = '#8bc34a'
yellow = '#ffeb3b'
orange = '#ff9800'
indigo = '#3f51b5'
color_set = [light_blue, indigo, orange, light_green, teal]
iterations = ['1', '2', '4', '6', '8', '10', '12', '14', '16', '18', '20']
iterations_num = len(iterations)
legend_set = []
with open('../Checkpoint/iterations_compare_best_one_all.csv', 'r') as f:
    content = csv.reader(f)
    content = numpy.array(list(content))
    content = content[1:, 5:10]
    content = string_to_float(content)
    # print(content)

    # average_success_rate on five models using five methods
    # we will create a matrix, size is (methods) * (iteration)
    data_ave = numpy.zeros((iterations_num, len(attack_method_set)), dtype=float)
    # plt.figure(figsize=(10, 5))
    for model_i, model in enumerate(model_name_set):
        # plt.subplot(2, 4, model_i + 1)
        plt.figure(figsize=(3, 3))
        legend_set = []
        if model_i != (len(model_name_set) - 1):
            data = content[model_i * iterations_num:(model_i + 1) * iterations_num]
            data_ave += data
        else:
            data_ave /= (len(model_name_set) - 1)
            data = data_ave
        for attack_i, attack_method in enumerate(attack_method_set):
            plt.plot(iterations, data[:, attack_i],
                     linestyle='-',
                     marker=marker[model_i],
                     color=color_set[attack_i],
                     markersize=3,
                     linewidth=0.5)
            legend_set.append(model_name_set[model_i] + ' vs. ' + attack_method)
        # 直接传入legend
        plt.legend(legend_set, loc='lower right', fontsize=6.5)
        plt.xlabel('Iterations')
        plt.ylabel('Success Rate(%)')
        plt.rcParams['xtick.direction'] = 'in'  # 将x周的刻度线方向设置向内
        plt.rcParams['ytick.direction'] = 'in'  # 将y轴的刻度方向设置向内
        # plt.title("epsilon compare", fontsize=18)
        plt.grid(axis='y')

        # # 调整x轴和y轴的范围
        # # plt.xlim(xmin=0, xmax=10)
        # plt.ylim(ymin=20, ymax=100)
        # y_major_locator = MultipleLocator(20)
        # 把y轴的刻度间隔设置为10，并存在变量里
        # ax = plt.gca()
        # #把x轴的主刻度设置为1的倍数
        # ax.yaxis.set_major_locator(y_major_locator)
        # 把y轴的主刻度设置为10的倍数
        # 添加一个坐标轴，默认0到1
        # plt.twinx()
        # plt.style.use('ggplot')
        '''
        切记下面的顺序不能更改
        '''
        fig = plt.gcf()
        fig.savefig('./output_pictures/%s_%s.png' % (file_name, model.replace('-', '_')))
        plt.show()
        fig.savefig('./output_pdfs/%s_%s.pdf' % (file_name, model.replace('-', '_')))
        print('result has been saved')

# with open('../Checkpoint/iterations_compare_best_one.csv', 'r') as f:
#     content = csv.reader(f)
#     content = numpy.array(list(content))
#     content = content[1:, 5:10]
#     content = string_to_float(content)
#     # print(content)
#     # average_success_rate on five models using five methods
#     # we will create a matrix, size is (methods) * (iteration)
#     iterations = ['1', '5', '10', '15', '20', '25', '30']
#     numpy_data = numpy.zeros((len(attack_method_set), len(iterations)), dtype=float)
#     for attack_i, attack_method in enumerate(attack_method_set):
#         for iter, iteration in enumerate(iterations):
#             ave_succ = 0.
#             for model_i, model in enumerate(model_name_set):
#                 ave_succ += content[model_i * len(model_name_set) + iter][attack_i]
#             numpy_data[attack_i][iter] = ave_succ / len(model_name_set)
#             # print(data)
#         plt.plot(iterations, numpy_data[attack_i], line_style_marker[0], color=color_set[attack_i], markersize=4,
#                  linewidth=1)
#         legend_set.append(model_name_set[0] + ' vs. ' + attack_method)
#     plt.legend(legend_set, loc='best', fontsize=6.5, edgecolor='w', facecolor='w')
