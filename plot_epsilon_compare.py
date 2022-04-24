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
mpl.rcParams['savefig.dpi'] = 300  # 保存图片分辨率
mpl.rcParams['figure.dpi'] = 300  # 分辨率
mpl.rcParams["figure.subplot.left"], mpl.rcParams["figure.subplot.right"] = 0.15, 0.995
mpl.rcParams["figure.subplot.bottom"], mpl.rcParams["figure.subplot.top"] = 0.11, 0.995
# mpl.rcParams["figure.subplot.wspace"], mpl.rcParams["figure.subplot.hspace"] = 0.21, 0.18
plt.rcParams['xtick.direction'] = 'in'  # 将x周的刻度线方向设置向内
plt.rcParams['ytick.direction'] = 'in'  # 将y轴的刻度方向设置向内

file_name = 'epsilon_compare_1_10_1'

model_name_set = ['FC-256-128', 'LeNet-5', 'VGG-16', 'VGG-19', 'Res-50', 'Res-101', 'Dense-121', 'Model-average']
# model_name_set = ['Models Average']
line_style = ['-+', '-+', '-o', ':+', '-.o', '-.+', '-o', '-+', ]
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
attack_num = len(attack_method_set)
model_num = len(model_name_set)
column_start, column_end = 5, 10
epsilons = ['1', '2', '3', '4', '5', '6', '7', '8', '9', '10']
large_epsilons = ['41', '42', '43', '44', '45', '46', '47', '48', '49', '50']
epsilons_num = len(epsilons)
# legend_set = []
# 宽和高
# plt.figure(figsize=(10, 5))
# plt.subplot(n_rows, n_cols, plot_num)
with open('../Checkpoint/epsilon_compare_best_one_all.csv', 'r') as f:
    content = list(csv.reader(f))
    content = numpy.array(content)[1:, column_start:column_end]
    content = string_to_float(content)
    print(content)
    data_ave = numpy.zeros((epsilons_num, len(attack_method_set)), dtype=float)
    for model_i, model in enumerate(model_name_set):
        # 2行3列
        # plt.subplot(2, 4, model_i + 1)
        # 单独一张图
        # figsize 列和行
        plt.figure(figsize=(3, 3))
        legend_set = []
        # 计算平均值
        if model_i != (len(model_name_set) - 1):
            data = content[model_i * epsilons_num:(model_i + 1) * epsilons_num]
            data_ave += data
        else:
            data_ave /= (len(model_name_set) - 1)
            data = data_ave

        for attack_i, attack_method in enumerate(attack_method_set):
            # 对不同的模型设置不同的x轴数据
            if model_i < 2:
                x_axis_array = large_epsilons
            else:
                x_axis_array = epsilons
            plt.plot(x_axis_array, data[:, attack_i],
                     linestyle='-',
                     marker=marker[model_i],
                     color=color_set[attack_i],
                     markersize=3,
                     linewidth=0.75)
            legend_set.append(model_name_set[model_i] + ' vs. ' + attack_method)
        # 直接传入legend
        plt.legend(legend_set, loc='lower right', fontsize=6.5)
        plt.xlabel('Perturbation Magnitude(1/255)', labelpad=1)
        plt.ylabel('Success Rate(%)', labelpad=0)
        plt.rcParams['xtick.direction'] = 'in'  # 将x周的刻度线方向设置向内
        plt.rcParams['ytick.direction'] = 'in'  # 将y轴的刻度方向设置向内
        # plt.title("epsilon compare", fontsize=18)
        plt.grid(axis='y')
        # plt.ylim(ymax=99)
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

# 绘制好图片后使用plt(pyplot)显示图片并保存图片
# plt.savefig('./output_pdfs/%s.eps' % (file_name), format='eps', dpi=1000)
# print('result has been saved')
