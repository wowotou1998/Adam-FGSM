import csv

import numpy as np
import matplotlib.pyplot as plt
import torch
from pylab import mpl
import os
from matplotlib.pyplot import MultipleLocator


# 从pyplot导入MultipleLocator类，这个类用于设置刻度间隔
def string_to_float(str_list):
    a, b = len(str_list), len(str_list[0])
    number_list = np.zeros((a, b), dtype=float)
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
mpl.rcParams["figure.subplot.left"], mpl.rcParams["figure.subplot.right"] = 0.08, 0.995
mpl.rcParams["figure.subplot.bottom"], mpl.rcParams["figure.subplot.top"] = 0.08, 0.995
# mpl.rcParams["figure.subplot.wspace"], mpl.rcParams["figure.subplot.hspace"] = 0.22, 0.20
plt.rcParams['xtick.direction'] = 'in'  # 将x周的刻度线方向设置向内
plt.rcParams['ytick.direction'] = 'in'  # 将y轴的刻度方向设置向内

file_name = 'confidence_compare'

# 宽度,高度
# plt.figure(figsize=(4, 3))
purple = '#9c27b0'
blue_gray = '#607dd2'
brown = '#795548'
blue = '#2196f3'

# 秋天成熟配色
# amber = '#ffc107'
# yellow = '#ffeb3b'
# lime = '#cddc39'
# light_green = '#8bc34a'
# green = '#4caf50'
# color_set = [amber, yellow, lime, light_green, green]
# 强对比色
light_blue = '#03a9f4'
teal = '#009688'
light_green = '#8bc34a'
yellow = '#ffeb3b'
orange = '#ff9800'
indigo = '#3f51b5'
color_set = [light_blue, indigo, orange, light_green, teal]
model_name_set = ['FC-256-128', 'LeNet-5', 'VGG-16', 'VGG-19', 'Res-50', 'Res-101', 'Dense-121']
attack_method_set = ['FGSM', 'I-FGSM', 'PGD', 'MI-FGSM', 'Adam-FGSM']


legend_set = []
with open('../Checkpoint/confidence.csv', 'r') as f:
    content = csv.reader(f)
    content = np.array(list(content))
    content = content[1:, 1:6]
    content = string_to_float(content)
    # print(content)
    # 列与行

    width_val = 0.15  # 若显示 n 个柱状图，则width_val的值需小于1/n ，否则柱形图会有重合
    x = np.arange(len(model_name_set))  # the label locations
    offset = [i * width_val - 2 * width_val for i in range(len(attack_method_set))]
    fig = plt.figure(figsize=(6, 3))
    ax = plt.axes()
    for attack_i, attack_method in enumerate(attack_method_set):
        data = content[:, attack_i]
        # print()
        histogram_i = ax.bar(x + offset[attack_i], data,
                             color=color_set[attack_i], width=width_val, label=attack_method)
        ax.bar_label(histogram_i, padding=3, fontsize=3)

    # Add some text for labels, title and custom x-axis tick labels, etc.
    ax.set_ylabel('Confidence')
    # ax.set_title('Scores by group and gender')
    ax.set_xticks(x)
    ax.set_xticklabels(model_name_set)
    ax.legend(loc='lower right')

    # ax.bar_label(rects1, padding=3)
    # ax.bar_label(rects2, padding=3)

    # fig.tight_layout()
    fig = plt.gcf()
    fig.savefig('./output_pictures/%s.png' % (file_name))
    plt.show()
    fig.savefig('./output_pdfs/%s.pdf' % (file_name))
    print('result has been saved')
