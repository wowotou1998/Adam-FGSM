import numpy as np
import matplotlib.pyplot as plt
from pylab import mpl
import os

os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'
# 指定默认字体
# 显示中文
# plt.rcParams['font.sans-serif'] = [u'SimHei']
mpl.rcParams['font.sans-serif'] = ['Times New Roman']
# 解决保存图像是负号'-'显示为方块的问题
mpl.rcParams['axes.unicode_minus'] = False
mpl.rcParams['savefig.dpi'] = 200  # 保存图片分辨率
mpl.rcParams['figure.dpi'] = 200  # 分辨率
mpl.rcParams["figure.subplot.left"], mpl.rcParams["figure.subplot.right"] = 0.20, 0.95
mpl.rcParams["figure.subplot.bottom"], mpl.rcParams["figure.subplot.top"] = 0.15, 0.95
mpl.rcParams["figure.subplot.wspace"], mpl.rcParams["figure.subplot.hspace"] = 0.1, 0.1
plt.rcParams['xtick.direction'] = 'in'  # 将x周的刻度线方向设置向内
plt.rcParams['ytick.direction'] = 'in'  # 将y轴的刻度方向设置向内
# plt.style.use(['matlab'])
# plt.rcParams['ztick.direction'] = 'in'  # 将y轴的刻度方向设置向内
ax = plt.figure().add_subplot(projection='3d')

# Plot a sin curve using the x and y axes.
x = np.linspace(0, 1, 100)
y = np.sin(x * 2 * np.pi) / 2 + 0.5
ax.plot(x, y, zs=0, zdir='z', label='curve in (x, y)')

# Plot scatterplot data (20 2D points per colour) on the x and z axes.
colors = ('r', 'g', 'b', 'k')

# Fixing random state for reproducibility
np.random.seed(19680801)

x = np.random.sample(20 * len(colors))
y = np.random.sample(20 * len(colors))
c_list = []
for c in colors:
    c_list.extend([c] * 20)
# By using zdir='y', the y value of these points is fixed to the zs value 0
# and the (x, y) points are plotted on the x and z axes.
ax.scatter(x, y, zs=0, zdir='y', c=c_list, label='points in (x, z)')

# Make legend, set axes limits and labels
ax.legend()
ax.set_xlim(0, 1)
ax.set_ylim(0, 1)
ax.set_zlim(0, 1)
ax.set_xlabel('X')
ax.set_ylabel('Y')
ax.set_zlabel('Z')


# Customize the view angle so it's easier to see that the scatter points lie
# on the plane y=0
ax.view_init(elev=20., azim=-35)
ax.tick_params(axis='both', direction='in')
plt.tick_params(axis='both', direction='in')
plt.show()
