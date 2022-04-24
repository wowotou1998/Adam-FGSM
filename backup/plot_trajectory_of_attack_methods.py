import numpy as np
import matplotlib.pyplot as plt


def f(u, v, sample,label):
    """
    计算高度的函数
    :param x: 向量
    :param y: 向量
    :return: dim(x)*dim(y)维的矩阵
    """
    # the height function
    return (1 - x / 2 + x ** 5 + y ** 3) * np.exp(-x ** 8 - y ** 2)


x = np.linspace(-5, 5, 256)
y = np.linspace(-5, 5, 256)
X, Y = np.meshgrid(x, y)  # 获得网格坐标矩阵

# 进行颜色填充
plt.contourf(X, Y, f(X, Y), 8, cmap='rainbow')
# 进行等高线绘制
c = plt.contour(X, Y, f(X, Y), 8, colors='black')
# 线条标注的绘制
plt.clabel(c, inline=True, fontsize=10)

plt.xticks(())
plt.yticks(())

plt.show()
