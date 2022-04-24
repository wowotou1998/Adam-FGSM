import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

plt.rcParams["font.sans-serif"] = "Simhei"
# 填充符号
marks = ["o", "X", "+", "*", "O", "."]
# 设置X,Y轴的值
y = np.random.randint(10, 100, len(marks))
x = range(len(marks))
# 画图
bars = plt.bar(x, y, color="w", edgecolor="k")
# 填充
for bar, mark in zip(bars, marks):
    bar.set_hatch(mark)
# 其他设置
plt.title("柱形图填充", fontsize=20)
plt.xlabel("柱形", fontsize=20)
plt.xticks(x, marks, fontsize=20)
plt.show()
# plt.savefig("out/1.png", dpi=200, bbox_inches="tight")
