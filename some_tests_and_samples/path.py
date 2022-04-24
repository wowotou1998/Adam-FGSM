import os
from os import path

d = path.dirname(__file__)  # 返回当前文件所在的目录
print(d)
# __file__ 为当前文件, 若果在ide中运行此行会报错,可改为  #d = path.dirname('.')
# 获得某个路径的父级目录:( 强烈建议使用该方法！可以逐层获取到根目录的地址，例如D：/）
parent_path = os.path.dirname(d)  # 获得d所在的目录,即d的父级目录
print(parent_path)
parent_path = os.path.dirname(parent_path)  ##获得parent_path所在的目录即parent_path的父级目录
print(parent_path)

nums = [4, 1, 5, 2, 9, 6, 8, 7]


def generate_tuple_list(l):
    L = []
    for idx, value in enumerate(l):
        L.append((idx, value))
    return L


print(enumerate(nums))
a = generate_tuple_list(nums)


def takeSecond(elem):
    return elem[1]


a.sort(key=takeSecond)
print(a)
# ----
nums = [4, 1, 5, 2, 9, 6, 8, 7]
sorted_nums = sorted(enumerate(nums), key=lambda x: x[1])
print('orginal idx,value\n', sorted_nums)
