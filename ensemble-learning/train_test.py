import random, datetime, numpy as npy
# 存储一个固定顺序，所有模型都按这个顺序切分数据集
random.seed(datetime.datetime.now().timestamp())
list = []
for i in range(1,220001):
    list.append(i)
random.shuffle(list)
vectors = npy.array(list)
# npy.save('division.npy', vectors)

