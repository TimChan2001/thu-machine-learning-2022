import random, datetime, numpy as npy

random.seed(datetime.datetime.now().timestamp())
list = []
for i in range(1,220001):
    list.append(i)
random.shuffle(list)
vectors = npy.array(list)
# npy.save('division.npy', vectors)
print(list)
