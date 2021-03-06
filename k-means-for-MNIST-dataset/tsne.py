from random import sample
import numpy as np
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE
from torchvision import datasets 

"""
部分参考https://blog.csdn.net/tszupup/article/details/84997804
《t-SNE算法的基本思想及其Python实现》by 烟雨风渡
"""

def sample(train_data): # 从60000个样本中按每个label的比例sample出1200个样本
    pics = {}
    samples = []
    for i in range(10):
        pics[i] = []
    for i in range(60000):
        pics[train_data[i][1]].append(i)
    for i in range(10):
        num = len(pics[i])
        if num%50 > 21:
            num+=50
        num=int(num/50)
        for j in range(num):
            samples.append((pics[i])[j])
    samples.sort() # samples存储的是图片的编号，排序
    return samples

def to_matrix(train_data,samples): # 把samples中的图片向量化后再放到一个矩阵中
    ma = []
    for idx in samples:
        pic = train_data[idx][0].resize((14,14))
        vec = []
        for i in range(14):
            for j in range(14):
                vec.append(pic.getpixel((i,j))+0.0)
        ma.append(vec)
    return np.array(ma)

def main():
    plt.figure(figsize=(7,7))
    train_data = datasets.MNIST(root = './data',train = True, download=False)
    samples = sample(train_data)
    colors = []
    for ele in samples:
        colors.append(train_data[ele][1]) # 颜色根据label确定，ground truth
    x = to_matrix(train_data,samples)
    ts = TSNE(n_components=2) # 运用tsne降维
    y = ts.fit_transform(x)
    plt.scatter(y[:, 0], y[:, 1],c=colors, cmap=plt.cm.Spectral)
    plt.savefig('./tsne/visualization_ground_truth.png')
    plt.show()
    y = np.array(y)
    samples = np.array(samples)
    colors = np.array(colors)
    np.save("./tsne/points.npy",y) # 存储降维后的向量，kmeans.py会用到
    np.save("./tsne/samples.npy",samples) # 存储sample的点，kmeans.py会用到
    np.save("./tsne/colors.npy",colors)

if __name__ == "__main__":
    main()
