实验环境python>=3.7

1.将`kmeans.py`中line 187

```python
train_data = datasets.MNIST(root = './data',train = True, download=False)
```

中的`download=False`改为`download=True`

2.运行

```
python kmeans.py k
```

即可，k目前支持5、10、20。运行时会显示当前的迭代轮数，类中心相对于上一次的移动，每个cluster的label和单类的准确率，总准确率。每五轮会在`./pics`中可视化一次，实验结束时所有信息会记录在`./statistics`下的log文件中。

