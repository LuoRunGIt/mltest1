import sklearn.cluster
from sklearn.datasets import make_blobs

import matplotlib.pyplot as plt

# make_blobs函数可以生成一个随机数样本
# n_samples=100 样本数量
# n_features=2 特征数量，这个会让x_train的维度变为3
# centerns=3 中心点,取值方式一般是0，1，2，3这种整数范围
# 返回值
# x_train 测试集 x为多个特征的
# y_train 特征值的标签
x_train, y_train = make_blobs(n_samples=4, n_features=3, centers=3)
print(y_train)
print(x_train)  # x 为一个二维矩阵
# scatter是一个散点图
plt.scatter(x_train[:, 0], x_train[:, 1], c=y_train, marker="*")
plt.show()

# 我的理解 创建一个100个3个特征值的样本 并选取3个中心点。这个是一个模拟算法，或者说是demo


# 接下来是一个demo2
# n_clusters 表示分几类
# 创建一个kmeans对象
# 这里存在一个n_init参数，现在还不清楚是啥但是这个默认值是10也就意味着样本不能小于10?,设为默认也能运行就是会有警告
# 已解决，该参数直接显示设置为auto，应该是和模型精度有关
kmeans = sklearn.cluster.KMeans(n_clusters=3, n_init="auto")
# 把x_train放进去训练
kmeans.fit(x_train)
# 相同数据进行预测，因为没有测试集
y_ = kmeans.predict(x_train)
plt.scatter(x_train[:, 0], x_train[:, 1], c=y_)
plt.show()
