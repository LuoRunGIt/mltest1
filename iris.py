# 逻辑分类问题
# 鸢尾花实验
from sklearn.linear_model import LogisticRegression
from sklearn.datasets import load_iris
import numpy as np
import matplotlib.pyplot as plt

# 导入鸢尾花数据
iris = load_iris()
# print(iris)

# target 只有0，1，2，表示3种类型的鸢尾花
# date 里面是4个维度分别表示花萼长度、花萼宽度、花瓣长度、花瓣宽度，等于是这里有4个特征

# 散点图绘制
# 这里散点图的特点是每次选取2个特征
# 注意这里有150个样本
# DD=iris.data
# X=[x[0] for x in DD]
# print(X)
# 注意这里是y1 写x1是不正确的
# Y=[y[1] for y in DD]
# print(Y)
# plt.scatter(X[:50], Y[:50], color='red', marker='o', label='setosa')
# plt.scatter(X[50:100], Y[50:100], color='blue', marker='x', label='versicolor')
# plt.scatter(X[100:], Y[100:],color='green', marker='+', label='Virginica') #后50个样本
# plt.legend(loc=2) #loc=1，2，3，4分别表示label在右上角，左上角，左下角，右下角，这里会有一个小框框表示点代表什么，然后出现在左上角
# plt.show()

X = iris.data[:, :2]
Y = iris.target
print(X.shape)
# 逻辑回归模型
# C正则化强度的倒数，必须是一个大于0的浮点数，不填写默认1.0，即默认正则项与损失函数的比值是1：1。
# C越小，损失函数会越小，模型对损失函数的惩罚越重，正则化的效力越强，参数会逐渐被压缩得越来越小。
# 训练
lr = LogisticRegression(C=1e5)
lr.fit(X, Y)
print(X.shape)
# meshgrid函数生成两个网格矩阵
h = .02
# 最大值+0.5 最大值+0.5
x_min, x_max = X[:, 0].min() - .5, X[:, 0].max() + .5
y_min, y_max = X[:, 1].min() - .5, X[:, 1].max() + .5
# arange 返回一个起点，终点，固定步长的排列（x_min,x_max,0.2）
# 生成网格坐标矩阵，等于是这一步即做了网格，又搞了一个测试集
print(x_min, x_max, y_min, y_max)
xx, yy = np.meshgrid(np.arange(x_min, x_max, h), np.arange(y_min, y_max, h))
# 前面是y的差除以0.2乘10+1 后面则是x的差除以0.2乘10+1
print(xx.shape, "|", yy.shape)

#这里171X231=35901
# pcolormesh 函数将xx,yy两个网格矩阵和对应的预测结果Z绘制在图片上
# np.c_用于连接两个矩阵，根据下面的结果两个一维矩阵连接起来就是二维矩阵
# ravel主要用于把多维度矩阵转为1维度[[1,2],[3,4]];[1,2,3,4]
#Z是一个分类结果1111，22222这样的维度是
b=xx.ravel()
print(b.shape)
C=np.c_[xx.ravel(), yy.ravel()]
print(C.shape)#等于我构造了一个35901个测试集
Z = lr.predict(np.c_[xx.ravel(), yy.ravel()])
print(Z)
print(Z.shape)
# 将Z的矩阵格式何XX变成一致的,2维度矩阵
Z = Z.reshape(xx.shape)

print(Z)
print(Z.shape)
plt.figure(1, figsize=(8, 6))
#这步是绘制伪彩图
plt.pcolormesh(xx, yy,Z, cmap=plt.cm.Paired)

# pcolormesh()**函数用于创建具有非规则矩形网格的伪彩色图
#plt.figure(1, figsize=(8, 6))
#plt.pcolormesh(xx, yy, Z, cmap=plt.cm.Paired)

plt.scatter(X[:50, 0], X[:50, 1], color='red', marker='o', label='setosa')
plt.scatter(X[50:100, 0], X[50:100, 1], color='blue', marker='x', label='versicolor')
plt.scatter(X[100:, 0], X[100:, 1], color='green', marker='s', label='Virginica')

plt.xlabel('Sepal length')  # x轴
plt.ylabel('Sepal width')  # y轴
plt.xlim(xx.min(), xx.max())  # x轴坐标范围
plt.ylim(yy.min(), yy.max())  # y轴坐标范围
m=range(3,9,1)
plt.xticks((m)) # x轴刻度
plt.yticks(())  # y轴刻度
plt.legend(loc=2)
plt.show()
#xx 3.8-8.4
#yy 1.5，4.9
#ZZ
print(Z)
