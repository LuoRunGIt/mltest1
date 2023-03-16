from mpl_toolkits.mplot3d import Axes3D
import pandas as pd
import sklearn.cluster  # 这里注意导入包的准确性，只导入sklearn是错误的
import matplotlib.pyplot as plt
# 虽然不知道为啥，打开虽然是乱码，但是使用函数进行分析还是可以获得完整的值
data = pd.read_csv("data/football.csv")
# x_train=data[["国籍","2019","2018","2015"]]
x_train = data[["2019", "2018", "2015"]]
# print(x_train)
# 这步等于是去除了表格的第一列，使得表格成为一个纯矩阵
df = pd.DataFrame(x_train)
# 分成3个簇
kmeans = sklearn.cluster.KMeans(n_clusters=3, n_init="auto")
# 归一化
# 这里做的是数据压缩，默认范围是归一化，下一句是压缩目标的范围，然后在进行数据本身范围的压缩
min_max_matrix = sklearn.preprocessing.MinMaxScaler()
x_train = min_max_matrix.fit_transform(x_train)
# print(x_train)

# 下一步是训练
kmeans.fit(x_train)
predict_y = kmeans.predict(x_train)
# print(predict_y)
# predict_y[1 0 0 0 0 1 2 1 1 2 2 2 2 2 2 0 2]
# 这里表示的是聚类出的簇的结果
#
result = pd.concat((data, pd.DataFrame(predict_y)), axis=1)
result.rename({0: u'聚类'}, axis=1, inplace=True)
print(result)
# axis 1 简单理解是数组横向输出

#这里我没有做中文化处理，原来的实验中有相关的处理
#这句是创建一个空三维坐标
#ax = plt.subplot(projection='3d')
plt.figure(figsize=(12,8))
ax = plt.subplot(projection='3d')
x = result['2019']
y = result['2018']
z = result['2015']
ax.scatter3D(x,y,z,c=predict_y,s=100,alpha=1)
ax.set_xlabel('2019')
ax.set_ylabel('2018')
ax.set_zlabel('2015')
plt.show()

#整个实验参考了2篇博客