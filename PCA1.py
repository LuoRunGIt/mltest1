import numpy as np
import scipy.io as sci

import matplotlib.pyplot as plt

# scipy 包可以读取数据文件
# face格式是个字典
faces_data = sci.loadmat('./ex7faces.mat')
print(faces_data)
# X里存储的是一个人脸矩阵
X = faces_data['X']
print(X.shape)
print(X[1])


# 5000 张图片 1024个特征
# 每一行是一个样本，每一列是一个特征

def plot_100_image(X):
    # 这里有必要好好了解下figsize的效果，和整个subplots的功能
    # 一张画布，有10x10个格子，每个各自大小为10x10
    fig, ax = plt.subplots(nrows=10, ncols=10, figsize=(10, 10))
    for c in range(10):
        for r in range(10):
            # Grey_r 大约是灰色
            #
            ax[c, r].imshow(X[10 * c + r].reshape(32, 32).T, cmap='Greys_r')
            # 设置x刻度轴,实验效果可以看出是每张图片是32X32个像素的
            ax[c, r].set_xticks([1, 5, 10, 32])
            # 设置y刻度轴
            ax[c, r].set_yticks([])
    plt.show()


plot_100_image(X)


# PCA 步骤
# 1.去均值化
# 2，计算协方差矩阵
# 3.计算特征值和特征向量

def reduce_mean(X):
    # 对二维矩阵的列求均值
    X_reduce_mean = X - X.mean(axis=0)
    return X_reduce_mean


X_reduce_mean = reduce_mean(X)


# 计算协方差矩阵
def sigma_matrix(X_reduce_mean):
    sigma = (X_reduce_mean.T @ X_reduce_mean) / X_reduce_mean.shape[0]
    return sigma


sigma = sigma_matrix(X_reduce_mean)


# sigma是协方差矩阵

# svd奇异矩阵
def usv(sigma):
    u, s, v = np.linalg.svd(sigma)
    return u, s, v


# 也就是说这里的u已经是一个特征值构成的矩阵了
u, s, v = usv(sigma)
print("-----------显示U")
print(u)


# 这里u是一个1024*1024的矩阵，奇异矩阵的目的是把mn矩阵分解为2个nn和mm的矩阵
# 奇异分解的时候s矩阵就是特征值矩阵，其对角线大小就是从大到小排列的

# 求特征值和特征向量
# K 表示前100个特征
# u_reduce表示特性向量矩阵？
def project_data(X_reduce_mean, u, k):
    u_reduced = u[:, :k]
    # dot仅仅是点乘，求出的是一个矩阵
    z = np.dot(X_reduce_mean, u_reduced)
    # 这里是吧5000*1024的均值矩阵和1024*1024的PCA矩阵相乘？这一步不就等于已经降维了嘛
    return z


# z相当于5000个样本前100个特征

z = project_data(X_reduce_mean, u, 100)
print("--------Z---------")


# print(z)

# 还原数据？

# 这里是降维后的图像，数据降维
def recover_data(z, u, k):
    # 从0 到第k个矩阵
    u_reduced = u[:, :k]
    X_recover = np.dot(z, u_reduced.T)  # 5000*100 和100*1024做矩阵乘法
    return X_recover


X_recover = recover_data(z, u, 100)

# 显示降维后的图像
plot_100_image(X_recover)
print(X_reduce_mean.shape)  # 5000*1024
print(sigma.shape)  # 1024*1024
print(u.shape)  # 1024*1024
print(z.shape)  # 5000,100
print(X_recover.shape)  # 5000*1024

# 降维是原数据集*降维矩阵，也就是5000*1024 与一个1024*1024的矩阵做点乘
