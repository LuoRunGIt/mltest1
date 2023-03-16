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
# ax[c, r].imshow(X[10 * c + r].reshape(32, 32).T, cmap='Greys_r')

# fig, ax = plt.subplots(nrows=10, ncols=10, figsize=(10, 10))
# for c in range(10):
# for r in range(10):
#
#
#       ax[c, r].imshow(X[10 * c + r].reshape(32, 32).T, cmap='Greys_r')
#        # 设置x刻度轴,实验效果可以看出是每张图片是32X32个像素的
#        ax[c, r].set_xticks([1, 5, 10, 32])
# 设置y刻度轴
#        ax[c, r].set_yticks([])
# plt
fig, ax = plt.subplots(nrows=2, ncols=2, figsize=(10, 10))
ax[0, 0].imshow(X[4999].reshape(32, 32).T, cmap='Greys_r')
plt.show()
print(X[0].shape)
print(X[0])
#为何一个1维度的矩阵能表示图像呢
#【计算机视觉 Python】1.图像数据基本概念
#这里实际上是将X[i]作为解析图像的一个参数