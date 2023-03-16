# canny算法学习
# 算法总共分为5步
# 1.把彩图转换为灰度图（这个不知道是不是必须）
# 2.高斯模糊
# 3.Sobel梯度算子
# 4.非极大值抑制
# 5.双阈值检测算法和连接边缘

from scipy import ndimage
# scipy是个科学计算库可以理解为和numpy类似得存在
from scipy.ndimage import convolve
# 用了scipy还是用了np 我giao
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import os


# 图像灰度化，这里我觉得应该有直接的库
def rgb2gray(rgb):
    r, g, b = rgb[:, :, 0], rgb[:, :, 1], rgb[:, :, 2]
    # 这里可以看出彩色图像的色彩三通道
    gray = (0.2989 * r + 0.5870 * g + 0.1140 * b)
    return gray


# 文件读取,好家伙这里还是个批处理灰度图嘞
def loaddata(dir_name):
    imgs = []
    for filename in os.listdir(dir_name):
        if os.path.isfile(dir_name + '/' + filename):
            img = mpimg.imread(dir_name + '/' + filename)
            img = rgb2gray(img)
            imgs.append(img)
    return imgs


# 可视化，简单来说把处理后的图像显示
def visualize(imgs, format=None, gray=False):
    plt.figure(figsize=(200, 200))
    # enumerate这个函数相当于返回一个索引和索引对应的值
    # 比如0，hi；1，hello
    for i, img in enumerate(imgs):
        if img.shape[0] == 3:
            # transpose的目标是交换rgb的位置，因为计算机不是rgb二十brg
            img = img.transpose(1, 2, 0)  # 这几段还得分析分析
            plt_idx = i + 1
            plt.subplot(2, 2, plt_idx)
            plt.imshow(img, format)  # 这里format是none
    plt.show()


# python对象也是个必要的知识点
class cannyEdgeDetector:
    def __init__(self, imgs, sigma=1, kernel_size=5, weak_pixel=75, strong_pixel=255, lowthreshold=0.05,
                 highthreshold=0.15):
        self.imgs = imgs
        # 处理完成的图像
        self.imgs_final = []
        self.img_smoothed = None  # 图像平滑？
        self.gradientMat = None  # 图像灰度？
        self.thetaMat = None
        self.nonMaxImg = None
        self.thresholdImg = None
        self.weak_pixel = weak_pixel
        self.strong_pixel = strong_pixel
        self.sigma = sigma
        self.kernel_size = kernel_size
        self.lowThreshold = lowthreshold
        self.highThreshold = highthreshold
        return

    # 原实验的sigma选择的是1，但是我之前做的实验sigma选的是2
    def gaussian_kernel(self, size, sigma=1):
        size = int(size)  # 这里keneral size取值应该是5
        x, y = np.mgrid[-size:size + 1, -size: size + 1]
        normal = 1 / (2.0 * np.pi * sigma ** 2)
        g = np.exp(-((x ** 2 + y ** 2) / (2.0 * sigma ** 2))) * normal  # exp是e
        return g

    # 这里的g显然只是一个值而不是一个矩阵

    # sobel 检测图片中的水平，垂直，和边缘
    def sobel_filters(self, img):
        # 分别是水平和垂直sobel
        Kx = np.array([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]], np.float32)
        Ky = np.array([[1, 2, 1], [0, 0, 0], [-1, -2, -1]], np.float32)

        # convolve直接求卷积
        Ix = ndimage.convolve(img, Kx)
        Iy = ndimage.convolve(img, Ky)

        # hypot 相当于相平方和根号
        G = np.hypot(Ix, Iy)
        G = G / G.max() * 255
        # arctan2 值域的范围是（-pi，pi）
        theta = np.arctan2(Iy, Ix)
        return (G, theta)

    # 非极大值抑制，D是梯度值,这里没有跑通，这里我有个问题，产生负角度是怎么处理的
    def non_max_suppression(self, img, D):
        M, N = img.shape
        Z = np.zeros((M, N), dtype=np.int32)  # 初始化一个新的矩阵
        angle = D * 180. / np.pi
        angle[angle < 0] += 180
        # 对夹角的处理，将弧度转换为角度
        for i in range(1, M - 1):
            for j in range(1, N - 1):
                try:
                    q = 255
                    r = 255

                    # angle 0，这里指的是角度
                    if (0 <= angle[i, j] < 22.5) or (157.5 <= angle[i, j] <= 180):
                        q = img[i, j + 1]
                        r = img[i, j - 1]
                    # angle 45
                    elif (22.5 <= angle[i, j] < 67.5):
                        q = img[i + 1, j - 1]
                        r = img[i - 1, j + 1]

                    elif (67.5 <= angle[i, j] < 112.5):
                        q = img[i + 1, j]
                        r = img[i - 1, j]
                    # angle 135
                    elif (112.5 <= angle[i, j] < 157.5):
                        q = img[i - 1, j - 1]
                        r = img[i + 1, j + 1]

                    if (img[i, j] >= q) and (img[i, j] >= r):
                        Z[i, j] = img[i, j]
                    else:
                        Z[i, j] = 0

                except IndexError as e:
                    pass

        return Z

    # 双阈值检测算法和边缘滤值
    # threshold 阈值函数，双阈值提取强边缘和若边缘
    def threshold(self, img):

        highThreshold = img.max() * self.highThreshold
        lowThreshold = highThreshold * self.lowThreshold

        M, N = img.shape
        res = np.zeros((M, N), dtype=np.int32)

        weak = np.int32(self.weak_pixel)
        strong = np.int32(self.strong_pixel)

        strong_i, strong_j = np.where(img >= highThreshold)
        zeros_i, zeros_j = np.where(img < lowThreshold)

        weak_i, weak_j = np.where((img <= highThreshold) & (img >= lowThreshold))

        res[strong_i, strong_j] = strong
        res[weak_i, weak_j] = weak

        return (res)

    # 最终的处理效果
    def hysteresis(self, img):

        M, N = img.shape
        weak = self.weak_pixel
        strong = self.strong_pixel

        for i in range(1, M - 1):
            for j in range(1, N - 1):
                if (img[i, j] == weak):
                    try:
                        if ((img[i + 1, j - 1] == strong) or (img[i + 1, j] == strong) or (img[i + 1, j + 1] == strong)
                                or (img[i, j - 1] == strong) or (img[i, j + 1] == strong)
                                or (img[i - 1, j - 1] == strong) or (img[i - 1, j] == strong) or (
                                        img[i - 1, j + 1] == strong)):
                            img[i, j] = strong
                        else:
                            img[i, j] = 0
                    except IndexError as e:
                        pass

        return img
        # 最后是一个批处理函数

    def detect(self):
        imgs_final = []
        for i, img in enumerate(self.imgs):
            self.img_smoothed = convolve(img, self.gaussian_kernel(self.kernel_size, self.sigma))
            self.gradientMat, self.thetaMat = self.sobel_filters(self.img_smoothed)
            self.nonMaxImg = self.non_max_suppression(self.gradientMat, self.thetaMat)
            self.thresholdImg = self.threshold(self.nonMaxImg)
            img_final = self.hysteresis(self.thresholdImg)
            self.imgs_final.append(img_final)

        return self.imgs_final


imgs = loaddata(dir_name='./mouhu2')
visualize(imgs, 'gray')
detector = cannyEdgeDetector(imgs, sigma=1, kernel_size=5, lowthreshold=0.09, highthreshold=0.17, weak_pixel=100)
imgs_final = detector.detect()
visualize(imgs_final, 'gray')
