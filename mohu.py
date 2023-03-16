# import the necessary packages
import os

from imutils import paths
import argparse
import cv2

#image = "C:\\Users\\Administrator\\Pictures\\人脸\\duo.jpg"


def getPhotopath(paths):
    imgfile = []
    #【xxx.xx ,xxx.xx】
    file_list = os.listdir(paths)
    for i in file_list:
        newph = os.path.join(paths, i)
        imgfile.append(newph)
    return imgfile

def variance_of_laplacian(image):
    # compute the Laplacian of the image and then return the focus
    # measure, which is simply the variance of the Laplacian
    # 这是核心算法
    return cv2.Laplacian(image, cv2.CV_64F).var()


# construct the argument parse and parse the arguments，这个地方等于是用来使用命令行
# ap = argparse.ArgumentParser()
# ap.add_argument("-i", "--images", required=True,
#                help="path to input directory of images")
# ap.add_argument("-t", "--threshold", type=float, default=100.0,
#                help="focus measures that fall below this value will be considered 'blurry'")

# args = vars(ap.parse_args())

# 这里我重写下上面的安排
path="./mouhu2"

args=getPhotopath(path)
print(args)
# loop over the input images
for imagePath in args:
    # load the image, convert it to grayscale, and compute the
    # focus measure of the image using the Variance of Laplacian
    # method
    image = cv2.imread(imagePath)
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    fm = variance_of_laplacian(gray)
    text = "Not Blurry"

# if the focus measure is less than the supplied threshold,
# then the image should be considered "blurry"
    if fm < 100:
      text = "Blurry"

# show the image
    cv2.putText(image, "{}: {:.2f}".format(text, fm), (10, 30),
            cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 3)
    cv2.imshow("Image", image)
#key = cv2.waitKey(0)

# 现在我的理解是先设置一个阈值，这个阈值可以通过已知的一个图片的计算出的值来进行规定
# CV2库为一个图像处理库

