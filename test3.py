import cv2
filePath="./mouhu2/001.jpg"
image = cv2.imread(filePath)
print(image)
cv2.imshow("sourcePic", image)
gray1 = cv2.imread(filePath, cv2.IMREAD_GRAYSCALE)
cv2.imshow("read2gray", gray1)
print(gray1)
cv2.waitKey(0)
cv2.destroyAllWindows()
