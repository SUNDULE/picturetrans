import cv2 as cv
import numpy as np
import matplotlib.pyplot as plt

imgBGR = cv.imread("D:\dog.jpg")
print(imgBGR.shape)

imgRGB = cv.cvtColor(imgBGR, cv.COLOR_BGR2RGB)
imgGRAY = cv.cvtColor(imgBGR, cv.COLOR_BGR2GRAY)
imgHSV = cv.cvtColor(imgBGR, cv.COLOR_BGR2HSV)
imgYCrCb = cv.cvtColor(imgBGR, cv.COLOR_BGR2YCrCb)
imgHLS = cv.cvtColor(imgBGR, cv.COLOR_BGR2HLS)
imgXYZ = cv.cvtColor(imgBGR, cv.COLOR_BGR2XYZ)
imgLAB = cv.cvtColor(imgBGR, cv.COLOR_BGR2LAB)
imgYUV = cv.cvtColor(imgBGR, cv.COLOR_BGR2YUV)

titles = ['BGR', 'RGB', 'GRAY', 'HSV', 'YCrCb', 'HLS', 'XYZ', 'LAB', 'YUV']
images = [imgBGR, imgRGB, imgGRAY, imgHSV, imgYCrCb,imgHLS, imgXYZ, imgLAB, imgYUV]
plt.figure(figsize=(10, 8))
for i in range(9):
    plt.subplot(3, 3, i + 1), plt.imshow(images[i], 'gray')
    plt.title(titles[i])
    plt.xticks([]), plt.yticks([])
plt.tight_layout()
plt.show()
