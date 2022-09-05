import numpy as np 
import matplotlib.pyplot as plt
import cv2 as cv

img = cv.imread("./im/aka.jpg")
image = cv.cvtColor(img, cv.COLOR_BGR2RGB)
gray_image = cv.cvtColor(cv.cvtColor(img,cv.COLOR_BGR2GRAY), cv.COLOR_BGR2RGB)

plt.subplot(211)
plt.hist(gray_image.ravel(), bins=256, range=(0, 255))

plt.subplot(212)
histr = cv.calcHist([gray_image],[0],None,[256],[0,256])
plt.plot(histr)
plt.show()