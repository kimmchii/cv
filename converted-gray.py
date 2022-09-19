import cv2 as cv
import matplotlib.pyplot as plt

#-------------Function----------------------------------
img = cv.imread("./im/aka.jpg")
image = cv.cvtColor(img, cv.COLOR_BGR2RGB)
gray_image = cv.cvtColor(cv.cvtColor(img,cv.COLOR_BGR2GRAY), cv.COLOR_BGR2RGB)


#-------------Manual------------------------------------ 
img1 = img.copy()

gray_b = 0.114*img1[:,:,0]
gray_g = 0.587*img1[:,:,1]
gray_r = 0.299*img1[:,:,2]
gray = gray_b + gray_g + gray_r
img1[:,:,0] = gray_b + gray_g + gray_r
img1[:,:,1] = gray_b + gray_g + gray_r
img1[:,:,2] = gray_b + gray_g + gray_r
img1 = cv.cvtColor(img1, cv.COLOR_BGR2RGB)


plt.subplot(131)
plt.imshow(image)

plt.subplot(132)
plt.imshow(gray_image)

plt.subplot(133)
plt.imshow(img1)

plt.show()


