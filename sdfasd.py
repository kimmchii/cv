import cv2 as cv
import matplotlib.pyplot as plt
img = cv.resize(cv.imread("/home/kjm/Pictures/Webcam/2022-08-15-101112.jpg"), (640,360))

# split the image from split function and merge them together
b,g,r = cv.split(img)
new_img = cv.merge((b,g,r)) 

# split the image by using the traditional array property and merge them with the function
b1 = img[:,:,0]
g1 = img[:,:,1]
r1 = img[:,:,2]
new_img1 = cv.merge((b1,g1,r1))

print("compare blue \n", b1==b)
print("compare green \n", g1==g)
print("compare red \n", r1==r)
print("compare new_img", new_img1==new_img)
# cv.imshow("new_img", new_img)
# cv.imshow("new_img1", new_img1)
# cv.waitKey(0)
# cv.destroyAllWindows()

plt.subplot(211)
plt.imshow(new_img, "gray")
plt.subplot(212)
plt.imshow(new_img1, "gray")
plt.show()