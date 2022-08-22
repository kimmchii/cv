import cv2 as cv
import matplotlib.pyplot as plt
img = cv.resize(cv.imread("/home/kjm/Pictures/Webcam/2022-08-15-101112.jpg"), (640,360))

#assignment 1.1: split the image by using normal method
# split the image from split function and merge them together
b1 = img[:,:,0]
g1 = img[:,:,1]
r1 = img[:,:,2]
cv.imshow("blue", b1)
cv.imshow("green", g1)
cv.imshow("red", r1)
cv.waitKey(0)
cv.destroyAllWindows()
#assignment 1.2: split the image by using split function
# split the image by using the traditional array property and merge them with the function
b,g,r = cv.split(img)
print("compare blue \n", b1==b)
print("compare green \n", g1==g)
print("compare red \n", r1==r)
cv.imshow("blue", b)
cv.imshow("green", g)
cv.imshow("red", r)
cv.waitKey(0)
cv.destroyAllWindows()


#assignment 1.3: merge the image using merge function 
new_img1 = cv.merge((b1,g1,r1))
new_img = cv.merge((b,g,r))
print("compare new_img", new_img1==new_img)
cv.imshow("new_image1", new_img1)
cv.imshow("new_image", new_img)
cv.waitKey(0)
cv.destroyAllWindows()

# cv.imshow("new_img", new_img)
# cv.imshow("new_img1", new_img1)
# cv.waitKey(0)
# cv.destroyAllWindows()
