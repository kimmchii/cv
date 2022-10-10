import cv2 as cv

img = cv.imread("/home/kjm/Pictures/Screenshots/Screenshot from 2022-09-27 22-13-46.png")

cv.imshow("", img)
cv.waitKey(0)
cv.destroyAllWindows()