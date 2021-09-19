from contextlib import closing
import cv2

gray = cv2.imread('D:\\HIPT\\WordDetector\\data\\gialai\\2.png')

gray = cv2.cvtColor(gray, cv2.COLOR_BGR2GRAY)

(_, gray) = cv2.threshold(gray, 200, 255, cv2.THRESH_BINARY_INV)

gray = (255 - gray)
cv2.imshow("gray", gray)

kernelSize = (1, 3)

kernel = cv2.getStructuringElement(cv2.MORPH_RECT, kernelSize)
gray = cv2.morphologyEx(gray, cv2.MORPH_CLOSE, kernel)
# cv2.imshow("Opening: ({}, {})".format(
#     kernelSize[0], kernelSize[1]), opening)

cv2.imshow("close 1 3", gray)

cv2.imwrite('D:\\HIPT\\WordDetector\\data\\gialai\\3.png', gray)

# closeSize = (3,3)
# close_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, closeSize)
# closing = cv2.morphologyEx(gray, cv2.MORPH_CLOSE, close_kernel)
# cv2.imshow("Closing", closing)
# cv2.imwrite('D:\\HIPT\\WordDetector\\data\\gialai\\2.png', closing)

cv2.waitKey(0)