import cv2
# Load image, convert to grayscale, Gaussian blur, Otsu's threshold
image = cv2.imread(
    '/Volumes/DATA/Sources/DocumentDigitalize/word_detector/WordDetector/data/gialai/4.png')
gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
blur = cv2.GaussianBlur(gray, (3, 3), 0)
thresh = cv2.threshold(
    blur, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)[1]

# Filter using contour area and remove small noise
cnts = cv2.findContours(thresh, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
cnts = cnts[0] if len(cnts) == 2 else cnts[1]
for c in cnts:
    area = cv2.contourArea(c)
    if area < 40:
        cv2.drawContours(thresh, [c], -1, (0, 0, 0), -1)

# Morph close and invert image
kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
close = 255 - cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, kernel, iterations=2)

cv2.imshow('thresh', thresh)
cv2.imshow('close', close)

cv2.imwrite(
    '/Volumes/DATA/Sources/DocumentDigitalize/word_detector/WordDetector/data/gialai/5.png', close)

cv2.waitKey()
