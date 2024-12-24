import cv2 as cv
import numpy as np
import matplotlib.pyplot as plt

img = cv.imread('photos/cat.jpg')

#****************************************
# rescale function
def rescalephoto(photo, scale = 1):
    width = int(photo.shape[1] * scale)
    height = int(photo.shape[0] * scale)
    dimensions = (width, height)

    return cv. resize(photo, dimensions, interpolation=cv.INTER_AREA)
img_resized = rescalephoto(img)

#Converting to grayscale
gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)

#Blur (using gaussian)
blur = cv.GaussianBlur(img, (3,3), cv.BORDER_DEFAULT)

#Edge Cascade using canny edge detection
canny = cv.Canny(img, 125, 175)

#dilating and eroding the image using canny as a structural element 
dilated = cv.dilate(canny, (3,3), iterations = 1)
eroded = cv.erode (canny, (3,3), iterations = 1)

#resize an image (interpolations: area, linear, cubic (use linear and cubic is upscaling, cubic  is slower but gives an image with a higher resolution))
resized = cv.resize(img, (500,500))

#cropping
cropped = img[0:225, 0:1]

#****************************************
#translation
def translate(img, x, y):
    transMat = np.float32([[1,0,x],[0,1,y]])
    dimensions = (img.shape[1], img.shape[0])
    return cv.warpAffine(img, transMat, dimensions)

translated = translate(img, 0, 100)

#Rotation
def rotate(img, angle, rotPoint=None):
    (height, width) = img.shape[:2] #first 2 values

    if rotPoint is None:
        rotPoint = (width//2, height//2)
    
    rotMat = cv.getRotationMatrix2D(rotPoint, angle, 1.0) #1.0 is rescale parameter
    dimensions = (width, height)

    return cv.warpAffine(img, rotMat, dimensions)

rotated = rotate(img, 90)

#flipping
flipped = cv.flip(img, 0) #0-> vertical, 1->horizontal, -1->both

#******************************************

#Contour Detection
img2 = cv.imread('photos/polygons.png')

#step 1: convert to grayscale
gray2 = cv.cvtColor(img2, cv.COLOR_BGR2GRAY)

#step 2: detect the edges using canny edge detector
canny2 = cv.Canny(img2, 125, 175)

#step 3: detect contours
contours, hierarchies = cv.findContours(canny2, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_NONE) #TREE-> all hierarchial contours, EXTERTNAL-> external contours only, LIST-> all contours

#step 4: draw contours
blank = np.zeros(img2.shape, dtype='uint8')
cv.drawContours(blank, contours, -1, (0,0,255), 1)

#*****************************************

#Avergaing using the all ones kernel (use gaussianBlur for gaussian filter)
average = cv.blur(img, (3,3))

#Median blur
median = cv.medianBlur(img, 3)

#bilateral blur
#****************************************

#masking
blank2 = np.zeros(img.shape[:2], dtype='uint8') 
mask = cv.circle(blank2, (img.shape[1]//2, img.shape[0]//2), 100, 255, -1)
masked = cv.bitwise_and(img, img, mask=mask)

#****************************************
#thresholding
grayed = cv.imread('photos/grayscale.jpg')
_, thresh = cv.threshold(grayed, 150, 255, cv.THRESH_BINARY)
#****************************************
#Histogram computation
img3 = cv.imread('photos/09.jpg')
gray3 = cv.cvtColor(img3, cv.COLOR_BGR2GRAY) 

gray_hist = cv.calcHist([gray3], [0], None, [256], [0,256])
plt.figure()
plt.title('histogram')
plt.xlabel('bins')
plt.ylabel('# of pixles')
plt.plot(gray_hist)
plt.xlim([0,256])
plt.show()

cv.imshow('cat', img3)
cv.waitKey(0)

