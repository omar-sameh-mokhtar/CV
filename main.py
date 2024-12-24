import cv2
import numpy as np
import matplotlib.pyplot as plt
import math

def blur_image(image):
    blur = cv2.GaussianBlur(image, (5,5), cv2.BORDER_DEFAULT)
    return image

def detect_salt_and_pepper(image):
    
    height, width = image.shape[:2]
    noise_pixels = 0
    if len(image.shape) == 3:
        image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    _, image = cv2.threshold(image, 0, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)

    for i in range(1, height - 1):  
        for j in range(1, width - 1):  
            
            center_pixel = image[i, j]
            
            surrounding_pixels = [
                image[i-1, j-1], image[i-1, j], image[i-1, j+1], image[i, j-1], 
                image[i, j+1], image[i+1, j-1], image[i+1, j], image[i+1, j+1]] 
            
            x=0
            for k in range(0,7): 
                if surrounding_pixels[k] == center_pixel:
                    x+=1
                else:
                    x-=1

            if x < 0:
                noise_pixels += 1
            #if center_pixel != surrounding_pixels[0] or center_pixel != surrounding_pixels[1] or \
            #   center_pixel != surrounding_pixels[2] or center_pixel != surrounding_pixels[3] or \
            #   center_pixel != surrounding_pixels[4] or center_pixel != surrounding_pixels[5] or \
            #   center_pixel != surrounding_pixels[6] or center_pixel != surrounding_pixels[7]:
            #    noise_pixels += 1  

    image_size = height*width
    percentage_noise = 100*noise_pixels / (height*width)
    print(f"Total pixels in image:  {image_size}")
    print(f"Total noisy pixels: {noise_pixels}")
    print(f"Salt and Pepper approximate percentage: {percentage_noise} %")

    return percentage_noise > 2 

def add_salt_and_pepper(image):
    noiseProb = 0.15
    noise = np.random.rand(image.shape[0], image.shape[1])
    image[noise < noiseProb/2] = 0
    image[noise > 1 - noiseProb/2] = 255

    return image

def fix_salt_and_pepper(image): 
    
    #kernel = np.array([[1/5], [1/5], [1/5]])
    #image = cv2.filter2D(image,-1,kernel)
    image = cv2.bilateralFilter(image, d=9, sigmaColor=75, sigmaSpace=75)
    display(image, "bi")
    _, image = cv2.threshold(image, 0, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU) 

    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3)) 
    image = cv2.morphologyEx(image, cv2.MORPH_CLOSE, kernel, iterations = 1)
    
    #kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (1, 10)) 
    #image = cv2.morphologyEx(image, cv2.MORPH_CLOSE, kernel, iterations = 3)
    #display(image, "1")
    
    #kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (2, 1))
    #image = cv2.morphologyEx(image, cv2.MORPH_CLOSE, kernel)
    display(image, "2")
    
    return image

def change_brightness(image):
    min_value = 50
    max_value = 200
    image = np.clip(image, 0, 255).astype(np.uint8)

    stretched_image = np.uint8((image / 255) * (max_value - min_value) + min_value)
    
    return stretched_image

def get_thresh_value(img): 
    hist = cv2.calcHist([img], [0], None, [256], [0, 256])
    peaks = np.argsort(hist.flatten())[-2:]
    thresh_value = (peaks[0]+peaks[1])/2

    _, thresholded = cv2.threshold(img, thresh_value+5, 255, cv2.THRESH_BINARY)
    plt.figure()
    plt.plot(hist)
    plt.scatter(peaks, hist[peaks], color='r', label='Peaks')
    plt.title("Histogram with Peaks")
    plt.legend()
    plt.show()
    return thresholded#, thresh_value

def display(image, name):
    cv2.imshow(name, image)
    cv2.waitKey(0)

def detect_distance_between_lines(gray):

    _, binary_image = cv2.threshold(gray, 150, 255, cv2.THRESH_BINARY_INV)  

    edges = cv2.Canny(binary_image, 50, 150, apertureSize=3)
    
    lines = cv2.HoughLinesP(edges, 1, np.pi / 180, threshold=100, minLineLength=100, maxLineGap=10)

    line_image = np.copy(gray)

    if lines is not None:
        # Sort lines by x
        lines = sorted(lines, key=lambda x: x[0][0])

        distances = []
        maxdistance = 0

        for i in range(1, len(lines)):
            # Get the x of the start and end points of  lines
            x1, y1, x2, y2 = lines[i-1][0] #2 point le awl line w 2 le tany line
            x3, y3, x4, y4 = lines[i][0]

            if abs(x3 - x1) == abs(x4 - x2): #to ignore some irregular lines
                distance = abs(x3 - x1)
                if distance > maxdistance:
                    maxdistance = distance
                distances.append(distance)
            
            cv2.line(line_image, (x1, y1), (x2, y2), 1)  
            cv2.line(line_image, (x3, y3), (x4, y4), 1)
        #cv2.imshow('Detected Lines', line_image)
        #cv2.waitKey(0)

        print(f"Distances between barcode bars: {distances}")
        return  maxdistance

    else:
        print("No lines detected.")
        return  -1

def detectBarcode(img):
    maxdistance = detect_distance_between_lines(img)
    print(maxdistance)
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (maxdistance+1, 1)) #+1 3shan maxdistance is 1 pixel short
    closed_image = cv2.morphologyEx(img, cv2.MORPH_OPEN, kernel)

    _, closed_image_2 = cv2.threshold(closed_image, 150, 255, cv2.THRESH_BINARY_INV)
    cv2.imshow('imagle', closed_image_2)
    cv2.waitKey(0)
    contours, _ = cv2.findContours(closed_image_2, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    largest_contour = None
    max_area = 0

    for contour in contours:
        area = cv2.contourArea(contour)
        if area > max_area:
            max_area = area
            largest_contour = contour

    x, y, deltax, deltay = cv2.boundingRect(largest_contour)
    #print(x,y,deltax,deltay)
    mabrook = img[y:y+deltay, x:x+deltax]
    display(mabrook,"ok")
    return mabrook

def detect_contrast(image):

    mean, stddev = cv2.meanStdDev(image)

    mean = mean[0][0]
    stddev = stddev[0][0]

    print(f"Mean intensity: {mean}")
    print(f"Standard Deviation: {stddev}")

    contrast_ratio = stddev / mean

    print(f"Contrast Ratio (stddev / mean): {contrast_ratio}")

    if contrast_ratio < 0.1:  # Example threshold for low contrast
        print("The image has low contrast.")
        return True  # Low contrast
    else:
        print("The image has sufficient contrast.")
        return False  # High contrast

def fix_contrast(gray):
    
    min_intensity = np.min(gray)
    max_intensity = np.max(gray)

    stretched_image = (gray - min_intensity) * (255 / (max_intensity - min_intensity))

    return stretched_image.astype(np.uint8)

def decode(out):
    # 0 means narrow, 1 means wide
    NARROW = "0"
    WIDE = "1"
    code11_widths = {
        "00110": "Stop/Start",
        "10001": "1",
        "01001": "2",
        "11000": "3",
        "00101": "4",
        "10100": "5",
        "01100": "6",
        "00011": "7",
        "10010": "8",
        "10000": "9",
        "00001": "0",
        "00100": "-",
    }

    # Get the average of each column in your image
    mean = out.mean(axis=0)
    
    # Set it to black or white based on its value
    mean[mean <= 127] = 1
    mean[mean > 128] = 0

    # Convert to string of pixels in order to loop over it
    pixels = ''.join(mean.astype(np.uint8).astype(str))

    # Need to figure out how many pixels represent a narrow bar
    narrow_bar_size = 0
    found_a_one = 0
    for pixel in pixels:
        if pixel == "0" and found_a_one == 0:
            continue
        if pixel == "1":
            found_a_one = 1
            narrow_bar_size += 1
        else:
            break

    wide_bar_size = narrow_bar_size * 2
    
    print(f"narrow: {narrow_bar_size}, wide: {wide_bar_size}")

    print(mean)

    digits = []
    pixel_index = 0
    current_digit_widths = ""
    skip_next = False

    pixels = pixels.lstrip('0')
    print(pixels)
    while pixel_index < len(pixels):

        if skip_next:
            pixel_index += narrow_bar_size
            skip_next = False
            continue

        count = 1
        try:
            while pixels[pixel_index] == pixels[pixel_index + 1]:
                count += 1
                pixel_index += 1
        except:
            pass
        pixel_index += 1

        tolerance = narrow_bar_size // 2

        current_digit_widths += NARROW if narrow_bar_size - tolerance <= count <= narrow_bar_size + tolerance else WIDE

        if current_digit_widths in code11_widths:
            digits.append(code11_widths[current_digit_widths])
            current_digit_widths = ""
            skip_next = True  # Next iteration will be a separator, so skip it

    print(digits)

def detect_rotation(image): 
    
    edges = cv2.Canny(image, 50, 200)
    contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    largest_contour = max(contours, key=cv2.contourArea)
    min_rect = cv2.minAreaRect(largest_contour)

    angle = min_rect[2]
    points = cv2.boxPoints(min_rect)
    print(points)
    #the 4 corner points are ordered clockwise starting from the point with the highest y (aktar no2ta south).
    #so the rule is to compare point 0 to point 1 and 3 to determine if its 45 or -45 (kda / walla kda \)

    point0 = points[0]
    point1 = points[1]
    point3 = points[3]
    
    distance1 = np.linalg.norm(point1 - point0)
    distance2 = np.linalg.norm(point3 - point0)

    print("d", angle)
    print("d1", distance1, "d2", distance2)
    return angle, distance1 < distance2

def rotate(img, angle, rotPoint=None):
    (height, width) = img.shape[:2] #first 2 values

    if rotPoint is None:
        rotPoint = (width//2, height//2)
    
    rotMat = cv2.getRotationMatrix2D(rotPoint, angle, 1.0) #1.0 is rescale parameter
    dimensions = (width, height)

    return cv2.warpAffine(image, rotMat, dimensions, flags=cv2.INTER_CUBIC, borderMode=cv2.BORDER_REPLICATE)

def extract_grayscale_regions_hsv(input_img, saturation_threshold=30):
    """ Extract grayscale regions from an image using low saturation in the HSV color space."""
    # Convert the image to HSV color space
    hsv_img = cv2.cvtColor(input_img, cv2.COLOR_BGR2HSV)
    
    # Split the HSV channels
    h, s, v = cv2.split(hsv_img)
    
    # Create a mask for low-saturation regions
    low_saturation_mask = s < saturation_threshold
    
    # Convert the mask to binary
    binary_mask = low_saturation_mask.astype(np.uint8) * 255
    
    # Create a white background
    white_background = np.ones_like(input_img, dtype=np.uint8) * 255 
    
    # Use the mask to isolate low-saturation regions (grayscale)
    grayscale_regions = cv2.bitwise_and(input_img, input_img, mask=binary_mask)
    
    # Invert the mask
    inverted_mask = cv2.bitwise_not(binary_mask)
    grayscale_regions += cv2.bitwise_and(white_background, white_background, mask=inverted_mask)
    
    return grayscale_regions

def Handle_rotation(image):
    
    padded_image = cv2.copyMakeBorder(image, 50, 50, 50, 50, cv2.BORDER_CONSTANT, value=(255, 255, 255))
    angle, pos = detect_rotation(padded_image)

    #pos means 2 things, either 45(/) or barcode is horizontal if angle is 90
    print("pos: ", pos)
    if angle == 90:
        if pos:
            return image
        else:
            image = rotate(image, -90)
            return image

    (h, w) = padded_image.shape[:2]
    center = (w // 2, h // 2)

    if pos:
        angle-=90

    M = cv2.getRotationMatrix2D(center, angle, 1.0)
    
    image = cv2.warpAffine(padded_image, M, (w, h), flags=cv2.INTER_CUBIC, borderMode=cv2.BORDER_REPLICATE)

    
    cv2.imshow('barcode2', image)
    cv2.waitKey(0)

    return image

def preprocess(image):
    isolated = extract_grayscale_regions_hsv(image, saturation_threshold=30)
    gray = cv2.cvtColor(isolated, cv2.COLOR_BGR2GRAY)
    #gray = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 31, 9)
    display(gray, "")
    kernel = np.array([[1/3], [1/3], [1/3]])
    blurred = cv2.filter2D(gray,-1,kernel)

    #blurred = add_salt_and_pepper(blurred)
    #blurred = change_brightness(blurred)
    #display(blurred, "")

    if detect_salt_and_pepper(blurred):
        blurred = fix_salt_and_pepper(blurred)

    _, binary_img = cv2.threshold(blurred, 0, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)

    #if detect_salt_and_pepper(binary_img):
    #    binary_img = fix_salt_and_pepper(binary_img)
    
    #binary_img = get_thresh_value(blurred)
    #if detect_contrast(blurred):
    #    print("nooo")
    #    binary_img = fix_contrast(blurred)
    #else:

    binary_img = Handle_rotation(binary_img)

    barcode = detectBarcode(binary_img)

    w, h = barcode.shape[:2]
    
    blured_barcode = cv2.medianBlur(cv2.blur(barcode,(1,h)),1)
    _, binary_barcode = cv2.threshold(blured_barcode, 200, 255, cv2.THRESH_BINARY)
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (1, h)) 
    final = cv2.morphologyEx(binary_barcode, cv2.MORPH_OPEN, kernel)
    
    return final



image = cv2.imread('photos/07.jpg')
Final = preprocess(image)

plt.subplot(1, 2, 1)
plt.imshow(image, cmap='gray')
plt.title('Test Case')
plt.axis('off')

plt.subplot(1, 2, 2)
plt.imshow(Final, cmap='gray')
plt.title('final_result')
plt.axis('off')

plt.show()
display(Final, "")
decode(Final)

#add_salt_and_pepper(gray)
#rotate(image, angle)
#change_brightness(image)