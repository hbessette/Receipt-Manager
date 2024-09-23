import cv2
import numpy as np
from matplotlib import pyplot as plt
import pytesseract
from PIL import Image
import os

pytesseract.pytesseract.tesseract_cmd = 'C:/Program Files/Tesseract-OCR/tesseract.exe'

def display(img_path :str) -> str:
    dpi = 80
    img_data = plt.imread(img_path)
    height, width = img_data.shape[:2]

    figsize = width / float(dpi), height / float(dpi)

    fig = plt.figure(figsize=figsize)
    ax = fig.add_axes([0, 0, 1, 1])

    ax.axis('off')
    ax.imshow(img_data, cmap='gray')

    plt.show()

def invert_image(img_path : str) -> str:
    img = cv2.imread(img_path)
    inverted_image = cv2.bitwise_not(img)
    new_img_path = "data/temp" + img_path[img_path.rindex("/"):len(img_path) - 4] + "_inverted.jpg"
    cv2.imwrite(new_img_path, inverted_image)
    return new_img_path

def grayscale(img_path : str) -> str:
    img = cv2.imread(img_path)
    gray_img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    new_img_path = "data/temp" + img_path[img_path.rindex("/"):len(img_path) - 4] + "_grayscaled.jpg"
    cv2.imwrite(new_img_path, gray_img)
    return new_img_path

def gaussian_blur(img_path : str) -> str:
    img = cv2.imread(img_path)
    blurred_img = cv2.GaussianBlur(img, (7, 7), 0)
    new_img_path = "data/temp" + img_path[img_path.rindex("/"):len(img_path) - 4] + "_gaussian.jpg"
    cv2.imwrite(new_img_path, blurred_img)
    return new_img_path

def noise_removal(img_path : str) -> str:
    kernel = np.ones((1, 1), np.uint8)
    img = cv2.imread(img_path)
    img = cv2.dilate(img, kernel, iterations=1)
    kernel = np.ones((1, 1), np.uint8)
    img = cv2.erode(img, kernel, iterations=1)
    img = cv2.morphologyEx(img, cv2.MORPH_CLOSE, kernel)
    img = cv2.medianBlur(img, 3)
    new_img_path = "data/temp" + img_path[img_path.rindex("/"):len(img_path) - 4] + "_noiseremoval.jpg"
    cv2.imwrite(new_img_path, img)
    return new_img_path

def erode(img_path : str) -> str:
    img = cv2.imread(img_path)
    img = cv2.bitwise_not(img)
    kernel = np.ones((2,2), np.uint8)
    img = cv2.erode(img, kernel, iterations=2)
    img = cv2.bitwise_not(img)
    new_img_path = "data/temp" + img_path[img_path.rindex("/"):len(img_path) - 4] + "_eroded.jpg"
    cv2.imwrite(new_img_path, img)
    return new_img_path

def getSkewAngle(img_path : str) -> float:
    img = cv2.imread(img_path)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    blur = cv2.GaussianBlur(gray, (9, 9), 0)
    thresh = cv2.threshold(blur, 180, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)[1]

    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (30, 5))
    dilate = cv2.dilate(thresh, kernel, iterations=1)
    contours, hierarchy = cv2.findContours(dilate, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
    contours = sorted(contours, key = cv2.contourArea, reverse = True)
    largestContour = contours[0]
    for c in contours:
        if cv2.boundingRect(largestContour)[3] > cv2.boundingRect(largestContour)[2]:
            largestContour = c
        rect = cv2.boundingRect(c)
        x,y,w,h = rect
        cv2.rectangle(img, (x,y), (x+w,y+h), (0,255,0), 2)
        
    minAreaRect = cv2.minAreaRect(largestContour)
    new_img_path = "data/temp" + img_path[img_path.rindex("/"):len(img_path) - 4] + "_boxed.jpg"
    cv2.imwrite(new_img_path, img)
    angle = minAreaRect[-1]
    if angle < -45:
        angle = 90 + angle
    return -1.0 * angle

def rotate(img_path : str) -> str:
    angle = getSkewAngle(img_path)
    img = cv2.imread(img_path)
    (w, h) = img.shape[:2]
    center = (w // 2, h // 2)
    M = cv2.getRotationMatrix2D(center, angle, 1.0)
    img = cv2.warpAffine(img, M, (w, h), flags=cv2.INTER_CUBIC, borderMode=cv2.BORDER_REPLICATE)
    new_img_path = "data/temp" + img_path[img_path.rindex("/"):len(img_path) - 4] + "_rotated.jpg"
    cv2.imwrite(new_img_path, img)
    return new_img_path

def crop_image(img_path : str) -> str:
    img = cv2.imread(img_path)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    blur = cv2.GaussianBlur(gray, (7, 7), 0)
    thresh = cv2.threshold(blur, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)[1]
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (1, 1))
    dilate = cv2.dilate(thresh, kernel, iterations=0)
    cv2.imwrite("data/temp" + img_path[img_path.rindex("/"):len(img_path) - 4] + "_cropblur.jpg", dilate)
    contours = cv2.findContours(dilate, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    contours = contours[0] if len(contours) == 2 else contours[1]
    sorted(contours, key=cv2.contourArea)
    x, y, w, h = cv2.boundingRect(contours[-1])
    roi = img[y + h // 80:y+h - h // 80, x + w // 22:x+w - w // 22]
    new_img_path = "data/temp" + img_path[img_path.rindex("/"):len(img_path) - 4] + "_crop.jpg"
    cv2.imwrite(new_img_path, roi)
    cv2.rectangle(img, (x + w // 22, y + h // 80), (x+w - w // 22, y+h - h // 80), (0, 255, 0), 2)
    cv2.imwrite("data/temp" + img_path[img_path.rindex("/"):len(img_path) - 4] + "_cropbox.jpg", img)
    return new_img_path

def bw(img_path : str) -> str:
    img = cv2.imread(img_path)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    blur = cv2.GaussianBlur(gray, (7, 7), 0)
    thresh, bw_img = cv2.threshold(blur, 180, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    new_img_path = "data/temp" + img_path[img_path.rindex("/"):len(img_path) - 4] + "_bw.jpg"
    cv2.imwrite(new_img_path, bw_img)
    return new_img_path

def dilate(img_path : str) -> str:
    img = cv2.imread(img_path)
    img = cv2.bitwise_not(img)
    kernel = np.ones((2,2), np.uint8)
    img = cv2.dilate(img, kernel, iterations=1)
    img = cv2.bitwise_not(img)
    new_img_path = "data/temp" + img_path[img_path.rindex("/"):len(img_path) - 4] + "_dilated.jpg"
    cv2.imwrite(new_img_path, img)
    return new_img_path


def bounding_box(img_path : str) -> str:
    img = cv2.imread(img_path)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    blur = cv2.GaussianBlur(gray, (7, 7), 0)
    thresh = cv2.threshold(blur, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)[1]
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (130, 25))
    dilate = cv2.dilate(thresh, kernel, iterations=3)
    cv2.imwrite("data/temp" + img_path[img_path.rindex("/"):len(img_path) - 4] + "_blur.jpg", dilate)
    contours = cv2.findContours(dilate, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    contours = contours[0] if len(contours) == 2 else contours[1]
    contours = sorted(contours, key=lambda x: cv2.boundingRect(x)[0])

    amount = 0;
    for c in contours:
        x, y, w, h = cv2.boundingRect(c)
        if w > 100 and h > 100:
            roi = img[y:y+h, x:x+w]
            cv2.imwrite("data/temp" + img_path[img_path.rindex("/"):len(img_path) - 4] + "_roi_" + str(amount) + ".jpg", roi)
            amount += 1
        cv2.rectangle(img, (x, y), (x+w, y+h), (0, 255, 0), 2)
    new_img_path = "data/temp" + img_path[img_path.rindex("/"):len(img_path) - 4] + "_boxed.jpg"
    cv2.imwrite(new_img_path, img)
    return new_img_path

def gamma_correction(img_path : str) -> str:
    img = cv2.imread(img_path)
    lookUpTable = np.empty((1,256), np.uint8)
    for i in range(256):
        lookUpTable[0,i] = np.clip(pow(i / 255.0, 1) * 255.0, 0, 255)
    res = cv2.LUT(img, lookUpTable)
    new_img_path = "data/temp" + img_path[img_path.rindex("/"):len(img_path) - 4] + "_gamma.jpg"
    cv2.imwrite(new_img_path, res)
    return new_img_path

def bound_each_word(img_path : str) -> dict:
    img = cv2.imread(img_path)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    blur = cv2.GaussianBlur(gray, (3, 3), 0)
    thresh = cv2.threshold(blur, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)[1]
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (15, 2))
    dilate = cv2.dilate(thresh, kernel, iterations=2)
    cv2.imwrite("data/temp" + img_path[img_path.rindex("/"):len(img_path) - 4] + "_blur.jpg", dilate)
    contours = cv2.findContours(dilate, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    contours = contours[0] if len(contours) == 2 else contours[1]
    contours = sorted(contours, key=lambda x: cv2.boundingRect(x)[1])

def resize_image(img_path: str, scale: float = 2.0) -> str:
    img = cv2.imread(img_path)
    width = int(img.shape[1] * scale)
    height = int(img.shape[0] * scale)
    resized_img = cv2.resize(img, (width, height), interpolation=cv2.INTER_LINEAR)
    new_img_path = "temp" + img_path[img_path.rindex("/"):len(img_path) - 4] + "_resized.jpg"
    cv2.imwrite(new_img_path, resized_img)
    return new_img_path
    
    words = {}
    amount = 0;
    for c in contours:
        x, y, w, h = cv2.boundingRect(c)
        roi_img_path = "data/temp" + img_path[img_path.rindex("/"):len(img_path) - 4] + "_boxed_" + str(y + w // 2) + ".jpg"
        if y + w // 2 in words.keys():
            words[y + w // 2].append(roi_img_path)
        else:
            words.update({y + w // 2 : [roi_img_path]})
        roi = img[y - 10: y + h + 10, x : x + w]
        cv2.imwrite(roi_img_path, roi)
    return words

def resize_image(img_path: str, scale: float = 2.0) -> str:
    img = cv2.imread(img_path)
    width = int(img.shape[1] * scale)
    height = int(img.shape[0] * scale)
    resized_img = cv2.resize(img, (width, height), interpolation=cv2.INTER_LINEAR)
    new_img_path = "data/temp" + img_path[img_path.rindex("/"):len(img_path) - 4] + "_resized.jpg"
    cv2.imwrite(new_img_path, resized_img)
    return new_img_path

img_path = "data/IMG_4404.JPG"
img = bounding_box(bw(crop_image(gamma_correction(img_path))))
processed_img = rotate(resize_image(noise_removal(bw(crop_image(gamma_correction(img_path))))))
p_img = Image.open(processed_img)
ocr_result = pytesseract.image_to_string(p_img, lang='eng', config='--psm 6')
print(ocr_result.strip())

'''filenames = []
for filename in os.listdir('data/temp'):
    if 'roi' in filename:
        filenames.append('data/temp/' + filename)

words = bound_each_word(filenames[1])

for x in words:
    print(x)
    for y in words[x]:
        print(':', y)

for x in words:
    print(x)
    for y in words[x]:
        p_img = Image.open(y)
        ocr_result = pytesseract.image_to_string(p_img, lang='eng')
        print(":", ocr_result)'''
