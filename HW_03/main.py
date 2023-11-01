import cv2
import numpy as np
import imutils
import easyocr

def create_file():
    with open('numbers.txt', 'w') as file:
        file.write("Found numbers on images:\n")


def add_number_to_file(number):
    try:
        with open('numbers.txt', 'r') as file:
            if number not in file.read():
                with open('numbers.txt', 'a') as file:
                    file.write(number + '\n')
                print(f"Number '{number}' is added to file.")
            else:
                print(f"Number '{number}' is already in file.")
    except Exception as e:
        print(f"Error: {e}")


def read_number(plate):
    text = easyocr.Reader(['en'])
    text = text.readtext(plate)
    res = text[0][-2]
    add_number_to_file(res)


def find_number_plate(img):
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    filter_ = cv2.bilateralFilter(gray, 11, 15, 15)
    edges = cv2.Canny(filter_, 30, 200)
    cont = cv2.findContours(edges.copy(), cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    cont = imutils.grab_contours(cont)
    cont = sorted(cont, key=cv2.contourArea, reverse=True)[:8]

    pos = None
    for c in cont:
        app_ = cv2.approxPolyDP(c, 10, True)
        if len(app_) == 4:
            pos = app_
            break

    mask = np.zeros(gray.shape, np.uint8)
    new_img = cv2.drawContours(mask, [pos], 0, 255, -1)
    bit_img = cv2.bitwise_and(img, img, mask = mask)

    x,y = np.where(mask == 255)
    x1, y1 = np.min(x), np.min(y)
    x2, y2 = np.max(x), np.max(y)

    cropp = gray[x1:x2, y1:y2]
    cv2.imshow("Number plate", cropp)
    read_number(cropp)


create_file()
for i in range(1, 6):
    img = cv2.imread(f'cars/c{i}.jpg')

    find_number_plate(img)
    cv2.waitKey(5000)