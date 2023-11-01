import cv2

def blur_face(img):
    (h, w) = img.shape[:2]
    dh = int(h / 5.0)
    dw = int(w / 5.0)
    if dh % 2 == 0:
        dh -= 1
    if dw % 2 == 0:
        dw -= 1
    return cv2.GaussianBlur(img, (dw, dh), 0)

xml = cv2.CascadeClassifier('faces.xml')

# ---------------------------- blur face on image ----------------------------

image = cv2.imread('faces.jpg')
image_gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

results = xml.detectMultiScale(image_gray, scaleFactor=1.15, minNeighbors=3)

for (x, y, w, h) in results:
    image[y: y + h, x: x + w] = blur_face(image[y: y + h, x: x + w])
    cv2.rectangle(image, (x, y), (x + w, y + h), (50, 50, 50), thickness=1)

cv2.imshow("Blured faces", image)
cv2.waitKey(0)

# ---------------------------- blur face on video ----------------------------

# vid = cv2.VideoCapture(0)
# vid.set(3, 500)
# vid.set(4, 500)
# while True:
#     success, img = vid.read()
#     gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
#     results = xml.detectMultiScale(gray, scaleFactor=1.2, minNeighbors=3)
#     for (x, y, w, h) in results:
#         img[y: y + h, x: x + w] = blur_face(img[y: y + h, x: x + w])
#         cv2.rectangle(img, (x, y), (x + w, y + h), (50, 50, 50), thickness=1)
#     cv2.imshow('Video with blured face', img)
#     if cv2.waitKey(2) & 0xFF == ord('q'):
#         break
