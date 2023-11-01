# ------------------------ task 2 (video) ------------------------
import cv2

face = cv2.CascadeClassifier('xml/face.xml')
vid = cv2.VideoCapture('videos/video.mp4')

while True:
    success, img = vid.read()

    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    results = face.detectMultiScale(gray, scaleFactor = 1.3, minNeighbors = 6)
    for (x, y, w, h) in results :
        cv2.rectangle(img, (x, y), (x + w, y + h), (0, 0, 255), thickness = 2)
    cv2.imshow('Video with detected faces', img)
    if cv2.waitKey(2) & 0xFF == ord('q'):
        break