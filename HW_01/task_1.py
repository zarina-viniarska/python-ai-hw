# ------------------------ task 1 (images) ------------------------

import cv2

# -------------------- smile detection --------------------

img = cv2.imread('img/smile.jpg')
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
smile = cv2.CascadeClassifier('xml/smile.xml')

results = smile.detectMultiScale(gray, scaleFactor = 1.5, minNeighbors = 4)
for (x, y, w, h) in results :
    cv2.rectangle(img, (x, y), (x + w, y + h), (0, 0, 255), thickness = 2)

cv2.imshow("Smiles detected", img)
cv2.waitKey(0)

# ------------------- eyes detection -------------------

# img = cv2.imread('img/eye.jpg')
# gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
# eyes = cv2.CascadeClassifier('xml/eye.xml')

# results = eyes.detectMultiScale(gray, scaleFactor = 1.3, minNeighbors = 4)

# for (x, y, w, h) in results :
#     cv2.rectangle(img, (x, y), (x + w, y + h), (0, 0, 255), thickness = 2)

# cv2.imshow("Eyes detected", img)
# cv2.waitKey(0)

# ------------------ upperbody detection ------------------

# img = cv2.imread('img/upperbody.jpg')
# gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
# upperbody = cv2.CascadeClassifier('xml/upperbody.xml')

# results = upperbody.detectMultiScale(gray, scaleFactor = 1.03, minNeighbors = 5)

# for (x, y, w, h) in results :
#     cv2.rectangle(img, (x, y), (x + w, y + h), (0, 0, 255), thickness = 2)

# cv2.imshow("Upperbodies detected", img)
# cv2.waitKey(0)