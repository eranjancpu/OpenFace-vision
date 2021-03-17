import os
import numpy
import cv2 as cv

img = cv.imread("face.jpg")
gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)

haar_cas = cv.CascadeClassifier('full_body.xml')

body_rect = haar_cas.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=3)

if len(body_rect) == 1:
    print("THERE IS ONLY 1 FACE")
else:
    print(f"NUMBER OF FACE FOUND IS {len(body_rect)}")
for (x,y,w,h) in body_rect:
    cv.rectangle(img, (x,y), (x + w, y + h), (0,255,0), thickness=2)


cv.imshow("face", img)
cv.waitKey(0)