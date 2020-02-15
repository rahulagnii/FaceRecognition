import numpy as np
import cv2
a = 3
b = 9
faceCascade = cv2.CascadeClassifier('Cascades/haarcascade_frontalface_default.xml')
img = cv2.imread("tmp/"+str(a)+"."+str(b)+".jpg")
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
faces = faceCascade.detectMultiScale(
    gray,
    scaleFactor=1.2,
    minNeighbors=5,
    minSize=(20, 20)
)
for (x,y,w,h) in faces:
    cv2.rectangle(img,(x,y),(x+w,y+h),(255,0,0),2)
    roi_gray = gray[y:y+h, x:x+w]
    roi_color = img[y:y+h, x:x+w]
    img_item = "dataset/User."+str(a)+"."+str(b)+".jpg"
    cv2.imwrite(img_item,roi_gray)
cv2.imshow("gray",img)
k = cv2.waitKey(0)
img.release()
cv2.destroyAllWindows()