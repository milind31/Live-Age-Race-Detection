#find and get position of face from webcame footage using frontalface haar cascade
#feed face to DeepFace model to detect race
#display result of model on screen
from deepface import DeepFace
import cv2
import numpy as np

#open webcam and cascade
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
cap = cv2.VideoCapture(0)

while True:
        ret, img = cap.read()
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        faces = face_cascade.detectMultiScale(gray, 1.3, 5) #find face using cascade
        for (x, y, w, h) in faces:
                detected_face = img[int(y):int(y+h), int(x):int(x+w)]
                margin_rate = 30
                #get some area around face to match position of face on images the model is trained on
                try:
                        margin_x = int(w * margin_rate / 100)
                        margin_y = int(h * margin_rate / 100)
                        detected_face = img[int(y-margin_y):int(y+h+margin_y), int(x-margin_x):int(x+w+margin_x)]  
                        detected_face = cv2.resize(detected_face, (224, 224)) #resize to 224x224
                except Exception as err:
                        detected_face = img[int(y):int(y+h), int(x):int(x+w)] 
                        detected_face = cv2.resize(detected_face, (224, 224))
                obj = DeepFace.analyze(detected_face, actions=["race"], enforce_detection=False) #feed to model
                cv2.rectangle(img, (x, y), (x + w, y + h), (36,255,12), 1) #draw rectangle
                cv2.putText(img, obj["dominant_race"], (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (36,255,12), 2) #add text

        cv2.imshow('img', img)
        k = cv2.waitKey(30) & 0xff
        if k == 27:
                break

cap.release()
cv2.destroyAllWindows()

