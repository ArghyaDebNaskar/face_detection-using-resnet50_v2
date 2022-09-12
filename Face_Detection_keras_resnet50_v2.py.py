#from ctypes.wintypes import RGB
import numpy as np
import cv2
# from PIL import Image
from keras.applications.vgg16 import preprocess_input
import base64
from io import BytesIO
import json
import random
from keras.models import load_model
model=load_model(r"E:\Jupyter\facefeatures_resnet50_model.h2")
cap=cv2.VideoCapture(0)
face_cascade=cv2.CascadeClassifier(r"E:\Jupyter\haarcascade_frontalface_default.xml")

while True:
 ret,frame=cap.read()
 #frame=frame[:1200,:]
 faces=face_cascade.detectMultiScale(frame)

 for (x,y,w,h) in faces:
  cx=int((x+x+w)/2)
  cy=int((y+y+h)/2)
  #cv2.circle(frame,(cx,cy),3,(0,255,0),-1)
  cv2.rectangle(frame,(x,y),(x+w,y+h),(125,125,0),2)
  face=faces[x:x+w,y:y+h]
  if type(face) is  np.ndarray:
    face=cv2.resize(face,(224,224))
    im=Image.fromarray(face,RGB)
    img_array=np.array(im)
    img_array=np.expand_dims(img_array,axis=0)
    pred=model.predict(img_array)
    name="None matching"
    if (pred[0][5])>0.1:
        cv2.putText(frame,"Arghya Deb Naskar",cv2.FONT_HERSHEY_COMPLEX,1,(200,100,50),2)
  else:
    cv2.putText(frame,"No face found",cv2.FONT_HERSHEY_COMPLEX,1,(200,100,50),2)       




  cv2.imshow("Frame",frame)
  key=cv2.waitKey(1)
  if key==30:
    break
cap.release()
cv2.destroyAllWindows()