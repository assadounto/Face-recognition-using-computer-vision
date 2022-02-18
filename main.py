import cv2
import face_recognition
import numpy as np

imgBillgate = face_recognition.load_image_file("images/Billgate.jpg")
imgBillgate=cv2.cvtColor(imgBillgate,cv2.COLOR_BGR2RGB)
imgbillgate2 = face_recognition.load_image_file("images/billgate2.jpg")
imgbillgate2=cv2.cvtColor(imgbillgate2,cv2.COLOR_BGR2RGB)

faceloc = face_recognition.face_locations(imgBillgate)[0]
encodeBillgate = face_recognition.face_encodings(imgBillgate)[0]
cv2.rectangle(imgBillgate,(faceloc[3],faceloc[0]),(faceloc[1],faceloc[2]),(255,0,255),2)

faceloc1 = face_recognition.face_locations(imgbillgate2)[0]
encodebillgate2 = face_recognition.face_encodings(imgbillgate2)[0]
cv2.rectangle(imgbillgate2,(faceloc1[3],faceloc1[0]),(faceloc1[1],faceloc1[2]),(255,0,255),2)

results=face_recognition.compare_faces([encodeBillgate],encodebillgate2)
faceDis = face_recognition.face_distance([encodeBillgate],encodebillgate2)
print(results,faceDis)
cv2.putText(imgbillgate2,f'{results} {round(faceDis[0],2)}',(50,50),cv2.FONT_HERSHEY_COMPLEX,1,(0,0,255),2)

cv2.imshow('Billgate',imgBillgate)
cv2.imshow('billgate',imgbillgate2)
cv2.waitKey(0)