
import numpy as np
import cv2
import pickle
from sklearn.decomposition import PCA
def runmodel(cropped):
    pca=pickle.load(open("pcatransform.sav","rb"))
    print(cropped.shape)
    loaded_model = pickle.load(open("savedNLPmodel.sav", 'rb'))
    #print(type(loaded_model))
    cropped=np.reshape(cropped,(1,784))
    cropped_pca=pca.transform(cropped)
    print(cropped_pca.shape)
    predict=loaded_model.predict(cropped_pca)
    #print(predict)
    if predict[0]==1:
        return "SON"
    else:
        return "MOM"
cap=cv2.VideoCapture(0)
#imgcount=1000
font = cv2.FONT_HERSHEY_PLAIN
face_cascade = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')
while(True):
    ret,frame=cap.read()
    
    gray=cv2.cvtColor(frame,cv2.COLOR_BGR2GRAY)
    faces=face_cascade.detectMultiScale(gray,1.3,5)
    for (x,y,w,h) in faces:
        cv2.rectangle(frame, (x,y),(x+w,y+h),(255,0,0),2)
        roi_gray=gray[y:y+h,x:x+w]        
    
    cropped=cv2.resize(roi_gray,(28,28))
    text=runmodel(cropped)
    #print(frame.shape)
    cv2.putText(frame,text,(x,y), font, 1.3,(255,255,255),2,cv2.LINE_AA)
    cv2.imshow('frame',frame)
    #cv2.imwrite("Mom/"+str(imgcount)+".jpg",cropped)
    #print(cropped.shape)
    
    if cv2.waitKey(1) & 0XFF==ord('q'):
        break;
    #cv2.imwrite("1"+".jpg",roi_gray)    
cap.release()
cv2.destroyAllWindows()