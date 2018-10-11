import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
from sklearn.decomposition import PCA
from sklearn.neural_network import MLPClassifier
import pickle
import cv2


no_of_records=1000
X=[]
for i in range(int(no_of_records/2)):
    img=cv2.imread('Mom/'+str(i+1)+".jpg",0)
    img=np.reshape(img, (784))    
    X.append(img)
for i in range(int(no_of_records/2)):
    img=cv2.imread('son/'+str(i+1)+".jpg",0)
    img=np.reshape(img, (784))
    X.append(img)
    

    
X=np.array(X)
print(X.shape)

y=[]
for i in range(int(no_of_records/2)):
    y.append(0)
    
for i in range(int(no_of_records/2)):
    y.append(1)
target_names=['Mom','Son']
y=np.array(y)
#print(y.shape)

#print(y)
def reshuffle(X,y):
    X1=[]
    y1=[]
    for i in range(500):
        X1.append(X[i,:])
        y1.append(0)
        X1.append(X[500+i,:])
        y1.append(1)
    return np.array(X1), np.array(y1)
X,y=reshuffle(X,y)
print(y)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)
print(X_train.shape)

n_components = 100
pca = PCA(n_components=n_components, whiten=True).fit(X_train)
 
# apply PCA transformation
X_train_pca = pca.transform(X_train)
X_test_pca = pca.transform(X_test)

print("Fitting the classifier to the training set")
clf = MLPClassifier(hidden_layer_sizes=(1024,), batch_size=256, verbose=True, early_stopping=True).fit(X_train_pca, y_train)


print(X_test_pca.shape)
y_pred = clf.predict(X_test_pca)
print(classification_report(y_test, y_pred, target_names=target_names))
test_img=cv2.imread("mom/5.jpg",0)
if test_img is not None:
    test_img=np.reshape(test_img,(1,784))
    print(test_img.shape)
    test_img=pca.transform(test_img)
    print(clf.predict(test_img))
else:
    print('none image')

filename="savedNLPmodel.sav"
pcafile="pcatransform.sav"
pickle.dump(clf, open(filename,"wb"))
pickle.dump(pca, open(pcafile, 'wb'))
