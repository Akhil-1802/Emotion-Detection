import os
import numpy as np
from skimage.io import imread
from skimage.transform import resize
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score
from sklearn.metrics import classification_report
import pickle
#Preparing the data

input_dr='train'
categories=['angry','fear','happy','sad']


data=[]
labels=[]
for category_idx,category in enumerate(categories):
    for file in os.listdir(os.path.join(input_dr, category)):
        img_path=os.path.join(input_dr, category, file)
        img=imread(img_path)
        img=resize(img,(30,30,3))
        data.append(img.flatten())
        labels.append(category_idx)

data=np.asarray(data)
labels=np.asarray(labels)

#Train the data
x_train,x_test,y_train,y_test=train_test_split(data,labels,test_size=0.2,shuffle=True,stratify=labels)

#Train classifier

classifier=SVC(gamma=0.1,C=100)
classifier.fit(x_train,y_train)

#test performance
y_pred=classifier.predict(x_test)
print("Accuracy Score:",accuracy_score(y_test,y_pred))
print("Classification report :",classification_report(y_test,y_pred))

def emotion_rec(spot_bgr):

    flat_data = []
    img_resized = resize(spot_bgr,(30,30,3))
    flat_data.append(img_resized.flatten())
    flat_data = np.array(flat_data)

    y_output = classifier.predict(flat_data)

    if y_output == 0:
        print("angry")
    elif y_output == 1:
        print("fear")
    elif y_output == 2:
        print("happy")
    else:
        print("sad")

pickle.dump(classifier,open('model1.pkl','wb'))