# Implementation-of-SVM-For-Spam-Mail-Detection

## AIM:
To write a program to implement the SVM For Spam Mail Detection.

## Equipments Required:
1. Hardware – PCs
2. Anaconda – Python 3.7 Installation / Jupyter notebook

## Algorithm
```
1.Import the required packages.
2.Import the dataset to operate on.
3.Split the dataset.
4.Predict the required output.
```

## Program:
```
import chardet
file='/content/spam.csv'
with open(file,'rb') as rawdata:
  result = chardet.detect(rawdata.read(100000))
result


import pandas as pd
data=pd.read_csv('/content/spam.csv',encoding='Windows-1252')

data.head()

data.info()

data.isnull().sum()

x=data["v1"].values
y=data["v2"].values

from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.2,random_state=0)

from sklearn.feature_extraction.text import CountVectorizer
cv=CountVectorizer()

x_train=cv.fit_transform(x_train)
x_test=cv.transform(x_test)

from sklearn.svm import SVC
svc=SVC()
svc.fit(x_train,y_train)
y_pred=svc.predict(x_test)
y_pred

from sklearn import metrics
accuracy=metrics.accuracy_score(y_test,y_pred)
accuracy

```

## Output:
![282243022-28a5795a-2580-433a-9443-e2f07c687b5e](https://github.com/swetha23013979/Implementation-of-SVM-For-Spam-Mail-Detection/assets/153823422/243c13f8-6cb5-4438-890d-39232b596bb8)

![282243035-5cdf8c27-cb0a-43b9-86c0-a2db5671c78d](https://github.com/swetha23013979/Implementation-of-SVM-For-Spam-Mail-Detection/assets/153823422/76361f36-c9f7-4495-92a8-1ef2a7f4d0a7)

![282243043-6c9e9c39-9def-41c3-993e-73ef4e37ac30](https://github.com/swetha23013979/Implementation-of-SVM-For-Spam-Mail-Detection/assets/153823422/826a54e2-3b3b-4c1a-9898-6d3edf101e7b)

![282243050-8cc08474-d436-4d1f-b9fd-300e03f40aca](https://github.com/swetha23013979/Implementation-of-SVM-For-Spam-Mail-Detection/assets/153823422/cb04deab-ed2f-40bc-af1a-8883a8208cec)

![282243051-4ae9ec2d-9001-432f-8bb9-9ae3de9e2311](https://github.com/swetha23013979/Implementation-of-SVM-For-Spam-Mail-Detection/assets/153823422/e90f3fe2-0a8c-4740-b96f-32a217d134b1)

![282243052-7d5ffca2-ba4e-4690-b5d1-524d12659f1c](https://github.com/swetha23013979/Implementation-of-SVM-For-Spam-Mail-Detection/assets/153823422/8633a2b9-fdef-4342-acdc-da17130d10ff)


## Result:
Thus the program to implement the SVM For Spam Mail Detection is written and verified using python programming.
