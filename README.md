# Implementation-of-Logistic-Regression-Model-to-Predict-the-Placement-Status-of-Student

## AIM:
To write a program to implement the the Logistic Regression Model to Predict the Placement Status of Student.

## Equipments Required:
1. Hardware – PCs
2. Anaconda – Python 3.7 Installation / Jupyter notebook

## Algorithm
1.Import the required packages and print the present data.

2.Print the placement data and salary data.

3.Find the null and duplicate values.

4.Using logistic regression find the predicted values of accuracy , confusion matrices.

## Program:
```
/*
Program to implement the the Logistic Regression Model to Predict the Placement Status of Student.
Developed by: THIRUMALAI K
RegisterNumber:  212224240176

import pandas as pd
data=pd.read_csv("C:/Users/admin/Downloads/Midhun/Placement_Data.csv")
data.head()

data1=data.copy()
data1=data1.drop(["sl_no","salary"],axis=1)#Browses the specified row or column
data1.head()

data1.isnull().sum()

data1.duplicated().sum()

from sklearn.preprocessing import LabelEncoder
le=LabelEncoder()
data1["gender"]=le.fit_transform(data1["gender"])
data1["ssc_b"]=le.fit_transform(data1["ssc_b"])
data1["hsc_b"]=le.fit_transform(data1["hsc_b"])
data1["hsc_s"]=le.fit_transform(data1["hsc_s"])
data1["degree_t"]=le.fit_transform(data1["degree_t"])
data1["workex"]=le.fit_transform(data1["workex"])
data1["specialisation"]=le.fit_transform(data1["specialisation"] )     
data1["status"]=le.fit_transform(data1["status"])
data1 

x=data1.iloc[:,:-1]
x

y=data1["status"]
y

from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.2,random_state=0)

from sklearn.linear_model import LogisticRegression
lr=LogisticRegression(solver="liblinear")
lr.fit(x_train,y_train)
y_pred=lr.predict(x_test)
y_pred

from sklearn.metrics import accuracy_score
accuracy=accuracy_score(y_test,y_pred)
accuracy

from sklearn.metrics import confusion_matrix
confusion=confusion_matrix(y_test,y_pred)
confusion

from sklearn.metrics import classification_report
classification_report1 = classification_report(y_test,y_pred)
print(classification_report1)

lr.predict([[1,80,1,90,1,1,90,1,0,85,1,85]])

*/
```

## Output:
### TOP 5 ELEMENTS:
![image](https://github.com/user-attachments/assets/42acb71a-095d-4f9a-9253-84e49d32ba5b)
![image](https://github.com/user-attachments/assets/2962ad7b-2266-4f81-a377-95d0ed81c568)

### DATA DUPLICATE:
![image](https://github.com/user-attachments/assets/c5ca2207-f502-4451-832c-e785d198b652)

### PRINT DATA:
![image](https://github.com/user-attachments/assets/127167e0-2a1f-45ef-9c03-9dbcaa1e11cb)

### DATA_STATUS:
![image](https://github.com/user-attachments/assets/ad909465-c3f3-4f1d-a112-fd630d83a3f4)

### Y_PREDICTION ARRAY:
![image](https://github.com/user-attachments/assets/a09a814c-3825-43a8-9c59-1142bd9245bb)

### CONFUSION ARRAY:
![image](https://github.com/user-attachments/assets/01f1f462-c6b4-4de2-a959-3afb7d915aaf)

### ACCURACY VALUE:
![image](https://github.com/user-attachments/assets/81059b7a-d7dc-4c4b-bb25-60af9f85f076)

### CLASSFICATION REPORT:

![image](https://github.com/user-attachments/assets/a3d3cd85-6ef3-4f55-867e-40bd45d0becf)

### PREDICTION:
![image](https://github.com/user-attachments/assets/1def4e1f-ff47-4e1c-b0ce-798da32832d8)











## Result:
Thus the program to implement the the Logistic Regression Model to Predict the Placement Status of Student is written and verified using python programming.
