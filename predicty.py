import numpy as np 
import pandas as pd 
import sklearn


data=pd.read_csv("/mnt/F8F8B8AFF8B86E0E/indian-liver-patient-records/indian_liver_patient.csv")

data=data.fillna(method="ffill")

X=data.iloc[:,:-1]
y=data.iloc[:,-1]


from sklearn.preprocessing import LabelEncoder
labelen=LabelEncoder()
X['Gender']=labelen.fit_transform(X['Gender'])

from sklearn.preprocessing import MinMaxScaler
mn=MinMaxScaler()
X=mn.fit_transform(X) 

'''from sklearn.preprocessing import StandardScaler
sc=StandardScaler()
X=sc.fit_transform(X)'''

'''from sklearn.decomposition import PCA
pca=PCA(n_components=6)      #SHOWS THE BEST ACCURACY 79%
X=pca.fit_transform(X) '''





from sklearn.model_selection import train_test_split

X_train,X_test,y_train,y_test=train_test_split(X,y,test_size=0.20,random_state=42)

import matplotlib.pyplot as plt
from sklearn.linear_model import LogisticRegression
def model_tester(n):
    logreg=LogisticRegression(C=n , solver='liblinear')
    logreg.fit(X_train,y_train)
    y_pred=logreg.predict(X_test)
    from sklearn.metrics import accuracy_score
    accuracy=accuracy_score(y_test,y_pred)
    return accuracy
for i in range(1,101):
    k=model_tester(i)
    print("when c:",i,"accuracy=",k)





