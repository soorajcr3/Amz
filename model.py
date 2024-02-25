import pandas as pd
import random as rd
import numpy as np
import pickle

data = pd.read_csv("Amazon_Sale_Report_new.csv")
x = data[["month","Category"]]
y = data["Qty"] 

from sklearn.preprocessing import LabelEncoder
le = LabelEncoder()
y = le.fit_transform(y)

from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.2,random_state=42)

from sklearn.ensemble import RandomForestRegressor
model_rf= RandomForestRegressor(n_estimators=100,max_depth=25).fit(x_train,y_train)
pickle.dump(model_rf,open('Aamzon_model.pkl','wb'))
