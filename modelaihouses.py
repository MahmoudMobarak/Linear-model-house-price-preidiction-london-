import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score
from sklearn.model_selection import train_test_split
LR=LinearRegression()
data=data = pd.read_csv(r"C:\Users\HP\Desktop\VS\streamlit\london_houses_cleaned.csv", encoding="utf-8")
y=data['Price (£)']
x = data.drop(['Address', 'Neighborhood', 'Price (£)'], axis=1)
x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.2,random_state=0)
LR.fit(x_train,y_train)
y_pred=LR.predict(x_test)
r2=r2_score(y_test,y_pred)
def get_model():
    return LR