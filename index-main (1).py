import streamlit as st
import numpy as np
import plotly.express as px
import pandas as pd
import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score
from sklearn.model_selection import train_test_split
LR=LinearRegression()
data=pd.read_csv('london_houses_cleaned.csv',encoding='utf-8')
st.write(dat.columns.tolist())
x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.2,random_state=0)
LR.fit(x_train,y_train)
y_pred=LR.predict(x_test)
r2=r2_score(y_test,y_pred)
st.title("Welcome to my linear regression model")
st.subheader("This model can predict the value of houses in london")
st.write("Here is the accuracy of the model:")
labels = ['Correct','Incorrect']
sizes = [r2*100,100-(r2*100)]
fig = px.pie(values=sizes, names=labels, title='Accuracy')
st.plotly_chart(fig)
st.subheader("So,How about you try it out and predict the value of your dream hosue!")
bedrooms=st.selectbox("Select the number of bedrooms you want",[1,2,3,4,5])
bathrooms=st.selectbox("Select the number of bathrooms you want",[1,2,3])
Square_Meters= st.number_input("Enter the prefered square meters of your dream house", min_value=0, value=1, step=1)
Building_Age= st.number_input("Enter the prefered building age of your dream house", min_value=0, value=1, step=1)
Garden=st.selectbox("Do you want a garden?",['Yes','No'])
Garage=st.selectbox("Do you want a garage?",['Yes','No'])
floors=st.selectbox("Select the number of floors you want",[1,2,3])
Property_Type=st.selectbox("Select the property type you want",['Semi-Detached','Apartment','Detached House'])
Heating_Type=st.selectbox("Select the heating type you want",['Electric Heating','Gas Heating','Underfloor Heating','Central Heating'])
Balcony=st.selectbox("Select the balcony type you want",['No Balcony','High-level Balcony','Low-level Balcony'])
Interior_Style=st.selectbox("Select the interior style you want",['Classic','Industrial','Modern','Minimalist'])
View=st.selectbox("Select the view you want",['Sea','Garden','Park','City','Street'])
Materials=st.selectbox("Select the materils to be used in the building",['Wood','Laminate Flooring','Marble','Granite'])
Building_status=st.selectbox("What building status do you want",['Renovated','New','Old'])
dmeters_mean=149.627
dmeters_st= 58.0562
dage_mean=49.965
dage_st=29.07086
dprice_mean=1840807.278
dprice_st=879348.407509
square_meters_scaled = (Square_Meters - dmeters_mean) / dmeters_st
building_age_scaled = (Building_Age - dage_mean) / dage_st
if Garage == 'yes':
    garage_final=1
else:
    garage_final=0
if Garden == 'yes':
    garden_final=1
else:
    garden_final=0
if Property_Type=='Semi-Detached':
    Property_Type_final=1
elif Property_Type=='Detcahed House':
    Property_Type_final=2
elif Property_Type=='Apartment':
    Property_Type_final=0
if Heating_Type=='Electric Heating':
    Heating_Type_final=1
elif Heating_Type=='Gas Heating':
    Heating_Type_final=2
elif Heating_Type=='Underfloor Heating':
    Heating_Type_final=3
elif Heating_Type=='Central Heating':
    Heating_Type_final=0
if Balcony=='High-level Balcony':
    Balcony_final=0
elif Balcony=='Low-level Balcony':
    Balcony_final=1
elif Balcony=='No Balcony':
    Balcony_final=2
if Interior_Style=='Classic':
    Interior_Style_final=0
elif Interior_Style=='Industrial':
    Interior_Style_final=1
elif Interior_Style=='Modern':
    Interior_Style_final=3
elif Interior_Style=='Minimalist':
    Interior_Style_final=2
if View=='City':
    View_final=0
elif View=='Street':
    View_final=4
elif View=='Garden':
    View_final=1
elif View=='Park':
    View_final=2
elif View=='Sea':
    View_final=3
if Materials=='Granite':
    Materials_final=0
elif Materials=='Wood':
    Materials_final=3
elif Materials=='Laminate':
    Materials_final=1
elif Materials=='Marble':
    Materials_final=2
if Building_status=='New':
    Building_status_final=0
elif Building_status=='Old':
    Building_Status_final=1
elif Building_status=='Renovated':
    Building_Status_final=2
feature_names = ['Unnamed: 0','Bedrooms','Bathrooms','Square Meters','Building Age','Garden','Garage'
,'Floors','Property Type','Heating Type','Balcony','Interior Style','View'
,'Materials','Building Status']
data_values = [0,bedrooms,bathrooms,square_meters_scaled,building_age_scaled,garden_final,garage_final,floors,Property_Type_final,
               Heating_Type_final,Balcony_final,Interior_Style_final,View_final,Materials_final,Building_Status_final]
data_df = pd.DataFrame([data_values], columns=feature_names)
if st.button('Go'):
    prediction = LR.predict(data_df).item()
    prediction_final1=(prediction*dprice_st)+dprice_mean
    prediction_final=np.ceil(prediction_final1)
    st.subheader('The final predicted value for your house is:')
    float(prediction_final)
    st.subheader(f"£{prediction_final}")
