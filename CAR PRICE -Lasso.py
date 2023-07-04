import pandas as pd
from sklearn.linear_model import Lasso
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn import metrics
car=pd.read_csv(r"C:\Users\dines\OneDrive\Documents\CSV Files\car data.csv")
map1={"Petrol":0,
      "Diesel":1,
      "CNG":2}
map2={"Dealer":0,
      "Individual":1}
map3={"Manual":0,
      "Automatic":1}

car["Fuel_Type"]=car["Fuel_Type"].replace(map1)
car["Seller_Type"]=car["Seller_Type"].replace(map2)
car["Transmission"]=car["Transmission"].replace(map3)

#Splitting the test and train data
X=car.drop(["Car_Name","Selling_Price"],axis=1).values
y=car["Selling_Price"].values

X_train,X_test,y_train,y_test=train_test_split(X,y,test_size=0.1,random_state=2023)
lasso= Lasso()
lasso.fit(X_train,y_train)

train_data_pred = lasso.predict(X_train)
train_error_score= metrics.r2_score(y_train,train_data_pred)
print("Lasso Train data - R Squared value is:",train_error_score)

test_data_pred = lasso.predict(X_test)
test_error_score= metrics.r2_score(y_test,test_data_pred)
print("Laaso Test data - R Squared value is:",test_error_score)

#Visualizing actual vs. predictes prices
plt.scatter(y_test,test_data_pred)
plt.xlabel("Actual price")
plt.ylabel("Predicted prices")
plt.title("Price comparison - Lasso Test data")
plt.show()