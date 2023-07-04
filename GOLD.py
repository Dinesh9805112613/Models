#Importing the libararies
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
import numpy as np
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
from sklearn import metrics
import seaborn as sns

#Data Collection and Processing
gold=pd.read_csv(r"C:\Users\dines\OneDrive\Documents\CSV Files\gld_price_data.csv")

#Printing first five rows
print(gold.head(5))
print(gold.shape)  
""" SPX - Stock Value
    GLD - Gold price
    USO - United States Oil Price
    SLV - Silver Price
    EUR/USD - Currency pair (1 Euro= 1 USD)"""

#Printing last 5 rows
print(gold.tail(5))

print(gold.info())

print(gold.describe())

#Correlation between features
correlation=gold.drop(["Date"],axis=1).corr()

#Constructing heat map to understand Correlation
plt.figure(figsize=(8,8))
#sns.heatmap(correlation,cbar=True,square=True,fmt=".1f",annot=True,annot_kws={"size":8},cmap="Blues")
#plt.show()
"""https://seaborn.pydata.org/generated/seaborn.heatmap.html
   cbar - Giving colors to the bar
   sqaure - Whether the bars should be square
   fmt parameter allows to add string (text) values on the cell
   annot- If True, write the data value in each cell
   cmap -  a matplotlib colormap name or object. This maps the data values to the color space."""

print(correlation["GLD"])

#Checking the distribution
#sns.distplot(gold["GLD"],color="Green")
#plt.show()

"""With the plot, we can infer that most of the points lie at and around 120"""

#Splitting the target and the features
X=gold.drop(["Date","GLD"],axis=1).values
y=gold["GLD"].values

#Getting Train and test data
X_train,X_test,y_train,y_test= train_test_split(X,y,test_size=0.2,random_state=2022)
rfr=RandomForestRegressor(n_estimators=100,random_state=2020)
#min_samples_leaf is The minimum number of samples required to be at a leaf node

rfr.fit(X_train,y_train)

#Model evaluation - Prediction on test data
y_pred= rfr.predict(X_test)
error_score=metrics.r2_score(y_test,y_pred)
print(error_score)

#Compare actual and predicted values
plt.plot(y_pred,color="red",label="Predicted")
plt.plot(y_test,color="Blue",label="Actual")
plt.title("Actual vs predicted")
plt.xlabel("No of values")
plt.ylabel("Gold price")
plt.legend()
plt.show()