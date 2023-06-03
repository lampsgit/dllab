import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
data = {
    'height': [178, 181, 162, 143, 184, 153, 154, 152, 185, 168, 150],
    'weight': [98, 74, 62, 50, 104, 53, 68, 48, 99, 64, 50]
}
df=pd.DataFrame(data)
X=df.iloc[:,:-1].values
Y=df.iloc[:,-1].values
X_train,X_test,Y_train,Y_test=train_test_split(X,Y,test_size=0.1)
regressor=LinearRegression()
regressor.fit(X,Y)
print(regressor.predict([[182]]))
