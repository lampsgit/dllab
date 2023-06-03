import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
data = {
    'Age': [18,22,27,28,46,47,49,52,55,56,60,61,62],
    'weight': [0,0,0,0,1,1,1,0,0,1,0,1,1]
}
df=pd.DataFrame(data)
X=df.iloc[:,:-1].values
Y=df.iloc[:,-1].values
scaler=StandardScaler()
scaler.fit_transform(X)
X_train,X_test,Y_train,Y_test=train_test_split(X,Y,test_size=0.2)
regressor=LogisticRegression()
regressor.fit(X_train,Y_train)
print(regressor.predict(X_test))
