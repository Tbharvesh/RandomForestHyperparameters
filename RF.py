import array
import pandas as pd
from matplotlib import pyplot as plt
import seaborn as sns
from sklearn.metrics import accuracy_score
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.metrics import mean_squared_error
import streamlit as st


st.title("Random Forest Classifier")
st.header("Dataset: titanic.csv")
df=pd.read_csv("titanic.csv")
st.write(df.head())
sel_col,disp_col = st.columns(2)


#Cleaning and standardising data
df['Sex'].replace(['male','female'],[0,1],inplace=True)
df["Age"].fillna(df.Age.mean(), inplace = True)
df=df.drop(['Name', 'SibSp','Parch','Ticket','Embarked'], axis=1)

from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score


X = df.drop(['Survived', 'Cabin'], axis=1)
y = df['Survived']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)


max_depth = sel_col.slider('What should be the max_depth of the model?', min_value=10, max_value=200, step=10)
n_estimators = sel_col.selectbox('How many trees should be there?', options=[100, 200, 300, 'No limit'], index=0)

# Create and train the RandomForestClassifier
classifier = RandomForestClassifier(n_estimators=n_estimators, criterion="entropy", max_depth=max_depth)
classifier.fit(X_train, y_train)

# Make predictions on the testing set
y_pred = classifier.predict(X_test)

# Calculate accuracy
accuracy = accuracy_score(y_test, y_pred)

disp_col.subheader('Accuracy of the model is:')
disp_col.write(accuracy)

disp_col.subheader('Mean squared error of model is:')
disp_col.write(mean_squared_error(y_test,y_pred))

