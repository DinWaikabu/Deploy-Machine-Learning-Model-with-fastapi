#import library
import pandas as pd 
import numpy as np 
from sklearn.ensemble import RandomForestClassifier
import pickle
from sklearn.model_selection import train_test_split

url = "https://archive.ics.uci.edu/ml/machine-learning-databases/iris/iris.data"
names = ['sepal-length', 'sepal-width', 'petal-length', 'petal-width', 'class']

df = pd.read_csv(filepath_or_buffer=url,header=None,sep=',',names=names)

array = df.values
X = array[:, 0:4]
y = array[:, 4]

#Split data ratio 70/30
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=0)

Clf = RandomForestClassifier()
Clf.fit(X_test, y_test)
#save the model
pickle.dump(Clf, open("rf_model.pkl", "wb"))
#load the model
#loaded_model = pickle.load(open("rf_model.pkl", "rb"))
#result = loaded_model.score(X_test, y_test)
#print(result)




