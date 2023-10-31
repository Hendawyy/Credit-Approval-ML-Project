import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns 

CRX = pd.read_csv('CRX.csv',header = None)
CRX.shape
CRX.head()
CRX.info()
CRX.describe()

char_cols = []
for x, xRow in CRX.iteritems():
    if(CRX[x].dtype == "object"):
        char_cols.append(x)

print("Lenght: ", len(char_cols))
char_cols

spCharascters = {'\?', '\+' }
has_characters = {}

for i in spCharascters:
    for xcols in char_cols:
        has_characters[str(xcols) + " " + i] = len(CRX[CRX[xcols].str.contains(i)] )

print(has_characters)


CRX[CRX[1].str.contains('\?')]
CRX = CRX.replace('?', np.nan)
CRX.tail(10)

CRX.isna().sum()
CRX.info()
CRX[1] = CRX[1].astype('float')
CRX[13] = CRX[13].astype('float')
CRX.info()

CRXx = CRX.copy()

# Select the Categorical Features
categorical_features = []
for x, xCols in CRXx.iteritems():
    if(CRXx[x].dtype == 'object'):
        categorical_features.append(x)

print("Categorical Features")
print("-"*30)
print("Total Numbers of Columns: ", len(categorical_features) )
print("\n")
print("List of Columns")
print("-"*20)
categorical_features

# Import Library for Label Encoding
from sklearn.preprocessing import LabelEncoder
le = LabelEncoder()

# Apply the Encoding on Categorical Features
for col in categorical_features:
    CRXx[col] = le.fit_transform(CRXx[col]).astype("int8")
    
CRXx.head(5)



X = CRXx.drop(15, axis = 1)
y = CRXx[13]
print(X.shape, y.shape)


from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split


X, y = make_classification(n_samples=100, random_state=1)
#X_train, X_test, y_train, y_test = train_test_split(X, y, stratify=y, random_state=1)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=1)

#Import Gaussian Naive Bayes model
from sklearn.naive_bayes import GaussianNB

#Create a Gaussian Classifier
gnb = GaussianNB()

#Train the model using the training sets
gnb.fit(X_train, y_train)

#Predict the response for test dataset
y_pred = gnb.predict(X_test)


#Import scikit-learn metrics module for accuracy calculation
from sklearn.metrics import mean_absolute_error, mean_squared_error, accuracy_score


print("Accuracy:",accuracy_score(y_test, y_pred))

from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import BaggingClassifier
from sklearn.ensemble import ExtraTreesClassifier
from sklearn import tree
from sklearn.tree import DecisionTreeClassifier

names = [
    "K-Nearest Neighbours",
    "Random Forest Classifier",
    "Bagging Classifier",
    "Extra Trees Classifier"
   ,"Decision Tree Classifier"
]

classifiers = [
    KNeighborsClassifier(n_neighbors=3),
    RandomForestClassifier(max_depth=5, n_estimators=100),
    BaggingClassifier(n_estimators=10, random_state=0),
    ExtraTreesClassifier(random_state=1)
    ,DecisionTreeClassifier()
]


modelScore = []
modelAccuracy = []

for name, clf in zip(names, classifiers):
    clf.fit(X_train, y_train)
    score = clf.score(X_train, y_train)
    predict = clf.predict(X_test)
    
    modelScore.append(score)
    modelAccuracy.append(accuracy_score(y_test, predict))

print("\n"*2)
print("Score of Train Models")
print("-"*25)
modelScore

print("Model Accuracy")
print("-"*15)
modelAccuracy

df_modeAccuracy = pd.DataFrame(columns=['name', 'score', 'accuracy'])
df_modeAccuracy['name'] = names
df_modeAccuracy['score'] = modelScore
df_modeAccuracy['accuracy'] = modelAccuracy
df_modeAccuracy
