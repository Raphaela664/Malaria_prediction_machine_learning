import pandas as pd
import joblib as joblib
from sklearn.tree import DecisionTreeClassifier
from sklearn.linear_model import LogisticRegression
from sklearn import svm
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
df = pd.read_excel('Dset.xlsx')
input("Press enter to view your data information")
# Learn about your dataset
print(df.info())
input("Press enter to view your dataset")
print(df.to_string())
input("Press enter to replace the empty cells with 11 ")
df["GRADE"].fillna(11,inplace=True)
# input("Press enter to view duplicates")
# print(df.duplicated().to_string())
# input("Press enter to remove duplicates")
# df.drop_duplicates(inplace=True)
input("Press enter to view our cleaned dataset")
print(df.to_string())

X=df.drop (columns=['COMBINATION','OBSERVATION'])
y=df['OBSERVATION']
X_train, x_test, y_train, y_test=train_test_split (X,y, test_size=0.3)

Decision_tree_model = DecisionTreeClassifier()
Logistic_regression_Model = LogisticRegression(solver='lbfgs',max_iter=10000)
SVM_model=svm.SVC(kernel='linear')
RF_model=RandomForestClassifier(n_estimators=100)
#Train the models using the training sets
input("Press enter to train the model")
Decision_tree_model.fit(X_train, y_train)
Logistic_regression_Model.fit(X_train, y_train)
SVM_model.fit(X_train, y_train)
RF_model.fit(X_train,y_train)
#Predict the Model
DT_Prediction =Decision_tree_model.predict(x_test)
LR_Prediction =Logistic_regression_Model.predict(x_test)
SVM_Prediction =SVM_model.predict(x_test)
RF_Prediction =RF_model.predict(x_test)
# Calculation of Model Accuracy
input("Press enter to view the accuracy")
DT_score=accuracy_score(y_test, DT_Prediction)
lR_score=accuracy_score(y_test, LR_Prediction)
SVM_score=accuracy_score(y_test, SVM_Prediction)
RF_score=accuracy_score(y_test, RF_Prediction)
# Display Accuracy
input("Press enter to view the best accurate model")
print ("Decistion Tree accuracy =", DT_score*100,"%")
print ("Logistic Regression accuracy =", lR_score*100,"%")
print ("Suport Vector Machine accuracy =", SVM_score*100,"%")
print ("Random Forest accuracy =", RF_score*100,"%")
#Create a Persisting Model
input("\nPress enter to create persisting model")
joblib.dump(RF_model, 'recommendation.joblib')
input("\nPress enter to use persisting mode;")
comb_code=int(input("Enter your combination code :"))
grade=int (input ("Enter your grade :"))
#Predict from the created model
input("\nPress enter to predict from the created model")
model=joblib.load('recommendation.joblib')
predictions = RF_model.predict ([[comb_code,grade]])
print("The observation is:",predictions)