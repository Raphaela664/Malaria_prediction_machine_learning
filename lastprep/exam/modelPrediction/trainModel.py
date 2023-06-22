import pandas as pd
import joblib as joblib
from sklearn.tree import DecisionTreeClassifier
from sklearn.linear_model import LogisticRegression
from sklearn import svm
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import numpy as np

df = pd.read_csv('malaria_clinical_data.csv')
print(df.columns)

# print(df.info())
#median Temp
medianTemp = df['temperature'].median()
print('The median is', medianTemp)
df['temperature'].fillna(medianTemp, inplace = True)
#median Parasite
medianPar = df['parasite_density'].median()
print('The median is', medianPar)
df['parasite_density'].fillna(medianPar, inplace = True)
#median wbc
medianwbc = df['wbc_count'].median()
print('The median is', medianwbc)
df['wbc_count'].fillna(medianwbc, inplace = True)
#median hb
medianhb = df['hb_level'].median
df['hb_level'].fillna(medianhb, inplace = True)
#median hema
medianHema = df['hematocrit'].median()
print('The median is', medianHema)
df['hematocrit'].fillna(medianHema, inplace = True)
#median mean_Cell_volume
medianCellVolume = df['mean_cell_volume'].median()
print('The median is', medianCellVolume)
df['mean_cell_volume'].fillna(medianCellVolume, inplace = True)

#median mean_corp_hb
medianCorpHb = df['mean_corp_hb'].median()
print('The median is', medianCorpHb)
df['mean_corp_hb'].fillna(medianCorpHb, inplace = True)

#median mean_cell_hb_conc
medianCellHbConc = df['mean_cell_hb_conc'].median()
print('The median is', medianCellHbConc)
df['mean_cell_hb_conc'].fillna(medianCellHbConc, inplace = True)


#median platelet_count
medianPlatCount = df['platelet_count'].median()
print('The median is', medianPlatCount)
df['platelet_count'].fillna(medianPlatCount, inplace = True)

#median platelet_Width
medianPlatWidth = df['platelet_distr_width'].median()
print('The median is', medianPlatWidth)
df['platelet_distr_width'].fillna(medianPlatWidth, inplace = True)

#median platelet_Vl
medianPlatVl = df['mean_platelet_vl'].median()
print('The median is', medianPlatVl)
df['mean_platelet_vl'].fillna(medianPlatVl, inplace = True)

#median neutroPercent
medianneutroPercent= df['neutrophils_percent'].median()
print('The median is', medianneutroPercent)
df['neutrophils_percent'].fillna(medianneutroPercent, inplace = True)

#median lymphoPercent
medianlymphoPercent= df['lymphocytes_percent'].median()
print('The median is', medianlymphoPercent)
df['lymphocytes_percent'].fillna(medianlymphoPercent, inplace = True)

#median mixedCellsPercent
medianmixedCellsPercent= df['mixed_cells_percent'].median()
print('The median is', medianmixedCellsPercent)
df['mixed_cells_percent'].fillna(medianmixedCellsPercent, inplace = True)

#median neutroCount
medianneutroCount= df['neutrophils_count'].median()
print('The median is', medianneutroCount)
df['neutrophils_count'].fillna(medianneutroCount, inplace = True)


#median lymphoCount
lymphoCount = df['lymphocytes_count'].median()
print('The median is', lymphoCount)
df['lymphocytes_count'].fillna(lymphoCount, inplace = True)

#median mixedCellsCount
mixedCellsCount = df['mixed_cells_count'].median()
print('The median is', mixedCellsCount)
df['mixed_cells_count'].fillna(mixedCellsCount, inplace = True)

print(df.duplicated().to_string())
df.drop_duplicates(inplace=True)

X=df.drop(columns=['SampleID', 'consent_given', 'location', 'Enrollment_Year', 'bednet',
       'fever_symptom', 'Suspected_Organism',
       'Suspected_infection', 'RDT', 'Blood_culture', 'Urine_culture',
       'Taq_man_PCR', 'Microscopy', 'Laboratory_Results',
       'Clinical_Diagnosis', 'rbc_count',
       'RBC_dist_width_Percent'])
y=df['Clinical_Diagnosis']
X_train, x_test, y_train, y_test=train_test_split(X,y, test_size=0.3)

# Decision_tree_model = DecisionTreeClassifier()
Logistic_regression_Model = LogisticRegression(solver='lbfgs',max_iter=10000)
# SVM_model=svm.SVC(kernel='linear')
# RF_model=RandomForestClassifier(n_estimators=100)
#Train the models using the training sets
#input("Press enter to train the model")
# Decision_tree_model.fit(X_train, y_train)
Logistic_regression_Model.fit(X_train, y_train)
# SVM_model.fit(X_train, y_train)
# RF_model.fit(X_train,y_train)
#Predict the Model
# DT_Prediction =Decision_tree_model.predict(x_test)
LR_Prediction =Logistic_regression_Model.predict(x_test)
# SVM_Prediction =SVM_model.predict(x_test)
# RF_Prediction =RF_model.predict(x_test)
# Calculation of Model Accuracy
# #input("Press enter to view the accuracy")
# DT_score=accuracy_score(y_test, DT_Prediction)
# lR_score=accuracy_score(y_test, LR_Prediction)
# SVM_score=accuracy_score(y_test, SVM_Prediction)
# RF_score=accuracy_score(y_test, RF_Prediction)
# # Display Accuracy
# #input("Press enter to view the best accurate model")
# print ("Decistion Tree accuracy =", DT_score*100,"%")
# print ("Logistic Regression accuracy =", lR_score*100,"%")
# print ("Suport Vector Machine accuracy =", SVM_score*100,"%")
# print ("Random Forest accuracy =", RF_score*100,"%")
#Create a Persisting Model
#input("\nPress enter to create persisting model")
joblib.dump(Logistic_regression_Model, 'predictions.joblib')
#input("\nPress enter to use persisting mode;")
temperature=int(input("Enter your temperature :"))
parasite_density=int(input("Enter your parasite_density :"))
wbc_count =int(input("Enter your wbc_count :"))
hb_level =int(input("Enter your hb_level :"))
hematocrit =int(input("Enter your hematocrit :"))
mean_cell_volume =int(input("Enter your mean_cell_volume :"))
mean_corp_hb =int(input("Enter your mean_corp_hb :"))
mean_cell_hb_conc =int(input("Enter your mean_cell_hb_conc :"))
platelet_count =int(input("Enter your platelet_count :"))
platelet_distr_width = int(input("Enter your platelet_distr_width :"))
mean_platelet_vl = int(input("Enter your mean_platelet_vl :"))
neutrophils_percent =int(input("Enter your neutrophils_percent :")) 
lymphocytes_percent= int(input("Enter your lymphocytes_percent :"))
mixed_cells_percent= int(input("Enter your mixed_cells_percent :")) 
neutrophils_count= int(input("Enter your neutrophils_count :"))
lymphocytes_count= int(input("Enter your lymphocytes_count :"))
mixed_cells_count= int(input("Enter your mixed_cells_count :"))

#Predict from the created model
#input("\nPress enter to predict from the created model")
model=joblib.load('predictions.joblib')
predictions = Logistic_regression_Model.predict ([[temperature,parasite_density,wbc_count,hb_level,hematocrit,mean_cell_volume,
                  mean_corp_hb,mean_cell_hb_conc,platelet_count,platelet_distr_width, mean_platelet_vl
                  ,neutrophils_percent,lymphocytes_percent,mixed_cells_percent,neutrophils_count,lymphocytes_count,
                  mixed_cells_count]])
print("The observation is:",predictions)