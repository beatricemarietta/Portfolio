pip install scikit-learn
import pandas as pd
import numpy as np
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

df = pd.read_excel('/content/drive/MyDrive/DATA PORTFOLIO/Data Penderita Liver.xlsx')
df.head()

df.info()

#Data PreProcessing
def categorize_age(age):
    if age < 33:
        return 1
    elif 33 <= age <= 61:
        return 2
    else:
        return 3
df['Age'] = df['Age'].apply(categorize_age)

def categorize_TB(TB):
    if 0.2 <= TB <= 0.9:
        return 1
    else:
        return 2
df['TB'] = df['TB'].apply(categorize_TB)

def categorize_DB(DB):
    if 0.1 <= DB <= 0.4:
        return 1
    else:
        return 2
df['DB'] = df['DB'].apply(categorize_DB)

def categorize_Alkphos(Alkphos):
    if 45 <= Alkphos <= 115:
        return 1
    else:
        return 2
df['Alkphos'] = df['Alkphos'].apply(categorize_Alkphos)

def categorize_SGPT(SGPT):
    if 7 <= SGPT <= 55:
        return 1
    else:
        return 2
df['SGPT'] = df['SGPT'].apply(categorize_SGPT)

def categorize_SGOT(SGOT):
    if 8 <= SGOT <= 48:
        return 1
    else:
        return 2
df['SGOT'] = df['SGOT'].apply(categorize_SGOT)

def categorize_TP(TP):
    if 6 <= TP <= 8:
        return 1
    else:
        return 2
df['TP'] = df['TP'].apply(categorize_TP)

def categorize_ALB(ALB):
    if 3 <= ALB <= 5:
        return 1
    else:
        return 2
df['ALB'] = df['ALB'].apply(categorize_ALB)

def categorize_AG(AG):
    if 1.5 <= AG <= 3:
        return 1
    else:
        return 2
df['A/G'] = df['A/G'].apply(categorize_AG)

df['Gender'] = df['Gender'].replace(to_replace={'Male':1, 'Female':2})

for i in df.columns:
    print('unique values in "{}":\n'.format(i),df[i].unique())

targetnames = df['Selector'].unique()
tgname = np.array(['Pasien','Non Pasien'])

(df['Selector'] == 1).sum()
(df['Selector'] == 2).sum()

#Data Determination
d_X = df.drop(columns=['Selector'])
d_Y = df['Selector']

#Split Data
Xtrain, Xtest, Ytrain, Ytest = train_test_split(d_X, d_Y, test_size=0.2, random_state=1)

#Data Processing
clf = DecisionTreeClassifier()
clf.fit(Xtrain, Ytrain)

#Predictinon and Accuracy Checking Data Training
Ypred_tr = clf.predict(Xtrain)
accuracy_tr = accuracy_score(Ytrain, Ypred_tr)
print("Accuracy:", accuracy_tr)

#Predictinon and Accuracy Checking Data Testing
Ypred_ts = clf.predict(Xtest)
accuracy_ts = accuracy_score(Ytest, Ypred_ts)
print("Accuracy:", accuracy_ts)

#Clasification Report
print("\nClassification Report\n")
print(classification_report(Ytest, Ypred, target_names=tgname))

#Confusion Matrix Data Training
Ypred_tr = clf.predict(Xtrain)
print('Actual Pasien & Predicted Pasien:', ((Ytrain == 1) & (Ypred_tr == 1)).sum())
print('Actual Pasien & Predicted Non Pasien:', ((Ytrain == 1) & (Ypred_tr == 2)).sum())
print('Actual Non Pasien & Predicted Pasien:', ((Ytrain == 2) & (Ypred_tr == 1)).sum())
print('Actual Non Pasien & Predicted Non Pasien:', ((Ytrain == 2) & (Ypred_tr == 2)).sum())
conf_matrix_tr = confusion_matrix(Ytrain, Ypred_tr, labels=[1,2])
confusion_tr = pd.DataFrame(conf_matrix_tr, columns=["Predicted Pasien", "Predicted Non Pasien"], index=["Actual Pasien", "Actual Non Pasien"])
print('\n------------- Confusion Matrix Data Training ------------\n')
print(confusion_tr)
print('---------------------------------------------------------\n')
accuracy_tr = accuracy_score(Ytrain, Ypred_tr)
correct_tr = ((Ytrain == 1) & (Ypred_tr == 1)).sum()+((Ytrain == 2) & (Ypred_tr == 2)).sum()
wrong_tr = ((Ytrain == 1) & (Ypred_tr == 2)).sum()+((Ytrain == 2) & (Ypred_tr == 1)).sum()
print('\n--- Data Training Accuracy ---\n')
print('Correct Classified  :', correct_tr)
print('Accuracy            :', round(accuracy_tr*100,2),'%')
print('Correct Classified  :', wrong_tr)
print('Error               :', round(100-accuracy_tr*100,2),'%')
print('------------------------------')

#Confusion Matrix Data Testing
Ypred_ts = clf.predict(Xtest)
print('Actual Pasien & Predicted Pasien:', ((Ytest == 1) & (Ypred_ts == 1)).sum())
print('Actual Pasien & Predicted Non Pasien:', ((Ytest == 1) & (Ypred_ts == 2)).sum())
print('Actual Non Pasien & Predicted Pasien:', ((Ytest == 2) & (Ypred_ts == 1)).sum())
print('Actual Non Pasien & Predicted Non Pasien:', ((Ytest == 2) & (Ypred_ts == 2)).sum())
conf_matrix_ts = confusion_matrix(Ytest, Ypred_ts, labels=[1,2])
confusion_ts = pd.DataFrame(conf_matrix_ts, columns=["Predicted Pasien", "Predicted Non Pasien"], index=["Actual Pasien", "Actual Non Pasien"])
print('\n------------- Confusion Matrix Data Testing -------------\n')
print(confusion_ts)
print('---------------------------------------------------------\n')
accuracy_ts = accuracy_score(Ytest, Ypred_ts)
correct_ts = ((Ytest == 1) & (Ypred_ts == 1)).sum()+((Ytest == 2) & (Ypred_ts == 2)).sum()
wrong_ts = ((Ytest == 1) & (Ypred_ts == 2)).sum()+((Ytest == 2) & (Ypred_ts == 1)).sum()
print('\n--- Data Testing Accuracy ---\n')
print('Correct Classified  :', correct_ts)
print('Accuracy            :', round(accuracy_ts*100,2),'%')
print('Correct Classified  :', wrong_ts)
print('Error               :', round(100-accuracy_ts*100,2),'%')
print('-----------------------------')

#Decision Tree Visualization
import matplotlib.pyplot as plt
from sklearn.tree import DecisionTreeClassifier, plot_tree
plt.figure(figsize=(75,75))
plot_tree(clf, filled=True, feature_names=['Age','Gender','TB','DB','Alkphos','SGPT','SGOT','TP','ALB','A/G'], class_names=tgname)
plt.show()
