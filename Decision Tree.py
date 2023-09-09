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

#Data Determination
d_X = df.drop(columns=['Selector'])
d_Y = df['Selector']

#Split Data
Xtrain, Xtest, Ytrain, Ytest = train_test_split(d_X, d_Y, test_size=0.2, random_state=1)

#Data Processing
clf = DecisionTreeClassifier()
clf.fit(Xtrain, Ytrain)

#Predictinon and Accuracy Checking 
Ypred = clf.predict(Xtest)
accuracy = accuracy_score(Ytest, Ypred)
print("Accuracy:", accuracy)

Ypredtr = clf.predict(Xtrain)
accuracytr = accuracy_score(Ytrain, Ypredtr)
print("Accuracy:", accuracytr)

#Clasification Report
print("\nClassification Report\n")
print(classification_report(Ytest, Ypred, target_names=tgname))

#Decision Tree Visualization
import matplotlib.pyplot as plt
from sklearn.tree import DecisionTreeClassifier, plot_tree
plt.figure(figsize=(75,75))
plot_tree(clf, filled=True, feature_names=['Age','Gender','TB','DB','Alkphos','SGPT','SGOT','TP','ALB','A/G'], class_names=tgname)
plt.show()
