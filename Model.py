# -*- coding: utf-8 -*-
"""
Created on Mon Jan  3 11:53:15 2022

@author: mathi
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import GridSearchCV, train_test_split
from sklearn.preprocessing import StandardScaler

from sklearn.linear_model import LogisticRegression, RidgeClassifier
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier

from sklearn.metrics import accuracy_score, confusion_matrix, f1_score

data = pd.read_csv('drug_consumption.data', sep=",",header=None)
data=data.rename(columns={0:'ID',1:'Age',2:'Gender',3:'Education',4:'Country',5:'Ethnicity',6:'Nscore',7:'Escore',8:'Oscore',9:'Ascore',10:'Cscore',11:'Impulsive',12:'SS',13:'Alcohol',14:'Amphet',15:'Amyl',16:'Benzos',17:'Caff',18:'Cannabis',19:'Choc',20:'Coke',21:'Crack',22:'Ecstasy',23:'Heroin',24:'Ketamine',25:'Legalh',26:'LSD',27:'Meth',28:'Mushrooms',29:'Nicotine',30:'Semer',31:'VSA'})
data=data.drop(columns=['ID'])

# On enlève les données relative au Semer, n'ayant pas de pertinence dans notre étude
data = data.drop(data[data['Semer'] != 'CL0'].index)

# On va également enlever les colonnes sans utilité dans notre étude
data = data.drop(['Choc','Semer'], axis=1)
data = data.reset_index(drop=True)

# On encode les string par des valeurs numériques

drugs = ['Alcohol',
         'Amyl',
         'Amphet',
         'Benzos',
         'Caff',
         'Cannabis',
         'Coke',
         'Crack',
         'Ecstasy',
         'Heroin',
         'Ketamine',
         'Legalh',
         'LSD',
         'Meth',
         'Mushrooms',
         'Nicotine',
         'VSA'    ]

def drug_encoder(x):
    if x == 'CL0':
        return 0
    elif x == 'CL1':
        return 1
    elif x == 'CL2':
        return 2
    elif x == 'CL3':
        return 3
    elif x == 'CL4':
        return 4
    elif x == 'CL4':
        return 5
    elif x == 'CL5':
        return 6
    else:
        return 7

         
for column in drugs:
    data[column] = data[column].apply(drug_encoder)


# MATRICE DE CORRELATIO

# On enlève les colonnes avec les corrélations les plus basses

low_corr = ['Age', 'Gender', 'Education', 'Alcohol','Ascore','Caff']
for column in low_corr:
    data = data.drop(column, axis=1)
data.head()

def preprocessing_inputs(df, column):
    df = df.copy()
    
    # Split df into X and y
    y = df[column]
    X = df.drop(column, axis=1)
    
    # Train-test split
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=1)  # 80% train et 20% test
    
    # Scale X
    scaler = StandardScaler()
    scaler.fit(X_train)
    
    X_train = pd.DataFrame(scaler.transform(X_train), 
                           index=X_train.index, 
                           columns=X_train.columns)
    X_test = pd.DataFrame(scaler.transform(X_test), 
                          index=X_test.index, 
                          columns=X_test.columns)
    
    return X_train, X_test, y_train, y_test
    


# Fonction pour matrice de confusion 

def plot_confusion_matrix(y,y_predict):
    #Function to easily plot confusion matrix
    cm = confusion_matrix(y, y_predict)
    ax= plt.subplot()
    sns.heatmap(cm, annot=True, ax = ax, fmt='g', cmap='Blues');
    ax.set_xlabel('Predicted labels')
    ax.set_ylabel('True labels')
    ax.set_title('Confusion Matrix'); 
    ax.xaxis.set_ticklabels(['non-user', 'user']); ax.yaxis.set_ticklabels(['non-user', 'user'])

### Nicotine Consumption Risk Prediction (NCRP)

# Prediction for Nicotine
# On crée une autre colonne pour indiquer les fumeurs et les non fumeurs (1 et 0)

nic_df = data.copy()
nic_df['Nicotine_User'] = nic_df['Nicotine'].apply(lambda x: 1 if x not in [0,1] else 0)
nic_df = nic_df.drop(['Nicotine'], axis=1)

X_train, X_test, y_train, y_test = preprocessing_inputs(nic_df, 'Nicotine_User')


#On entraine les différents modèles de prédictions 

models = {
            '     Logisitc Regression': LogisticRegression(),
            '        Ridge Classifier': RidgeClassifier(),
            ' Support Vector Machines': SVC(),
            'Random Forest Classifier': RandomForestClassifier()}

for name, model in models.items():
    model.fit(X_train, y_train)
    print(name + ' trained.')


#Matrice de confusion avec le modèle le plus performant 
model1 = RandomForestClassifier()  # ici c'est le RFC
model1.fit(X_train, y_train)
yhat = model1.predict(X_test)

import pickle as pkl
pkl.dump(model1,open("save","wb"))