import streamlit as st
import pandas as pd
from sklearn import datasets
from sklearn.ensemble import RandomForestClassifier
import pickle as pkl
import numpy as np
from sklearn.model_selection import GridSearchCV, train_test_split
from sklearn.preprocessing import StandardScaler

from sklearn.linear_model import LogisticRegression, RidgeClassifier
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier

from sklearn.metrics import accuracy_score, confusion_matrix, f1_score
#streamlit run C:\Users\mathi\OneDrive\Bureau\API.py


st.set_page_config(page_title="Prédiction de risque d'addiction à la nicotine",page_icon="⚕️",layout="centered",initial_sidebar_state="expanded")


html_temp = """ 
    <div style ="background-color:blue;padding:13px"> 
    <h1 style ="color:black;text-align:center;">Prédiction de risque d'addiction à la nicotine</h1> 
    </div> 
    """
      
# display the front end aspect
st.markdown(html_temp, unsafe_allow_html = True) 
st.subheader('by Mathis')
st.subheader("Bienvenue ! \n Nous allons évaluer grâce aux informations que vous nous donnerez \nles risques qui les votres d'être accro à la nicotine !")
st.text('Voici une légende vous permettant de configurer votre profil :')
st.text('Pays : \n0.24923 Canada \n-0.46841 New Zealand \n-0.28519 Other\n0.21128 Republic of Ireland\n0.96082 UK \n-0.57009 USA\n-0.09765 Australia  ')
st.text('Origines :\n-0.50212 Asian \n-1.10702 Black\n1.90725 Mixed-Black/Asian \n0.12600 Mixed-White/Asian \n-0.22166 Mixed-White/Black \n0.11440 Other \n-0.31685 White ')
st.text('Névrosisme :\nScore moyen par rapport à une auto évalution sur les aspects suivant :\nAnxiété\Colère-Hostilité\Dépression\Timidité sociale\Impulsivité\Vulnérabilité')
st.text('Extraversion :\nScore moyen par rapport à une auto évalution sur les aspects suivant :\nChaleur\Grégarité\Assertivité\Activité\Recherche de sensations\Émotions positives')
st.text('Ouverture :\nScore moyen par rapport à une auto évalution sur les aspects suivant :\nOuverture aux rêveries\Ouverture à l’esthétique\Ouverture aux sentiments\Ouverture aux actions\Ouverture aux idées\Ouverture aux valeurs')
st.text('Consience :\nScore moyen par rapport à une auto évalution sur les aspects suivant :\nCompétence\Ordre\Sens du devoir\Recherche de réussite\Autodiscipline\Délibération')
st.text('Drogues:\n0 = never used the drug\n1 = used it over a decade ago\n2 = in the last decade\n3 = used in the last year\n4 = used in the last month\n5 = used in the last week\n6 = used in the last day')

def User_infos():
    
    Country=st.sidebar.select_slider(
    "De quel pays venez-vous?",
    (0.24923, -0.46841, -0.28519, 0.21128, 0.96082, -0.57009, -0.9765))
    Ethnicity=st.sidebar.select_slider('De quelle origine êtes vous ?',
                                       (-0.50212, -1.10702, 1.90725, 0.12600, -0.22166, 0.11440, -0.31685 ))
    Nscore=st.sidebar.slider('Névrosismes score ',min_value=-2.0,max_value=2.0,step=0.00001)
    Escore=st.sidebar.slider('Extraversion score' ,min_value=-2.0,max_value=2.0,step=0.00001)
    Oscore=st.sidebar.slider("Ouverture score" ,min_value=-2.0,max_value=2.0,step=0.00001)
    Cscore=st.sidebar.slider('Consience score' ,min_value=-2.0,max_value=2.0,step=0.00001)
    Impulsive=st.sidebar.slider('Impulsivité score' ,min_value=-2.0,max_value=2.0,step=0.00001)
    SS=st.sidebar.slider('Recherche de sensation' ,min_value=-2.0,max_value=2.0,step=0.00001)
    Amphet=st.sidebar.select_slider('Amphet', (0,1,2,3,4,5,6))
    Amyl=st.sidebar.select_slider('Amyl', (0,1,2,3,4,5,6))
    Benzos=st.sidebar.select_slider('Benzos', (0,1,2,3,4,5,6))
    Cannabis=st.sidebar.select_slider('Cannabis', (0,1,2,3,4,5,6))
    Coke=st.sidebar.select_slider('Coke', (0,1,2,3,4,5,6))
    Crack=st.sidebar.select_slider('Crack', (0,1,2,3,4,5,6))
    Ecstasy=st.sidebar.select_slider('Ecstasy', (0,1,2,3,4,5,6))
    Heroin=st.sidebar.select_slider('Heroin', (0,1,2,3,4,5,6))
    Ketamine=st.sidebar.select_slider('Ketamine', (0,1,2,3,4,5,6))
    Legalh=st.sidebar.select_slider('Legalh', (0,1,2,3,4,5,6))
    LSD=st.sidebar.select_slider('LSD', (0,1,2,3,4,5,6))
    Meth=st.sidebar.select_slider('Meth', (0,1,2,3,4,5,6))
    Mushrooms=st.sidebar.select_slider('Mushrooms', (0,1,2,3,4,5,6))
    VSA=st.sidebar.select_slider('VSA', (0,1,2,3,4,5,6))


    Info_user={'Country':Country,
        'Ethnicity':Ethnicity,
        'Nscore':Nscore,
        'Escore':Escore,
        'Oscore':Oscore,
        'Cscore':Cscore,
        'Impulsive':Impulsive,
        'SS':SS,
        'Amphet':Amphet,
        'Amyl':Amyl,
        'Benzos':Benzos,
        'Cannabis':Cannabis,
        'Coke':Coke,
        'Crack':Crack,
        'Ecstasy':Ecstasy,
        'Heroin':Heroin,
        'Ketamine':Ketamine,
        'Legalh':Legalh,
        'LSD':LSD,
        'Meth':Meth,
        'Mushrooms':Mushrooms,
        'VSA':VSA}
    
    informations = pd.DataFrame(Info_user,index=[0])
    return informations
    
df=User_infos()

st.subheader('Voici vos informations :')
st.write(df)

import os 

path = os.path.dirname(__file__)
my_file = path+'/save'

Model = pkl.load( open( my_file, "rb" ) )
prediction=Model.predict(df)
st.subheader('Résultat')

if st.button("Predict"):    
  if prediction[0] == 1:
    st.error('Attention! Vous avez des gros risques de devenir addicte à la nicotine')
    
  else:
    st.success('Vous avez peu de chance devenir addicte!')
