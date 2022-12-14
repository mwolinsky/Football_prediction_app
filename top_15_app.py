#!/usr/bin/env python
# coding: utf-8

# In[1]:

import xgboost as xgb
import pandas as pd
import numpy as np
import streamlit as st
from PIL import Image,ImageFile
from numpy import loadtxt
from xgboost import XGBClassifier
import urllib.request
import streamlit.components.v1 as components
#import shap
import lime
from lime.lime_tabular import LimeTabularExplainer
import eli5

#Colocamos el lugar de donde extraer el csv
data_location= 'https://raw.githubusercontent.com/mwolinsky/Ranking_predictor/main/england-premier-league-players-2018-to-2019-stats.csv'

#Leemos el csv con lo jugadores como indice
df= pd.read_csv(data_location)

import pandas as pd
import numpy as np
import base64


#Colocamos el lugar de donde extraer el csv
data_location= 'https://raw.githubusercontent.com/mwolinsky/Ranking_predictor/main/england-premier-league-players-2018-to-2019-stats.csv'

#Leemos el csv con lo jugadores como indice
df= pd.read_csv(data_location)

df_def=df[df.position=="Defender"]

df_def["min_per_total_match"]=df_def.minutes_played_overall/(38*90)

df_def= df_def.replace(-1,99999) 


df_def['top_rank']=df_def.loc[:,['rank_in_league_top_attackers','rank_in_league_top_midfielders','rank_in_league_top_defenders']].apply(lambda x: x.min(),axis=1)


categorical_columns=['position','Current Club','nationality']
for column in categorical_columns:
    dummies = pd.get_dummies(df_def[column], prefix=column,drop_first=True)
    df_def = pd.concat([df_def, dummies], axis=1)
    df_def = df_def.drop(columns=column)

df_def['top_def_15']=df.rank_in_league_top_defenders.apply(lambda x: 1 if x>0 and x<=15 else 0)
df_def['top_15']= df_def.apply(lambda x: 1 if x.top_def_15==1 else 0,axis=1)


X=df_def.loc[:,["min_per_conceded_overall","red_cards_overall","minutes_played_overall","goals_overall","clean_sheets_away"]]
y= df_def.top_15
from sklearn.model_selection import train_test_split

#Con estratificación en y

X_train, X_test, y_train, y_test = train_test_split(X, y, stratify=y,random_state=162)

from sklearn.preprocessing import StandardScaler
scaler = StandardScaler()
X_train = pd.DataFrame(scaler.fit_transform(X_train),columns=X_train.columns)
X_test = pd.DataFrame(scaler.transform(X_test),columns=X_test.columns)

model_def = XGBClassifier(subsample= 0.5,n_estimators=100, max_depth=15, learning_rate= 0.3,colsample_bytree= 0.8999, colsample_bylevel= 0.4)
model_def.fit(X_train, y_train)






df_mid=df[df.position=="Midfielder"]

df_mid["min_per_total_match"]=df_mid.minutes_played_overall/(38*90)

df_mid= df_mid.replace(-1,99999) 


df_mid['top_rank']=df_mid.loc[:,['rank_in_league_top_attackers','rank_in_league_top_midfielders','rank_in_league_top_defenders']].apply(lambda x: x.min(),axis=1)


categorical_columns=['position','Current Club','nationality']
for column in categorical_columns:
    dummies = pd.get_dummies(df_mid[column], prefix=column,drop_first=True)
    df_mid = pd.concat([df_mid, dummies], axis=1)
    df_mid = df_mid.drop(columns=column)

df_mid['top_mid_15']=df.rank_in_league_top_midfielders.apply(lambda x: 1 if x>0 and x<=15 else 0)
df_mid['top_15']= df_mid.apply(lambda x: 1 if x.top_mid_15==1 else 0,axis=1)

df_mid['goals_involved_per_90_overall']= df_mid.goals_involved_per_90_overall/38
df_mid['goals_per_90_overall']= df_mid.goals_per_90_overall/38
df_mid['assists_overall']= df_mid.assists_overall/38
df_mid['min_per_conceded_overall']= df_mid.min_per_conceded_overall/38
df_mid['yellow_cards_overall']= df_mid.yellow_cards_overall/38
df_mid['appearances_overall']= df_mid.appearances_overall/38



X=df_mid.loc[:,["goals_involved_per_90_overall","goals_per_90_overall","assists_overall","min_per_conceded_overall","yellow_cards_overall"]]
y= df_mid.top_15
from sklearn.model_selection import train_test_split

#Con estratificación en y

X_train, X_test, y_train, y_test = train_test_split(X, y, stratify=y,random_state=162)

from sklearn.preprocessing import StandardScaler
scaler = StandardScaler()
X_train = pd.DataFrame(scaler.fit_transform(X_train),columns=X_train.columns)
X_test = pd.DataFrame(scaler.transform(X_test),columns=X_test.columns)


model_mid=XGBClassifier(subsample= 0.6,n_estimators=100, max_depth=15, learning_rate= 0.01,colsample_bytree= 0.6, colsample_bylevel= 0.5)
model_mid.fit(X_train, y_train)


#model_mid=xgb.XGBClassifier()

#model_mid.load_model("C:/Users/Maty/Documents/App_football_preediction/model_mid.json")

#model_att=xgb.XGBClassifier()

#model_att.load_model("C:/Users/Maty/Documents/App_football_preediction/model_att.json")

df_ata=df[df.position=="Forward"]
df_ata["min_per_total_match"]=df_ata.minutes_played_overall/(38*90)

df_ata= df_ata.replace(-1,99999) 


df_ata['top_rank']=df_ata.loc[:,['rank_in_league_top_attackers','rank_in_league_top_midfielders','rank_in_league_top_defenders']].apply(lambda x: x.min(),axis=1)


categorical_columns=['position','Current Club','nationality']
for column in categorical_columns:
    dummies = pd.get_dummies(df_ata[column], prefix=column,drop_first=True)
    df_ata = pd.concat([df_ata, dummies], axis=1)
    df_ata = df_ata.drop(columns=column)

df_ata['top_att_15']=df.rank_in_league_top_attackers.apply(lambda x: 1 if x>0 and x<=15 else 0)
df_ata['top_15']= df_ata.apply(lambda x: 1 if x.top_att_15==1 else 0,axis=1)

df_ata['goals_per_90_overall']= df_ata.goals_per_90_overall/38
df_ata['goals_involved_per_90_overall']= df_ata.goals_involved_per_90_overall/38
df_ata['min_per_conceded_overall']= df_ata.min_per_conceded_overall/38
df_ata['clean_sheets_away']= df_ata.clean_sheets_away/38
df_ata['penalty_goals']= df_ata.penalty_goals/38


#goals_per_90_overall,goals_involved_per_90_overall,min_per_conceded_overall,clean_sheets_away,penalty_goals

X=df_ata.loc[:,["goals_per_90_overall","goals_involved_per_90_overall","min_per_conceded_overall","clean_sheets_away","penalty_goals"]]
y= df_ata.top_15
from sklearn.model_selection import train_test_split

#Con estratificación en y

X_train, X_test, y_train, y_test = train_test_split(X, y, stratify=y,random_state=162)

from sklearn.preprocessing import StandardScaler
scaler = StandardScaler()
X_train = pd.DataFrame(scaler.fit_transform(X_train),columns=X_train.columns)
X_test = pd.DataFrame(scaler.transform(X_test),columns=X_test.columns)

model_att = XGBClassifier(subsample= 0.6,n_estimators=100, max_depth=15, learning_rate= 0.01,colsample_bytree= 0.6, colsample_bylevel= 0.5)
model_att.fit(X_train, y_train)







@st.cache
def st_shap(plot, height=None):
    shap_html = f"<head>{shap.getjs()}</head><body>{plot.html()}</body>"
    components.html(shap_html, height=height)
    
#def explainer(model):
    #explainer0 = shap.TreeExplainer(model)

    #shap_values0= ""
    #return(explainer0,shap_values0)
    
def st_shap(plot, height=None):
    print(type(shap))
    print(dir(shap))
    js=shap.getjs()
    shap_html = f"<head>{js}</head><body>{plot.html()}</body>"
    components.html(shap_html, height=height)
    

    
def welcome(): 
    return 'welcome all'
  
def prediction(x1, x2, x3, x4, x5,model):   
    mw=np.array([x1, x2, x3, x4, x5]).reshape(1,-1)
    prediction = model.predict(mw)
    print(prediction) 
    return prediction 
  
def main(): 
    
    st.title("Objetive")

    st.text("The objetive of this App is to predict if the player will or not be part of the Top\n15 in his position.\nEvery input stat was selected regarding their relevance in the prediction model.\nIn order to use the model in each moment of the whole season,\n all input stats must be divided by the total match played by the team.")

    st.title("Top 15 Prediction") 
    html_temp = ""
    
    ImageFile.LOAD_TRUNCATED_IMAGES = True
    urllib.request.urlretrieve('https://img.freepik.com/vector-premium/silueta-jugador-futbol-ilustracion-bola_62860-180.jpg',"JUGADOR.jpg")
    image = Image.open("JUGADOR.jpg").resize((300,400))
    st.image(image)
    st.markdown(html_temp, unsafe_allow_html = True) 

    position=st.selectbox(
    'Which position is the player?',
    ('Defender', 'Midlefielder', 'Forward'))
    if position=="Defender":
        min_per_conceded_overall = st.number_input("Amount of minutes in which the goal recieve a goal") 
        clean_sheets_away = st.number_input("Clean sheets playing away") 
        red_cards_overall = st.number_input("Red Cards received") 
        goals_overall = st.number_input("Goals") 
        minutes_played_overall= st.number_input("Minutes played per match") 
        result =""
        #explainer_1,shap_values0=explainer(model)
        #shap_value = explainer_1.shap_values(np.array([goals_per_90_overall_input,assists_per_90_overall_input,goals_involved_per_90_overall_input,goals_involved_per_90_overall_input,min_per_conceded_overall_input]).reshape(1,-1))
        #exp = explainer.explain_instance(np.array([min_per_conceded_overall,conceded_per_90_overall,minutes_played_overall,goals_overall,clean_sheets_away]), model.predict_proba, num_features=6)

        
                                        
        
    


        
            

    

        if st.button("Predict"): 
            result =prediction(min_per_conceded_overall,red_cards_overall,goals_overall,minutes_played_overall,clean_sheets_away,model=model_def)
            
            
        
        
            if result==1:
                result='top 15 in Defenders'
            else:
                result= 'not top 15 in Defenders'
                
            st.success('The player is {}'.format(result)) 
        
        
            st.subheader('Analizando la prediccion:')
            test= pd.DataFrame(np.array([min_per_conceded_overall,red_cards_overall,minutes_played_overall,goals_overall,clean_sheets_away]).reshape(1,-1), 
             columns=['min_per_conceded_overall', 
                      'red_cards_overall','minutes_played_overall','goals_overall','clean_sheets_away'])
            

            html_object= eli5.show_prediction(model_def,test,show_feature_values=True,feature_names=['Minutes team Concede a goal', 
                      'Red Cards','Minutes played','Goals','Clean Sheets Away'])

            raw_html = html_object._repr_html_()
            components.html(raw_html,height=200)
            #show_prediction(model_def, np.array([min_per_conceded_overall,conceded_per_90_overall,goals_overall,minutes_played_overall,clean_sheets_away]), show_feature_values=True).format_as_html
            #st_shap(shap.force_plot(explainer_1.expected_value, shap_value, np.array([goals_per_90_overall_input,assists_per_90_overall_input,goals_involved_per_90_overall_input,goals_involved_per_90_overall_input,min_per_conceded_overall_input]).reshape(1,-1)))
            #components.html(exp.as_html(show_table=True), height=800)
                                   
    if position=="Midlefielder":
        goals_involved_per_90_overall = st.number_input("Goals involved per match") 
        assists_overall = st.number_input("Assits") 
        goals_per_90_overall = st.number_input("Goals per 90 minutes") 
        min_per_conceded_overall = st.number_input("Minutes in which team received a goal") 
        yellow_cards_overall= st.number_input("Yellow Cards received") 
        result =""
        #explainer_1,shap_values0=explainer(model)
        #shap_value = explainer_1.shap_values(np.array([goals_per_90_overall_input,assists_per_90_overall_input,goals_involved_per_90_overall_input,goals_involved_per_90_overall_input,min_per_conceded_overall_input]).reshape(1,-1))
        #exp = explainer.explain_instance(np.array([min_per_conceded_overall,conceded_per_90_overall,minutes_played_overall,goals_overall,clean_sheets_away]), model.predict_proba, num_features=6)

      
        if st.button("Predict"): 
            result =prediction(goals_involved_per_90_overall,goals_per_90_overall,assists_overall,min_per_conceded_overall,yellow_cards_overall,model=model_mid)
            
        
        
            if result==1:
                result='top 15 in midfielders'
            else:
                result= 'not top 15 in midfielders'
                
            st.success('The player is {}'.format(result)) 
        
        
            st.subheader('Analizando la prediccion:')
            test= pd.DataFrame(np.array([goals_involved_per_90_overall,goals_per_90_overall,assists_overall,min_per_conceded_overall,yellow_cards_overall]).reshape(1,-1), 
             columns=['goals_involved_per_90_overall', 'goals_per_90_overall',
       'assists_overall', 'min_per_conceded_overall', 'yellow_cards_overall' ])
            

            html_object= eli5.show_prediction(model_mid,test,show_feature_values=True,feature_names=['Goals involved per match', 'Goals per Match',
       'Assists', 'Minutes in which team conced a goal', 'Yellow Cards'])

            raw_html = html_object._repr_html_()
            components.html(raw_html,height=200)
            #st_shap(shap.force_plot(explainer_1.expected_value, shap_value, np.array([goals_per_90_overall_input,assists_per_90_overall_input,goals_involved_per_90_overall_input,goals_involved_per_90_overall_input,min_per_conceded_overall_input]).reshape(1,-1)))
            #components.html(exp.as_html(show_table=True), height=800)
    if position=="Forward":
        goals_per_90_overall = st.number_input("Goals per match") 
        goals_involved_per_90_overall = st.number_input("Goals involved per match") 
        clean_sheets_away = st.number_input("Clean Sheets Away") 
        min_per_conceded_overall = st.number_input("Minutes in which team received a goal") 
        penalty_goals= st.number_input("Penalty Goals") 
        result =""
        #explainer_1,shap_values0=explainer(model)
        #shap_value = explainer_1.shap_values(np.array([goals_per_90_overall_input,assists_per_90_overall_input,goals_involved_per_90_overall_input,goals_involved_per_90_overall_input,min_per_conceded_overall_input]).reshape(1,-1))
        #exp = explainer.explain_instance(np.array([min_per_conceded_overall,conceded_per_90_overall,minutes_played_overall,goals_overall,clean_sheets_away]), model.predict_proba, num_features=6)

        
    


        
            

    

        if st.button("Predict"): 
            result =prediction(goals_per_90_overall, goals_involved_per_90_overall,min_per_conceded_overall,clean_sheets_away,penalty_goals,model=model_att)
            
        
        
            if result==1:
                result='top 15 in Forwards'
            else:
                result= 'not top 15 in Forwards'
                
            st.success('The player is {}'.format(result)) 
        
        
            st.subheader('Analizando la prediccion:')
            test= pd.DataFrame(np.array([goals_per_90_overall, goals_involved_per_90_overall,min_per_conceded_overall,clean_sheets_away,penalty_goals]).reshape(1,-1), 
             columns=['goals_per_90_overall', 'goals_involved_per_90_overall','min_per_conceded_overall', 'clean_sheets_away', 'penalty_goals'])
            
            html_object= eli5.show_prediction(model_att,test,show_feature_values=True,feature_names=['Goals per match', 
                      'Goals involved','Minutes teams conced a goal','Clean Sheets playing away','Penalty goals'])
            

            raw_html = html_object._repr_html_()
            components.html(raw_html,height=200)
            #st_shap(shap.force_plot(explainer_1.expected_value, shap_value, np.array([goals_per_90_overall_input,assists_per_90_overall_input,goals_involved_per_90_overall_input,goals_involved_per_90_overall_input,min_per_conceded_overall_input]).reshape(1,-1)))
            #components.html(exp.as_html(show_table=True), height=800)                                    


if __name__=='__main__': 
        main() 
        


    # In[2]:





    # In[ ]:





    # In[ ]:





    # In[ ]:





    # In[ ]:





    # In[ ]:





    # In[ ]:




