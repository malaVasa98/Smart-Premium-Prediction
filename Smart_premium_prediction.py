# Import packages
import pandas as pd
import numpy as np
import joblib
import streamlit as st
from streamlit_option_menu import option_menu
import warnings
warnings.filterwarnings('ignore')

# Setting up the Streamlit page
st.title(":green[SMART PREMIUM PREDICTION]")
smart_prem = pd.read_csv('Smart Premium.csv')

# Load the models
loaded_rf = joblib.load('random_forest_smart.pkl')

loaded_ord = joblib.load('ordinal_encoder_smart.pkl')

loaded_ohe = joblib.load('ohe_encoder_smart.pkl')

loaded_sc = joblib.load('scaler_smart.pkl')

col1, col2, col3 = st.columns(3)
CUST_FEED = tuple(smart_prem['Customer Feedback'].unique())
EXER_FREQ = tuple(smart_prem['Exercise Frequency'].unique())
EDN_LVL = tuple(smart_prem['Education Level'].unique())
SMOK_STAT = tuple(smart_prem['Smoking Status'].unique())

GEND = tuple(smart_prem['Gender'].unique())
MAR_STAT = tuple(smart_prem['Marital Status'].unique())
OCCPN = tuple(smart_prem['Occupation'].unique())
LOCN = tuple(smart_prem['Location'].unique())
POL_TYPE = tuple(smart_prem['Policy Type'].unique())
PROP_TYPE = tuple(smart_prem['Property Type'].unique())

with col1:
    age = st.text_input("**Age**",key='age')
    annual_income = st.text_input("**Annual Income**",key='annual_income')
    num_dept = st.text_input("**Number of Dependents**",key='num_dep')
    heal_sc = st.text_input("**Health Score**",key='health')
    prev_cla = st.text_input("**Previous Claims**",key='prev')
    veh_age = st.text_input("**Vehicle Age**",key='vehcle')
    cred_sc = st.text_input("**Credit score**",key='credit')

with col2:
    pol_yr = st.text_input('**Policy Year**',key='yr')
    pol_mon = st.text_input("**Policy Month**",key='mon')
    cust_feed = st.selectbox("**Customer Feedback**",CUST_FEED,index=None,key='cust',placeholder='Select one')
    exer_freq = st.selectbox("**Exercise Frequency**",EXER_FREQ,index=None,key='exer',placeholder='Select one')
    ed_lvl = st.selectbox("**Education Level**",EDN_LVL,index=None,key='edc',placeholder='Select one')
    sm_st = st.selectbox("**Smoking Status**",SMOK_STAT,index=None,key='smoke',placeholder='Select one')
    
with col3:
    gend = st.selectbox("**Gender**",GEND,index=None,key='gend',placeholder='Select one')
    marry = st.selectbox("**Marital Status**",MAR_STAT,index=None,key='mary',placeholder='Select one')
    occup = st.selectbox("**Occupation**",OCCPN,index=None,key='occupy',placeholder='Select one')
    loc = st.selectbox("**Location**",LOCN,index=None,key='loc',placeholder='Select one')
    polcy = st.selectbox("**Policy Type**",POL_TYPE,index=None,key='policy',placeholder='Select one')
    propt = st.selectbox("**Property Type**",PROP_TYPE,index=None,key='property',placeholder='Select one')
    if st.button('Get Premium'):
        X_val = np.array([[float(age),float(annual_income),float(num_dept),float(heal_sc),float(prev_cla),float(veh_age),float(cred_sc),int(pol_yr),int(pol_mon),cust_feed,exer_freq,ed_lvl,sm_st,gend, marry,occup,loc,polcy,propt]])
        X_num = X_val[:,[0,1,2,3,4,5,6,7,8]]
        X_ord = loaded_ord.transform(X_val[:,[9,10,11,12]])
        X_ohe = loaded_ohe.transform(X_val[:,[13,14,15,16,17,18]])
        X_ref = np.concatenate((X_num,X_ord,X_ohe),axis=1)
        X_ref = loaded_sc.transform(X_ref)
        Prem_pred = loaded_rf.predict(X_ref)
        st.write(f"{Prem_pred[0]:.2f}")