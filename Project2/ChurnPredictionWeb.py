import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import StandardScaler
import pickle
import streamlit as st

loaded_model = pickle.load(open('C:/Users/hp/Desktop/All/projects/Project2/trained_model.sav','rb'))

def prediction(New_example):
    # New_example=np.array([502,'France','Female',42,8,159660.80,3,1,0,113931.57])
    encoded=LabelEncoder().fit_transform(New_example)
    encoded=encoded.reshape(1,-1)
    new_pred=loaded_model.predict(encoded)
    return (new_pred) 

def main():
    st.title('Churn Prediction')

    CreditScore = st.text_input('CreditScore')
    Geography = st.text_input('Geography')
    Gender = st.text_input('Gender')
    Age = st.text_input('Age')
    Tenure = st.text_input('Tenure')
    Balance = st.text_input('Balance')
    NumOfProducts = st.text_input('NumOfProducts')
    HasCrCard = st.text_input('HasCrCard')
    IsActiveMember = st.text_input('IsActiveMember')
    EstimatedSalary = st.text_input('EstimatedSalary')

    output = ''

    if st.button("RESULT"):
        out = prediction([CreditScore,Geography,Gender,Age,Tenure,Balance,NumOfProducts,HasCrCard,IsActiveMember,EstimatedSalary])
        if out == 0:
            output = "NO CHURN"
        else:
            output = "LIKELY TO CHURN"

    st.success(output)	


if __name__=='__main__':
    main()						

