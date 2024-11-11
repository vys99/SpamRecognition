import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
from sklearn.feature_extraction.text import CountVectorizer
import tkinter as tk





df = pd.read_csv(r"C:\Users\vysak\Downloads\spam.csv", encoding="ISO-8859-1")

df=df.iloc[:,0:2]

df['spam'] = df['v1'].apply(lambda x:1 if x == 'spam' else 0)

xtrain,xtest,ytrain,ytest=train_test_split(df.v2,df.spam,test_size=0.2)

v=CountVectorizer()

xtraincv=v.fit_transform(xtrain)

from sklearn.naive_bayes import MultinomialNB

model=MultinomialNB()
model.fit(xtraincv,ytrain)

xtestcv=v.transform(xtest)

pred=model.predict(xtestcv)

import streamlit as st

st.markdown(
    """
        <style>
    
    .stApp {
        background-color: #90EE90; /* Replace with your preferred color */
    }

    
    </style>
    """,
    unsafe_allow_html=True
)

st.markdown(f"<p style='font-size:50px ; font-family:Helvetica; color:beige; text-align:center'>SPAM RECOGNITION</p>", unsafe_allow_html=True)

inp=st.text_input("Enter your Message",key="big-font-input")

if st.button("submit"):
    inpcv=v.transform([inp])
    predi=model.predict(inpcv)
    if predi[0]==1:
        res="The message is SPAM"
    elif predi[0]==0:
        res="The message is HAM"
    st.markdown(f"<p style='font-size:30px; font-family:Verdana; color:green; text-align:center'>{res}</p>", unsafe_allow_html=True)

    


# inp=["Are you available for a call this afternoon? Let me know what time works for you.","Please review the attached report and let me know your feedback."]
# inpcv=v.transform(inp)
# pred=model.predict(inpcv)
# pred