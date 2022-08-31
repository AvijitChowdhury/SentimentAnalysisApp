# Library imports
import pickle
import string

import joblib
import nltk
import numpy as np
import pandas as pd
import spacy
import streamlit as st
from flask import Flask, jsonify, render_template, request
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.pipeline import Pipeline
from sklearn.svm import LinearSVC
from spacy.lang.en.stop_words import STOP_WORDS

model = joblib.load('./c1_SentimentAnalysis_Model_Pipeline.pkl')

stopwords = list(STOP_WORDS)


# creating a function for data cleaning
from custom_tokenizer_function import CustomTokenizer

st.title('Sentiment Analyzer')
text = st.text_area('Enter a text')


if st.button('Predict'):
    new_review = [str(x) for x in text]
    #     data = pd.DataFrame(new_review)
    #     data.columns = ['new_review']

    predictions = model.predict(new_review)[0]

    if predictions==0:
        st.header('Postive')
    else:
        st.header('Negative')    
