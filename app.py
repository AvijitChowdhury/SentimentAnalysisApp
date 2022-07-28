# Library imports
import string

import joblib
import nltk
import numpy as np
import pandas as pd
import spacy
from flask import Flask, jsonify, render_template, request
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.pipeline import Pipeline
from sklearn.svm import LinearSVC
from spacy.lang.en.stop_words import STOP_WORDS

# Load trained Pipeline
model = joblib.load('./c1_SentimentAnalysis_Model_Pipeline.pkl')

stopwords = list(STOP_WORDS)

# Create the app object
app = Flask(__name__)


# creating a function for data cleaning
from custom_tokenizer_function import CustomTokenizer


# Define predict function
@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict',methods=['POST'])
def predict():
    new_review = [str(x) for x in request.form.values()]
#     data = pd.DataFrame(new_review)
#     data.columns = ['new_review']

    predictions = model.predict(new_review)[0]
    if predictions==0:
        return render_template('index.html', prediction_text='Negative')
    else:
        return render_template('index.html', prediction_text='Positive')


if __name__ == "__main__":
    app.run(debug=True)
