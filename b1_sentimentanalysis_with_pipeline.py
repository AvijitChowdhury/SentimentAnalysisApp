# -*- coding: utf-8 -*-
"""b1_SentimentAnalysis_with_Pipeline.ipynb

Automatically generated by Colaboratory.

Original file is located at
    https://colab.research.google.com/drive/1XVun2k7x1UBkRWG0IqbNjDN728UzUpQO

## Plan of Action


1.   We are using **Amazon Alexa Reviews dataset (3150 reviews)**, that contains: **customer reviews, rating out of 5**, date of review, Alexa variant 
2.   First we  **generate sentiment labels: positive/negative**, by marking *positive for reviews with rating >3 and negative for remaining*
3. Then, we **clean dataset through Vectorization Feature Engineering** (TF-IDF) - a popular technique
4. Post that, we use **Support Vector Classifier for Model Fitting** and check for model performance (*we are getting >90% accuracy*)
5. Last, we use our model to do **predictions on real Amazon reviews** using: a simple way and then a fancy way

## Import datasets
"""

import numpy as np
import pandas as pd

from google.colab import drive
drive.mount('/content/drive')

# Commented out IPython magic to ensure Python compatibility.
# %cd /content/drive/My Drive/Project6_SentimentAnalysis_with_Pipeline
!ls

#Loading the dataset
dump = pd.read_csv('a1_AmazonAlexa_ReviewsDataset.tsv',sep='\t') 

dump

"""## Data Pre-Processing"""

dataset = dump[['verified_reviews','rating']]
dataset.columns = ['Review', 'Sentiment']

dataset.head()

# Creating a new column sentiment based on overall ratings
def compute_sentiments(labels):
  sentiments = []
  for label in labels:
    if label > 3.0:
      sentiment = 1
    elif label <= 3.0:
      sentiment = 0
    sentiments.append(sentiment)
  return sentiments

dataset['Sentiment'] = compute_sentiments(dataset.Sentiment)

dataset.head()

# check distribution of sentiments
dataset['Sentiment'].value_counts()

# check for null values
dataset.isnull().sum()

# no null values in the data

"""### Data Transformation"""

x = dataset['Review']
y = dataset['Sentiment']

# import tokenizer_input
from b2_tokenizer_input import CustomTokenizerExample

! cat b2_tokenizer_input.py

# if root form of that word is not proper noun then it is going to convert that into lower form
# and if that word is a proper noun, then we are directly taking lower form,
# because there is no lemma for proper noun

# stopwords and punctuations removed

# let's do a test
token = CustomTokenizerExample()
token.text_data_cleaning("Those were the best days of my life!")

"""### Feature Engineering (TF-IDF)"""

from sklearn.feature_extraction.text import TfidfVectorizer

tfidf = TfidfVectorizer(tokenizer=token.text_data_cleaning)
# tokenizer=text_data_cleaning, tokenization will be done according to this function

"""## Train the model

### Train/ Test Split
"""

from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size = 0.2, stratify = dataset.Sentiment, random_state = 0)

x_train.shape, x_test.shape
# 2520 samples in training dataset and 630 in test dataset

"""### Fit x_train and y_train"""

from sklearn.svm import LinearSVC
from sklearn.pipeline import Pipeline

classifier = LinearSVC()

pipeline = Pipeline([('tfidf',tfidf), ('clf',classifier)])
# it will first do vectorization and then it will do classification

pipeline.fit(x_train, y_train)

"""## Check Model Performance"""

y_pred = pipeline.predict(x_test)

from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

# With pipeline, we don't need to prepare the dataset for testing (x_test)

# confusion_matrix
confusion_matrix(y_test, y_pred)

# we are getting almost 91% accuracy

# classification_report
print(classification_report(y_test, y_pred))

# round(accuracy_score(y_test, y_pred)*100,2)

"""# Model Serialization"""

import joblib
joblib.dump(pipeline,'c1_SentimentAnalysis_Model_Pipeline.pkl')

"""# Predict Sentiments using Model

### Simple way
"""

prediction = pipeline.predict(["Alexa is good"])

if prediction == 1:
  print("Result: This review is positive")
else:
  print("Result: This review is negative")

"""### Fancy way"""

new_review = []
pred_sentiment = []

while True:
  
  # ask for a new amazon alexa review
  review = input("Please type an Alexa review - ")

  if review == 'skip':
    print("See you soon!")
    break
  else:
    prediction = pipeline.predict([review])

    if prediction == 1:
      result = 'Positive'
      print("Result: This review is positive\n")
    else:
      result = 'Negative'
      print("Result: This review is negative\n")
  
  new_review.append(review)
  pred_sentiment.append(result)

Results_Summary = pd.DataFrame(
    {'New Review': new_review,
     'Sentiment': pred_sentiment,
    })

Results_Summary.to_csv("./c2_Predicted_Sentiments.tsv", sep='\t', encoding='UTF-8', index=False)
Results_Summary

"""## From Pipeline Pickle"""

# model = joblib.load('c1_SentimentAnalysis_Model_Pipeline.pkl')

# new_review=["bad product"]

# model.predict(new_review)[0]

"""# Model Deployment"""

! which python

import joblib

joblib.__version__