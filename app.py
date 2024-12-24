import numpy as np
import tensorflow as tf
from tensorflow.keras.datasets import imdb
from tensorflow.keras.preprocessing import sequence
from tensorflow.keras.models import load_model

word_index = imdb.get_word_index()
reverse_word_index = {value:key for key,value in word_index.items()}

model = load_model('simple_rnn_imdb.h5')

def decode_review(encoded_review):
    return " ".join([reverse_word_index.get(i-3,'?') for i in encoded_review])

def preprocessText(text):
    words = text.lower().split()
    encoded_review = [word_index.get(word,2)+3 for word in words]
    padded_review = sequence.pad_sequences([encoded_review],maxlen=500)
    return padded_review

def predict_sentiment(review):
    prediction = model.predict(preprocessed_text)
    sentiment  = 'Positive' if prediction>0.5 else 'Negative'
    return sentiment, prediction[0][0]



import streamlit as stl
stl.title('IMDB Movie Review Sentiment Analysis')
stl.write('Enter a movie review to classify as postive or negative')

user_input = stl.text_area('Movie Review')

if(stl.button('Classify')):
    preprocessed_text = preprocessText(user_input)
    sentiment, prediction = predict_sentiment(preprocessed_text)
    
    stl.write(f'Sentiment : {sentiment}')

else:
    stl.write('Please Enter a movie review') 