import streamlit as st
import pickle
import string
import nltk
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer
ps=PorterStemmer()

def text_transformation(text):
    # To Lower Case
    text = text.lower()

    # Tokenization
    text = nltk.word_tokenize(text)

    y = []
    for i in text:
        if i.isalnum():
            y.append(i)

    # Removing Stopwords and punctuation
    text = y[:]  # Cloning list
    y.clear()

    for i in text:
        if i not in stopwords.words('english') and i not in string.punctuation:
            y.append(i)

    # Stemming
    text = y[:]
    y.clear()

    for i in text:
        y.append(ps.stem(i))

    return " ".join(y)


tfidf = pickle.load(open('vectorizer.pkl', 'rb'))
model = pickle.load(open('model.pkl', 'rb'))

st.title('SMS Spam Detection System')

input_sms = st.text_area('Enter the message')

if st.button('Predict'):

    # Preprocess
    transformed_sms = text_transformation(input_sms)
    # Vectorize
    vector_input = tfidf.transform([transformed_sms])

    # Predict
    result=model.predict(vector_input)[0]

    # Display
    if result==1:
        st.error('Spam')
    else:
        st.success('Not Spam')