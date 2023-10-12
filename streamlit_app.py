import streamlit as st
import pickle
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB
import re
import nltk
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer

# Define the preprocess_text function
def preprocess_text(text):
    text = re.sub('[^a-zA-Z]', ' ', text).lower()
    words = text.split()
    words = [word for word in words if word not in set(stopwords.words('english'))]
    ps = PorterStemmer()
    words = [ps.stem(word) for word in words]
    return ' '.join(words

# Load the trained classifier
with open("classifier_model.pkl", "rb") as file:
    classifier = pickle.load(file)

# Create and fit the CountVectorizer with the same parameters used for training
cv = CountVectorizer(max_features=1500)
cv.fit(corpus)  # Fit the CountVectorizer on your preprocessed training data

# Streamlit web app
st.set_page_config(page_title="Restaurant Sentiment Analysis App", page_icon=":sushi:")
st.title("Welcome to the Restaurant Sentiment Analysis App")

# Header
st.header("Restaurant Sentiment Analysis")
st.subheader("Enter your restaurant review below:")

# Input text box for user to enter a review
user_input = st.text_area("Write your review:")

if st.button("Predict Sentiment"):
    if user_input:
        sample_review = preprocess_text(user_input)
        sample_review_vectorized = cv.transform([sample_review]).toarray()
        predicted_sentiment = classifier.predict(sample_review_vectorized)
        result = 'POSITIVE' if predicted_sentiment[0] == 1 else 'NEGATIVE'
        st.write(f"Predicted sentiment: {result}")

# Footer
st.markdown("---")
st.write("Restaurant Sentiment Analysis App")
st.write("© 2023. All rights reserved.")

# Run the Streamlit app
if __name__ == '__main__':
    st.run()
