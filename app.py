import streamlit as st
import pickle
import os
import nltk
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.model_selection import train_test_split
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer

# Download necessary NLTK data
nltk.download('punkt')
nltk.download('stopwords')

ps = PorterStemmer()

# Function to preprocess text
def transform_text(text):
    text = text.lower()
    text = nltk.word_tokenize(text)

    y = [i for i in text if i.isalnum()]
    y = [i for i in y if i not in stopwords.words("english")]
    y = [ps.stem(i) for i in y]

    return " ".join(y)

# Paths
current_dir = os.path.dirname(__file__)
vectorizer_path = os.path.join(current_dir, 'vectorizer.pkl')
model_path = os.path.join(current_dir, 'model.pkl')

# Load or train model/vectorizer
if os.path.exists(vectorizer_path) and os.path.exists(model_path):
    with open(vectorizer_path, 'rb') as f:
        tfidf = pickle.load(f)
    with open(model_path, 'rb') as f:
        model = pickle.load(f)
else:
    # Load sample dataset (you can replace with your own)
    data_url = "https://raw.githubusercontent.com/justmarkham/pycon-2016-tutorial/master/data/sms.tsv"
    df = pd.read_table(data_url, header=None, names=["label", "message"])
    df['label_num'] = df['label'].map({'ham': 0, 'spam': 1})
    df['transformed'] = df['message'].apply(transform_text)

    # Vectorize
    tfidf = TfidfVectorizer()
    X = tfidf.fit_transform(df['transformed'])
    y = df['label_num']

    # Train
    model = MultinomialNB()
    model.fit(X, y)

    # Save
    with open(vectorizer_path, 'wb') as f:
        pickle.dump(tfidf, f)
    with open(model_path, 'wb') as f:
        pickle.dump(model, f)

# Streamlit UI
st.title("📩 Spam Detection App")
st.write("Enter a message below to check if it's spam or not.")

input_sms = st.text_area("🔹 Type your message here:", "")

if st.button("Check Spam"):
    if not input_sms.strip():
        st.warning("⚠️ Please enter a message first.")
    else:
        transformed_sms = transform_text(input_sms)
        vector_input = tfidf.transform([transformed_sms])
        result = model.predict(vector_input)[0]
        if result == 1:
            st.error("🚫 SPAM: This message is likely spam!")
        else:
            st.success("✅ NOT SPAM: This message looks safe.")


##   You can now view your Streamlit app in your browser.

## Local URL: http://localhost:8501
## Network URL: http://192.168.60.58:8501
