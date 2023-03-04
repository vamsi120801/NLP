# importing the libraries
import pandas as pd
import nltk, re, string
from nltk.stem import WordNetLemmatizer, PorterStemmer
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.util import ngrams
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
import itertools
from sklearn import metrics
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_score, recall_score
from xgboost import XGBClassifier
import streamlit as st 
from textblob import TextBlob

def clean_text(text):
    #1) converting all characters to lower case
    text = text.lower()
    #2) removing punctuations
    punc = str.maketrans(string.punctuation, ' '*len(string.punctuation))
    text = text.translate(punc)
    #2) removing extra white spaces
    text = re.sub('\s+', ' ', text).strip()
    #4) removing special characters
    text = re.sub('[^A-Za-z0-9\s]+', '', text)
    #5) removing numbers
    text = re.sub('\d+', '', text)
    ###5) removing punctuations
    ###text = re.sub('[%s]' % re.escape(string.punctuation), '', text)
    #6) removing links
    text = re.sub('https?://\S+|www\.\S+', '', text)
    #7) Deleting newlines
    text = re.sub('\n', '', text)
    lemmatizer = WordNetLemmatizer()
    Stopwords = set(nltk.corpus.stopwords.words("english")) - set(["not"])
    text = text.lower()
    tokens = nltk.word_tokenize(text)
    tokens = [token for token in tokens if token not in Stopwords ]
    tokens = [lemmatizer.lemmatize(token) for token in tokens]
    text = " ".join(tokens)
    return text

# Define function for sentiment analysis using TextBlob
def get_sentiment(text):
    blob = TextBlob(text)
    sentiment = blob.sentiment.polarity
    if sentiment >= 0.1:
        return 'Positive'
    elif sentiment <= -0.1:
        return 'Negative'
    else:
        return 'Neutral'


def build_model(data):
    X_train_t_xg, X_test_t_xg, y_train_t_xg, y_test_t_xg = train_test_split(X, y, test_size=0.2, random_state=42)

# Initialize a dictionary to store the ngram_range and corresponding shapes

# Loop through different ngram_ranges and fit and transform the features
    
# Vectorize the features using a CountVectorizer with the current ngram_range and max_features=10000
    tfidf_vectorizer  = TfidfVectorizer(ngram_range=(1,3), max_features=10000)
    X_train_transformed_t_xg = tfidf_vectorizer.fit_transform(X_train_t_xg)
    X_test_transformed_t_xg = tfidf_vectorizer.transform(X_test_t_xg)
    
   
    # Calculate the class weights
    class_weights = {
        2: 1.0,  # Neutral class weight
        1: len(y_train_t_xg) / sum(y_train_t_xg==1),  # Positive class weight
        0: len(y_train_t_xg) / sum(y_train_t_xg==0)  # Negative class weight
        }
    # Initialize a Naive Bayes model with weighted loss function
    model = XGBClassifier(learning_rate=0.2,max_depth=30, n_estimators=1000, gamma=0.5, reg_alpha=0.5)

    # Fit the model with weighted loss function
    model.fit(X_train_transformed_t_xg, y_train_t_xg, sample_weight=[class_weights[c] for c in y_train_t_xg])

    # Predict the sentiment for the testing set
    y_pred_t_xg = model.predict(X_test_transformed_t_xg)
# Perform sentiment analysis on text(s)
    a = print(metrics.classification_report(y_test_t_xg, y_pred_t_xg))  
    return model,a, tfidf_vectorizer

# Set up Streamlit app
st.title('Financial Sentiment Analysis')
text_input = st.text_area('Enter financial text(s) (one per line)')

data = pd.read_csv("C:/Users/Adithya/OneDrive/Documents/Projects/Project-2/financial_sentiment_data.csv", index_col = False)
# Split data into features and target
X = data['Filtered_Sentence']
y = data['Predicted_Sentiment']
model, accuracy, tfidf_vectorizer = build_model(data)
# Perform sentiment analysis on text(s)
if text_input:
    texts = text_input.split('\n')
    cleaned_texts = [clean_text(text) for text in texts] # Clean the text(s)
    text_sentiments = []
    for text in cleaned_texts:
        text_sentiment = model.predict(tfidf_vectorizer.transform([text]))[0]
        text_sentiments.append(text_sentiment)
    sentiment_df = pd.DataFrame({'Text': texts, 'Sentiment': text_sentiments})
    st.write(sentiment_df)
else:
    st.write('Enter financial text(s) to get started.')




