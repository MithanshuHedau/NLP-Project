#!/usr/bin/env python
# coding: utf-8

# In[1]:

# get_ipython().system('pip install pandas numpy scikit-learn nltk matplotlib seaborn')


# In[2]:

# get_ipython().system('pip install tensorflow keras torch transformers')


# In[3]:

import pandas as pd

# Load the dataset (update filename if needed)
file_path = "twitter_sentiment.csv"
df = pd.read_csv(file_path)

# Display first few rows
df.tail()


# In[4]:

import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer
import string

# Download necessary NLTK resources
nltk.download('punkt')
nltk.download('stopwords')
nltk.download('wordnet')


# In[5]:

# Rename columns to simpler names
df.columns = ['id', 'place', 'feedback', 'text']

# Print new column names
print("Updated Column Names:", df.columns)

# Display first few rows
df.head()


# In[6]:

import nltk
nltk.download('punkt')


# In[7]:

import nltk
print(nltk.data.path)


# In[8]:

import pandas as pd
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer
import string

# Download necessary NLTK resources
nltk.download('punkt')
nltk.download('stopwords')
nltk.download('wordnet')

# Initialize lemmatizer and stopwords
lemmatizer = WordNetLemmatizer()
stop_words = set(stopwords.words('english'))

def preprocess_text(text):
    if not isinstance(text, str):  # Handle NaN or non-string values
        return ""
    text = text.lower()  # Convert to lowercase
    tokens = word_tokenize(text)  # Tokenization
    tokens = [word for word in tokens if word.isalpha()]  # Remove punctuation
    tokens = [word for word in tokens if word not in stop_words]  # Remove stopwords
    tokens = [lemmatizer.lemmatize(word) for word in tokens]  # Lemmatization
    return " ".join(tokens)

# Apply preprocessing safely
df['text'] = df['text'].astype(str)  # Ensure all values in 'text' column are strings
df['clean_text'] = df['text'].apply(preprocess_text)  

df.head()


# In[9]:

df.tail()


# #  Convert Text into Vectors (TF-IDF or CountVectorizer)

# In[10]:

from sklearn.feature_extraction.text import TfidfVectorizer

# Convert text to numerical vectors
vectorizer = TfidfVectorizer(max_features=5000)
X = vectorizer.fit_transform(df['clean_text'])  # Use 'clean_text' for processing

# Use 'feedback' as the target variable
y = df['feedback']  

print(X.shape, y.shape)  # Verify shapes


# In[11]:

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report

# Split dataset into training and testing
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train model
model = LogisticRegression()
model.fit(X_train, y_train)

# Predictions
y_pred = model.predict(X_test)

# Evaluation
print("Accuracy:", accuracy_score(y_test, y_pred))
print(classification_report(y_test, y_pred))


# In[12]:

import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

# Split dataset into training and testing
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train model
model = LogisticRegression()
model.fit(X_train, y_train)

# Predictions
y_pred = model.predict(X_test)

# Evaluation Metrics
accuracy = accuracy_score(y_test, y_pred)
print(f"\n✨ Model Accuracy: {accuracy:.2%} ✨\n")
print(classification_report(y_test, y_pred))

# 📊 1. Confusion Matrix Visualization
plt.figure(figsize=(6,4))
cm = confusion_matrix(y_test, y_pred)
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=model.classes_, yticklabels=model.classes_)
plt.xlabel("Predicted Label")
plt.ylabel("True Label")
plt.title("Confusion Matrix")
plt.show()

# 📊 2. Bar Chart for Class Distribution
plt.figure(figsize=(6,4))
sns.countplot(x=y, palette="viridis")
plt.xlabel("Sentiment Labels")
plt.ylabel("Count")
plt.title("Class Distribution")
plt.show()


# In[13]:

import pickle

# Save the model and vectorizer
with open('model.pkl', 'wb') as model_file:
    pickle.dump(model, model_file)

with open('vectorizer.pkl', 'wb') as vectorizer_file:
    pickle.dump(vectorizer, vectorizer_file)

def predict_sentiment(text):
    processed_text = preprocess_text(text)  # Preprocess new text
    vectorized_text = vectorizer.transform([processed_text])  # Convert to vector
    prediction = model.predict(vectorized_text)[0]  # Get prediction
    return prediction

# Example
user_input = "I went to the store to buy some groceries today."
print("Predicted Sentiment:", predict_sentiment(user_input))


# In[14]:

user_input = "This product broke after just one day. Very disappointed."
print("Predicted Sentiment:", predict_sentiment(user_input))


# In[15]:

user_input = "I didn’t expect much, but this turned out to be one of the most exhilarating experiences of my life."
print("Predicted Sentiment:", predict_sentiment(user_input))


# In[16]:

def predict_sentiment():
    user_input = input("Enter a sentence: ")  
    processed_text = preprocess_text(user_input)  
    vectorized_text = vectorizer.transform([processed_text])  
    prediction = model.predict(vectorized_text)[0]  
    print("Predicted Sentiment:", prediction)

# Run the function
predict_sentiment()