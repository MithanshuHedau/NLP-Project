from flask import Flask, request, jsonify, render_template
import pickle
import os
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer

# Download NLTK resources (ensure they're available)
nltk.download('punkt')
nltk.download('stopwords')
nltk.download('wordnet')

# Initialize Flask app
app = Flask(__name__)

# Load the trained model and vectorizer
with open('model.pkl', 'rb') as model_file:
    model = pickle.load(model_file)

with open('vectorizer.pkl', 'rb') as vectorizer_file:
    vectorizer = pickle.load(vectorizer_file)

# Text Preprocessing
lemmatizer = WordNetLemmatizer()
stop_words = set(stopwords.words('english'))

def preprocess_text(text):
    text = text.lower()
    tokens = word_tokenize(text)
    tokens = [word for word in tokens if word.isalpha()]
    tokens = [word for word in tokens if word not in stop_words]
    tokens = [lemmatizer.lemmatize(word) for word in tokens]
    return ' '.join(tokens)

# Home Route
@app.route('/')
def index():
    return render_template('index.html')

# Prediction Route
@app.route('/predict', methods=['POST'])
def predict():
    try:
        text = request.form['text']
        processed_text = preprocess_text(text)
        vectorized_text = vectorizer.transform([processed_text])
        prediction = model.predict(vectorized_text)[0]

        sentiment = 'Positive' if prediction == 1 else 'Negative'
        return render_template('index.html', input_text=text, result=sentiment)

    except Exception as e:
        return jsonify({'error': str(e)})

# Run the app
if __name__ == '__main__':
    app.run(debug=True)
