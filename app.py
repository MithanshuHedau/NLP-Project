from flask import Flask, request, jsonify, render_template
import pickle
import os
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer

# Ensure NLTK resources are downloaded
nltk.download('punkt')
nltk.download('stopwords')
nltk.download('wordnet')

# Initialize Flask app
app = Flask(__name__)

# Load the trained model and vectorizer
try:
    with open('model.pkl', 'rb') as model_file:
        model = pickle.load(model_file)

    with open('vectorizer.pkl', 'rb') as vectorizer_file:
        vectorizer = pickle.load(vectorizer_file)

    print("✅ Model and vectorizer loaded successfully.")
except Exception as e:
    print(f"❌ Error loading model/vectorizer: {e}")
    exit(1)

# Text Preprocessing (Ensure consistency with train.py)
lemmatizer = WordNetLemmatizer()
stop_words = set(stopwords.words('english'))

def preprocess_text(text):
    # Convert to lowercase
    text = text.lower()

    # Tokenize and remove non-alphabetic characters
    tokens = word_tokenize(text)
    tokens = [word for word in tokens if word.isalpha()]

    # Remove stopwords and apply lemmatization
    tokens = [lemmatizer.lemmatize(word) for word in tokens if word not in stop_words]

    return ' '.join(tokens)

# Prediction Mapping (3 Classes)
sentiment_map = {
    0: 'Negative',
    1: 'Neutral',
    2: 'Positive'
}

# Home Route (Renders Input Form)
@app.route('/')
def index():
    return render_template('index.html')

# Prediction Route
@app.route('/predict', methods=['POST'])
def predict():
    try:
        # Get user input
        text = request.form.get('text', '')

        # Validate input
        if not text.strip():
            return render_template('index.html', error="Input cannot be empty.")

        # Preprocess and vectorize input
        processed_text = preprocess_text(text)

        print(f"📄 Original Input: {text}")
        print(f"🔍 Preprocessed Text: {processed_text}")

        vectorized_text = vectorizer.transform([processed_text])
        print(f"📊 Vectorized Shape: {vectorized_text.shape}")

        # Make prediction
        prediction = model.predict(vectorized_text)[0]
        print(f"🎯 Model Output: {prediction}")

        # Map prediction to sentiment
        if prediction == 0:
            sentiment = 'Negative'
        elif prediction == 1:
            sentiment = 'Neutral'
        elif prediction == 2:
            sentiment = 'Positive'
        else:
            sentiment = f"{prediction}"

        return render_template('index.html', input_text=text, result=sentiment)

    except Exception as e:
        print(f"❌ Error during prediction: {e}")
        return jsonify({'error': str(e)})


# Run Flask App
if __name__ == '__main__':
    app.run(debug=True)
