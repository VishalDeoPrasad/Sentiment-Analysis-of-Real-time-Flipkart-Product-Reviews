from flask import Flask, render_template, request
import joblib
import re
import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.ensemble import RandomForestClassifier

# Initialize Flask app
app = Flask(__name__)

# nltk.download('stopwords')
# nltk.download('wordnet')
stop_words = set(stopwords.words('english'))
stop_words.discard('not')

def preprocess_text(text):
    # Convert text to lowercase
    text = text.lower()
    
    # Remove non-alphabetic characters using regex
    text = re.sub(r'[^a-zA-Z\s]', '', text)
    
    # Tokenize the text
    tokens = nltk.word_tokenize(text)
    
    # Remove stopwords
    filtered_tokens = [word for word in tokens if word not in stop_words]
    
    # Lemmatize the tokens
    lemmatizer = WordNetLemmatizer()
    normalized_text = ' '.join([lemmatizer.lemmatize(token) for token in filtered_tokens])
    
    return normalized_text
###################################################################################
# Define routes
@app.route('/')
def home():
    return render_template('input.html')

@app.route('/predict', methods=['POST', 'GET'])
def predict():
    if request.method == 'POST':
        review = request.form['review']

        review_tfide = preprocess_text(review)

        cv = joblib.load('model/tfidf_vectorizer.pkl')
        review_tfide = cv.transform([review_tfide]).toarray()

        rf_model = joblib.load('model/rf_model.pkl')
        predict = rf_model.predict(review_tfide)[0]
        if predict == 1:
            sentiment = 'Positive'
        else:
            sentiment = 'Negative'

        return render_template('output.html', review=review, sentiment=sentiment)
###################################################################################
if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0')