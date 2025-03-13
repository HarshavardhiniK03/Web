from flask import Flask, request, jsonify, render_template
import joblib
from sklearn.feature_extraction.text import CountVectorizer

# Load the pre-trained model and vectorizer
model = joblib.load('sentiment_model.pkl')
vectorizer = joblib.load('vectorizer.pkl')

app = Flask(__name__)

@app.route('/')
def index():
    """Render the homepage with a simple post sentiment analysis form."""
    return render_template('index.html')

@app.route('/predict_sentiment', methods=['POST'])
def predict_sentiment():
    """Predict sentiment of a user's post (positive/negative)."""
    data = request.json
    text = data.get('text', '')
    
    # Preprocess and vectorize text
    text_vectorized = vectorizer.transform([text])
    
    # Predict sentiment
    sentiment = model.predict(text_vectorized)[0]
    sentiment_label = 'positive' if sentiment == 1 else 'negative'
    
    return jsonify({'sentiment': sentiment_label})

@app.route('/api/sentiment_trends', methods=['GET'])
def sentiment_trends():
    """Generate sentiment trends over time for visualization."""
    # Dummy data for trends, you can replace this with actual data
    timestamps = ['2025-02-01', '2025-02-02', '2025-02-03']
    positive_sentiments = [30, 40, 35]
    negative_sentiments = [10, 5, 7]
    
    return jsonify({
        'timestamps': timestamps,
        'positive_sentiments': positive_sentiments,
        'negative_sentiments': negative_sentiments
    })

if __name__ == '__main__':
    app.run(debug=True)
