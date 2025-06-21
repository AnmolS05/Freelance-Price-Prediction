from flask import Flask, request, jsonify
from flask_cors import CORS  # 👈 ADD THIS
import pickle
from scipy.sparse import hstack

# Load the trained model and vectorizer
model = pickle.load(open("model.pkl", "rb"))
vectorizer = pickle.load(open("vectorizer.pkl", "rb"))

app = Flask(__name__)
CORS(app)  # 👈 ADD THIS TOO to fix your browser CORS error

@app.route("/")
def home():
    return "🔥 Freelance API is running. POST to /predict with job info."

@app.route("/predict", methods=["POST"])
def predict():
    data = request.get_json()

    skills = [data['skills']]  # Expecting plain string like "React Node"
    timeline = [int(data['timeline_days'])]  # e.g., 5

    # Multiply timeline to increase importance, flip to negative for "less = more"
    amplified_timeline = [[-1 * timeline[0] * 100]]

    # Vectorize skill input and combine with timeline
    skill_features = vectorizer.transform(skills)
    final_input = hstack([skill_features, amplified_timeline])

    # Predict price
    predicted_price = model.predict(final_input)[0]
    return jsonify({'predicted_price': round(predicted_price)})

if __name__ == "__main__":
    print("Server running at http://127.0.0.1:5000")
    app.run(debug=True)
