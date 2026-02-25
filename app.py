from flask import Flask, request, jsonify, render_template
import pickle

app = Flask(__name__)

model = pickle.load(open("model.pkl", "rb"))
vectorizer = pickle.load(open("vectorizer.pkl", "rb"))

@app.route("/")
def home():
    return render_template("index.html")

@app.route("/predict", methods=["POST"])
def predict():
    text = request.json["text"]
    text_lower = text.lower()

    if "registration fee" in text_lower:
        return jsonify({"result": "Fake Job ❌"})

    transformed = vectorizer.transform([text])
    prediction = model.predict(transformed)
    probability = model.predict_proba(transformed)
    confidence = round(max(probability[0]) * 100, 2)

    if prediction[0] == 1:
        result = f"Fake Job ❌ ({confidence}% confidence)"
    else:
        result = f"Real Job ✅ ({confidence}% confidence)"

    return jsonify({"result": result})

if __name__ == "__main__":
    app.run(debug=True)