# 🛒 E-commerce Order Completion Prediction

This project predicts whether an e-commerce order will likely be **completed** or **canceled**, using machine learning (scikit-learn) and a Flask API.

---

## 🚀 Setup & Run (All-in-One Guide)

# 1️⃣ Clone the repository
git clone https://github.com/Nurujjaman329/regression_model.git
cd regression_model

# 2️⃣ (Optional) Set up a virtual environment (recommended)
# --- For Windows:
python -m venv venv
venv\Scripts\activate
# --- For macOS/Linux:
python3 -m venv venv
source venv/bin/activate

# 3️⃣ Install dependencies
pip install -r requirements.txt

# 4️⃣ Train the model (⚠️ Must run at least once to create model.pkl & scaler.pkl)
python model.py

# ✅ This will generate:
# - model.pkl
# - scaler.pkl

# 5️⃣ Run the Flask app
python app.py

# 🚀 Flask API will start at: http://127.0.0.1:5000

# --------------------------------------------
# ▶️ How to test:

# ✅ Option 1: Use the built-in HTML form UI
# 🔗 Open your browser and visit:
http://127.0.0.1:5000

# You’ll see a web form where you can input data and submit to get predictions.

# ✅ Option 2: Test with Postman or CURL
# Example CURL:
curl -X POST http://127.0.0.1:5000/predict \
     -H "Content-Type: application/json" \
     -d @sample_payload.json


