from flask import Flask, jsonify, render_template
import pandas as pd
from data_cleaning import clean_data
import os

app = Flask(__name__)

# ✅ Run data cleaning and get cleaned file path
cleaned_file = clean_data()

# API Route to get cleaned data
@app.route("/api/data", methods=["GET"])
def get_data():
    if not os.path.exists(cleaned_file):
        return jsonify({"error": "Cleaned data not found"}), 404

    data = pd.read_csv(cleaned_file)
    return jsonify(data.to_dict(orient="records"))

# Frontend Dashboard Route
@app.route("/")
def dashboard():
    return render_template("dashboard.html")

if __name__ == "__main__":
    app.run(debug=True)