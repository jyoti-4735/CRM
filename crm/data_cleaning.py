import pandas as pd
import numpy as np
from sklearn.impute import KNNImputer
from sklearn.ensemble import IsolationForest
from sklearn.preprocessing import LabelEncoder
import os

def clean_data(file_path="data/raw/shopping_trends.csv"):
    try:
        data = pd.read_csv(file_path)
    except FileNotFoundError:
        print(f"❌ ERROR: The dataset '{file_path}' does not exist.")
        return
    except pd.errors.EmptyDataError:
        print("❌ ERROR: The dataset is empty.")
        return

    # Drop empty columns
    data = data.dropna(axis=1, how='all')

    if data.empty:
        print("❌ ERROR: No valid data found after cleaning.")
        return

    # Identify categorical columns
    categorical_cols = data.select_dtypes(include=['object']).columns

    # Encode categorical columns
    encoders = {}
    for col in categorical_cols:
        le = LabelEncoder()
        data[col] = le.fit_transform(data[col].astype(str))
        encoders[col] = le

    # Handle missing values with KNN Imputer
    imputer = KNNImputer(n_neighbors=5)
    data.iloc[:, :] = imputer.fit_transform(data)

    # Remove duplicates
    data = data.drop_duplicates()

    # Outlier detection using Isolation Forest
    iso = IsolationForest(contamination=0.01, random_state=42)
    data["Anomaly"] = iso.fit_predict(data)
    data = data[data["Anomaly"] == 1].drop(columns=["Anomaly"])

    # Ensure the 'data/' folder exists before saving
    os.makedirs("data", exist_ok=True)

    # Save cleaned dataset
    cleaned_file_path = "data/cleaned_data.csv"
    data.to_csv(cleaned_file_path, index=False)
    print(f"✅ Data cleaning complete! Cleaned data saved to {cleaned_file_path}")

    return cleaned_file_path  # Return the file path

