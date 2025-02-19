import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.ensemble import IsolationForest
from sklearn.metrics import confusion_matrix, roc_auc_score

# Ensure 'static/' folder exists for saving images
os.makedirs("static", exist_ok=True)

# Load cleaned data
file_path = "data/cleaned_data.csv"

try:
    data = pd.read_csv(file_path)
except FileNotFoundError:
    print(f"❌ ERROR: File '{file_path}' not found. Run data_cleaning.py first.")
    exit()
except pd.errors.EmptyDataError:
    print("❌ ERROR: Cleaned data file is empty. Check the dataset.")
    exit()

# ✅ 1. Purchase Amount Distribution
plt.figure(figsize=(10, 6))
sns.histplot(data['Purchase Amount (USD)'], kde=True, bins=30, color='blue')
plt.title('Distribution of Purchase Amounts')
plt.xlabel('Purchase Amount (USD)')
plt.ylabel('Frequency')
plt.savefig("static/purchase_distribution.png")
plt.close()

# ✅ 2. Review Ratings Distribution
plt.figure(figsize=(10, 6))
sns.countplot(x='Review Rating', data=data, palette='viridis')
plt.title('Distribution of Review Ratings')
plt.xlabel('Review Rating')
plt.ylabel('Count')
plt.savefig("static/review_ratings.png")
plt.close()

# ✅ 3. Correlation Heatmap
plt.figure(figsize=(12, 8))
sns.heatmap(data.corr(), annot=True, cmap='coolwarm')
plt.title('Feature Correlation Heatmap')
plt.savefig("static/correlation_heatmap.png")
plt.close()

# ✅ 4. Anomaly Detection (Isolation Forest)
iso = IsolationForest(contamination=0.01, random_state=42)
data['Anomaly'] = iso.fit_predict(data.select_dtypes(include=['float64', 'int64']))

plt.figure(figsize=(10, 6))
sns.scatterplot(x=data.index, y=data['Purchase Amount (USD)'], hue=data['Anomaly'], palette={1: "blue", -1: "red"})
plt.title("Anomaly Detection in Purchase Amounts")
plt.xlabel("Transaction Index")
plt.ylabel("Purchase Amount (USD)")
plt.legend(title="Anomaly", labels=["Normal", "Anomaly"])
plt.savefig("static/anomaly_detection.png")
plt.close()

# Dummy test data for model evaluation
y_test = np.array([1, 0, 1, 1, 0, 1, 0, 0, 1, 0])  # Example true labels
y_pred = np.array([1, 0, 1, 1, 0, 0, 0, 1, 1, 0])  # Example predictions
y_scores = np.array([0.95, 0.1, 0.85, 0.9, 0.2, 0.7, 0.3, 0.1, 0.8, 0.4])  # Example predicted probabilities

# ✅ 5. Confusion Matrix
conf_matrix = confusion_matrix(y_test, y_pred)
plt.figure(figsize=(6, 5))
sns.heatmap(conf_matrix, annot=True, fmt='d', cmap='Blues')
plt.title("Confusion Matrix")
plt.xlabel("Predicted Label")
plt.ylabel("True Label")
plt.savefig("static/confusion_matrix.png")
plt.close()

# ✅ 6. ROC AUC Score
roc_auc = roc_auc_score(y_test, y_scores)
with open("static/roc_auc_score.txt", "w") as f:
    f.write(f"ROC AUC Score: {roc_auc:.2f}\n")

print("✅ All visualizations generated and saved in 'static/' folder.")
