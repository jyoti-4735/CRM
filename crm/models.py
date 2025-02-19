import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score

def train_model(file_path):
    data = pd.read_csv(file_path)

    # Assuming 'target' is the column to predict
    X = data.drop(columns=['target'])
    y = data['target']

    # Split into training and test sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Decision Tree Model
    tree = DecisionTreeClassifier(max_depth=3)
    tree.fit(X_train, y_train)
    tree_acc = accuracy_score(y_test, tree.predict(X_test))

    # Random Forest Model
    forest = RandomForestClassifier(n_estimators=50, max_depth=5, random_state=42)
    forest.fit(X_train, y_train)
    forest_acc = accuracy_score(y_test, forest.predict(X_test))

    print(f"Decision Tree Accuracy: {tree_acc:.2f}")
    print(f"Random Forest Accuracy: {forest_acc:.2f}")

    return forest  # Return the trained model
