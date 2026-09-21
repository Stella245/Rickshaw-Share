import pandas as pd
import joblib

from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix
)


# Load our dataset
df = pd.read_csv("ai/dataset/rickshaw_matching_dataset.csv")

print("Dataset loaded successfully!")
print("Number of records:", len(df))


# Features used by the ML model
features = [
    "pickup_distance_km",
    "time_difference_min",
    "route_overlap_percent",
    "detour_km",
    "destination_compatibility",
    "preference_compatible"
]


# X = information the model uses to make a prediction
X = df[features]

# y = answer the model is trying to learn
y = df["match"]


print("\nFeatures:")
print(X.head())

print("\nTarget:")
print(y.head())


# Split the dataset into training and testing data
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.20,
    random_state=42,
    stratify=y
)


print("\nTraining records:", len(X_train))
print("Testing records:", len(X_test))


# Create the Decision Tree model
model = DecisionTreeClassifier(
    max_depth=5,
    random_state=42
)


# Train the model
model.fit(X_train, y_train)

print("\nModel training completed!")


# Save the trained model
joblib.dump(model, "AI/model.pkl")

print("Trained model saved as AI/model.pkl")


# Make predictions on the test data
y_pred = model.predict(X_test)


print("\nFirst 20 predictions:")
print(y_pred[:20])

print("\nActual values:")
print(y_test.iloc[:20].values)


# Calculate evaluation metrics
accuracy = accuracy_score(y_test, y_pred)
precision = precision_score(y_test, y_pred)
recall = recall_score(y_test, y_pred)
f1 = f1_score(y_test, y_pred)


print("\nModel Evaluation:")
print("Accuracy :", round(accuracy, 4))
print("Precision:", round(precision, 4))
print("Recall   :", round(recall, 4))
print("F1 Score :", round(f1, 4))


# Confusion matrix
cm = confusion_matrix(y_test, y_pred)

print("\nConfusion Matrix:")
print(cm)