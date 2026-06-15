import pandas as pd
import pickle

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

# Load Dataset
data = pd.read_csv(r"C:\Users\Dell\Downloads\archive (1)\Churn_Modelling.csv")

# Display dataset
print("Dataset Preview:")
print(data.head())

# Check missing values
print("\nMissing Values:")
print(data.isnull().sum())

# Remove missing values if any
data.dropna(inplace=True)

# Remove unnecessary columns
data = data.drop(["RowNumber", "CustomerId", "Surname"], axis=1)

# Encode categorical columns
label_encoder = LabelEncoder()

for column in data.columns:
    if data[column].dtype == 'object':
        data[column] = label_encoder.fit_transform(data[column])

# Features and Target
X = data.drop("Exited", axis=1)
y = data["Exited"]

# Split dataset
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42
)

# Train Random Forest Model
model = RandomForestClassifier(
    n_estimators=100,
    random_state=42
)

model.fit(X_train, y_train)

# Predictions
y_pred = model.predict(X_test)

# Accuracy
accuracy = accuracy_score(y_test, y_pred)

print("\nAccuracy:")
print(f"{accuracy * 100:.2f}%")

# Confusion Matrix
print("\nConfusion Matrix:")
print(confusion_matrix(y_test, y_pred))

# Classification Report
print("\nClassification Report:")
print(classification_report(y_test, y_pred))

# Feature Importance
print("\nFeature Importance:")
for feature, importance in zip(X.columns, model.feature_importances_):
    print(f"{feature}: {importance:.4f}")

# Save Model
pickle.dump(model, open("churn_model.pkl", "wb"))
print("\nModel saved successfully as churn_model.pkl")

# Load Model
loaded_model = pickle.load(open("churn_model.pkl", "rb"))
print("Model loaded successfully!")

# Sample Predictions
sample_predictions = loaded_model.predict(X_test.iloc[:10])

print("\nSample Predictions:")
print(sample_predictions)

# Actual Values
print("\nActual Values:")
print(y_test.iloc[:10].values)