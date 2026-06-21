
# IMPORT LIBRARIES
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, roc_auc_score, confusion_matrix, classification_report
# LOAD DATASET
import os
BASE_DIR = os.path.dirname(__file__)
file_path = os.path.join(BASE_DIR, "heart.csv")

df = pd.read_csv(file_path)

print(df.head())
print(df.info())
print(df.isnull().sum())
# Drop duplicates 
df = df.drop_duplicates()
df = df.dropna()
# Target distribution
sns.countplot(x='target', data=df)
plt.title("Heart Disease Distribution")
plt.show()
# Correlation heatmap
plt.figure(figsize=(10,6))
sns.heatmap(df.corr(), cmap="coolwarm", annot=False)
plt.title("Feature Correlation Heatmap")
plt.show()
# Age distribution
sns.histplot(df['age'], bins=20, kde=True)
plt.title("Age Distribution")
plt.show()
# 5. SPLIT FEATURES & TARGET
X = df.drop('target', axis=1)
y = df['target']
# 6. TRAIN-TEST SPLIT
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)
# 7. FEATURE SCALING
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)
# 8. MODEL TRAINING
model = LogisticRegression()
model.fit(X_train, y_train)
# 9. PREDICTIONS
y_pred = model.predict(X_test)
y_prob = model.predict_proba(X_test)[:, 1]
# 10. EVALUATION
# Accuracy
acc = accuracy_score(y_test, y_pred)
print("Accuracy:", acc)
# ROC-AUC Score
roc = roc_auc_score(y_test, y_prob)
print("ROC-AUC Score:", roc)
# Confusion Matrix
cm = confusion_matrix(y_test, y_pred)
print("Confusion Matrix:\n", cm)
# Classification Report
print("\nClassification Report:\n", classification_report(y_test, y_pred))

sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
plt.title("Confusion Matrix")
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.show()
