import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split, GridSearchCV, StratifiedKFold
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (confusion_matrix, classification_report,
                             accuracy_score, precision_score, recall_score, f1_score)
import warnings
warnings.filterwarnings("ignore")

# Load dataset directly from GitHub
url = "https://raw.githubusercontent.com/IBM/telco-customer-churn-on-icp4d/master/data/Telco-Customer-Churn.csv"
df = pd.read_csv(url)

# numeric conversion
df['TotalCharges'] = pd.to_numeric(df['TotalCharges'], errors='coerce')
df.dropna(inplace=True)

# Encode binary categorical columns
df['Churn'] = df['Churn'].map({'Yes': 1, 'No': 0})
df['gender'] = df['gender'].map({'Female': 1, 'Male': 0})
# Identify remaining categorical columns
categorical_cols = df.select_dtypes(include='object').columns.tolist()
# Exclude 'customerID' as it's not a feature
categorical_cols.remove('customerID')


# One-hot encode categorical columns
df = pd.get_dummies(df, columns=categorical_cols, drop_first=True) # Added drop_first=True to avoid multicollinearity


# Select features
X = df.drop(['customerID', 'Churn'], axis=1)
y = df['Churn']

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# Cross-validation strategy
cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

# ------------------------------
# Decision Tree with GridSearch
# ------------------------------
dt_param_grid = {
    'max_depth': [3, 5, 10, None],
    'min_samples_split': [2, 5, 10]
}

dt_grid = GridSearchCV(DecisionTreeClassifier(random_state=42),
                       dt_param_grid, cv=cv, scoring='accuracy', n_jobs=-1)
dt_grid.fit(X_train, y_train)
dt_best = dt_grid.best_estimator_
dt_preds = dt_best.predict(X_test)

# ------------------------------
# Random Forest with GridSearch
# ------------------------------
rf_param_grid = {
    'n_estimators': [50, 100],
    'max_depth': [5, 10, None],
    'min_samples_split': [2, 5]
}

rf_grid = GridSearchCV(RandomForestClassifier(random_state=42),
                       rf_param_grid, cv=cv, scoring='accuracy', n_jobs=-1)
rf_grid.fit(X_train, y_train)
rf_best = rf_grid.best_estimator_
rf_preds = rf_best.predict(X_test)


# ------------------------------
# Metrics function
# ------------------------------
def print_metrics(model_name, y_true, y_pred):
    print(f"\n=== {model_name} Metrics ===")
    print("Accuracy:", accuracy_score(y_true, y_pred))
    print("Precision:", precision_score(y_true, y_pred))
    print("Recall:", recall_score(y_true, y_pred))
    print("F1-score:", f1_score(y_true, y_pred))
    print("Classification Report:\n", classification_report(y_true, y_pred))

# ------------------------------
# Results
# ------------------------------
print("Best Decision Tree Parameters:", dt_grid.best_params_)
print("Best Random Forest Parameters:", rf_grid.best_params_)

# Decision Tree Results
dt_cm = confusion_matrix(y_test, dt_preds)
print_metrics("Decision Tree", y_test, dt_preds)

# Random Forest Results
rf_cm = confusion_matrix(y_test, rf_preds)
print_metrics("Random Forest", y_test, rf_preds)

# ------------------------------
# Visualize Confusion Matrices
# ------------------------------
plt.figure(figsize=(12, 5))

plt.subplot(1, 2, 1)
sns.heatmap(dt_cm, annot=True, fmt='d', cmap='Blues')
plt.title('Decision Tree Confusion Matrix')
plt.xlabel('Predicted')
plt.ylabel('Actual')

plt.subplot(1, 2, 2)
sns.heatmap(rf_cm, annot=True, fmt='d', cmap='Greens')
plt.title('Random Forest Confusion Matrix')
plt.xlabel('Predicted')
plt.ylabel('Actual')

plt.tight_layout()
plt.savefig('confusion_matrices.png')
plt.show()
