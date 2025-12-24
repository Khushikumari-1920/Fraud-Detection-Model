import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
import warnings
import pickle

warnings.filterwarnings('ignore')
sns.set(style='whitegrid')

# Load dataset
df = pd.read_csv('FraudDetectionDataset.csv')

# Debug: print column names once (optional)
print("Dataset columns:", df.columns.tolist())

# Feature Engineering
df['balanceDiffOrig'] = df['oldbalanceOrg'] - df['newbalanceOrig']
df['balanceDiffDest'] = df['newbalanceDest'] - df['oldbalanceDest']

# Drop unused columns
df_model = df.drop(columns=['nameOrig', 'nameDest', 'step'])

categorical = ['type']
numerical = [
    'amount', 'oldbalanceOrg', 'newbalanceOrig',
    'oldbalanceDest', 'newbalanceDest',
    'balanceDiffOrig', 'balanceDiffDest'
]

X = df_model.drop(columns=['isFraud', 'isFlaggedFraud'])
y = df_model['isFraud']

# Train/Test Split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.3, stratify=y, random_state=42
)

# Preprocessing
preprocessor = ColumnTransformer(
    transformers=[
        ('num', StandardScaler(), numerical),
        ('cat', OneHotEncoder(drop='first'), categorical)
    ]
)

# Build Pipeline
pipeline = Pipeline([
    ('prep', preprocessor),
    ('clf', LogisticRegression(max_iter=2000, class_weight='balanced'))
])

# Train
pipeline.fit(X_train, y_train)

# Predictions
y_pred = pipeline.predict(X_test)

print("Classification Report:")
print(classification_report(y_test, y_pred))

print("Confusion Matrix:")
print(confusion_matrix(y_test, y_pred))

print("Accuracy:", accuracy_score(y_test, y_pred))

# ✅ Save the trained pipeline/model
with open("fraud_detection_pipeline.pkl", "wb") as f:
    pickle.dump(pipeline, f)

print("✅ Model saved as fraud_detection_pipeline.pkl")
