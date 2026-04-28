# ===============================
# Credit Card Fraud Detection
# Correct Industry Version
# ===============================

import pandas as pd
import numpy as np

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix

from imblearn.over_sampling import SMOTE
import joblib

# ===============================
# 1. LOAD DATA
# ===============================
print("📥 Loading data...")

df = pd.read_csv('data/creditcard.csv')

# Optional: use sample for speed
df = df.sample(n=20000, random_state=42)

print("✅ Data loaded:", df.shape)


# ===============================
# 2. SPLIT FEATURES & TARGET
# ===============================
print("🔀 Splitting features and target...")

X = df.drop('Class', axis=1)
y = df['Class']


# ===============================
# 3. SCALING
# ===============================
print("⚙️ Scaling data...")

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)


# ===============================
# 4. TRAIN-TEST SPLIT (IMPORTANT)
# ===============================
print("📊 Splitting train and test...")

X_train, X_test, y_train, y_test = train_test_split(
    X_scaled, y, test_size=0.2, random_state=42, stratify=y
)


# ===============================
# 5. APPLY SMOTE (ONLY TRAIN DATA)
# ===============================
print("⚖️ Applying SMOTE on training data...")

smote = SMOTE(random_state=42)
X_train, y_train = smote.fit_resample(X_train, y_train)

print("✅ After SMOTE:", X_train.shape)


# ===============================
# 6. MODEL TRAINING
# ===============================
print("🤖 Training model...")

model = RandomForestClassifier(n_estimators=50, random_state=42)
model.fit(X_train, y_train)

print("✅ Model trained")


# ===============================
# 7. PREDICTION
# ===============================
print("🔮 Making predictions...")

y_pred = model.predict(X_test)


# ===============================
# 8. EVALUATION
# ===============================
print("\n📈 Evaluation Results:\n")

print("Classification Report:")
print(classification_report(y_test, y_pred))

print("Confusion Matrix:")
print(confusion_matrix(y_test, y_pred))


# ===============================
# 9. SAVE MODEL
# ===============================
print("💾 Saving model...")

joblib.dump(model, 'models/fraud_model.pkl')

print("✅ Model saved at models/fraud_model.pkl")


# ===============================
# 10. SAMPLE PREDICTION
# ===============================
print("\n🧪 Testing on one sample...")

sample = X_test[0].reshape(1, -1)
prediction = model.predict(sample)

if prediction[0] == 1:
    print("⚠️ FRAUD DETECTED!")
else:
    print("✅ NORMAL TRANSACTION")

print("\n🎉 DONE!")