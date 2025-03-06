import pickle
import pandas as pd
from xgboost import XGBClassifier
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder

# Load dataset
df = pd.read_csv("data.csv")  # Ensure this file has all expected features

# Define all feature columns
feature_columns = ["Age", "Gender", "Education Level", "Employment Status", "Location", "Drug Type", "Frequency",
                   "Duration (Years)", "Mental Health", "Physical Health", "Monthly Income (Ksh)",
                   "Family Background", "Peer Influence", "Rehab Attendance", "Counseling Sessions"]

# Ensure all categorical features are of type 'category' before encoding
categorical_columns = ["Gender", "Education Level", "Employment Status", "Location", "Drug Type",
                       "Frequency", "Mental Health", "Physical Health", "Family Background",
                       "Peer Influence", "Rehab Attendance"]

for col in categorical_columns:
    df[col] = df[col].astype("category")

# Encode categorical variables
label_encoders = {}
df_encoded = df.copy()

for col in categorical_columns:
    le = LabelEncoder()
    df_encoded[col] = le.fit_transform(df[col])
    label_encoders[col] = le  # Store encoders for later use

# Encode target labels
severity_encoder = LabelEncoder()
df_encoded["Severity"] = severity_encoder.fit_transform(df["Severity"])

relapse_encoder = LabelEncoder()
df_encoded["Relapse Case"] = relapse_encoder.fit_transform(df["Relapse Case"])

# Save encoders
with open("label_encoders.pkl", "wb") as file:
    pickle.dump(label_encoders, file)

with open("severity_encoder.pkl", "wb") as file:
    pickle.dump(severity_encoder, file)

with open("relapse_encoder.pkl", "wb") as file:
    pickle.dump(relapse_encoder, file)

# Ensure all data types are numeric
df_encoded = df_encoded.apply(pd.to_numeric)

# Define features and targets
X = df_encoded[feature_columns]
y_severity = df_encoded["Severity"]
y_relapse = df_encoded["Relapse Case"]

# Split data
X_train_sev, X_test_sev, y_train_sev, y_test_sev = train_test_split(
    X, y_severity, test_size=0.2, random_state=42)
X_train_rel, X_test_rel, y_train_rel, y_test_rel = train_test_split(
    X, y_relapse, test_size=0.2, random_state=42)

# Train models
xgb_severity = XGBClassifier(use_label_encoder=False, eval_metric="mlogloss")
xgb_severity.fit(X_train_sev, y_train_sev)

xgb_relapse = XGBClassifier(use_label_encoder=False, eval_metric="logloss")
xgb_relapse.fit(X_train_rel, y_train_rel)

# Save models
with open("xgb_severity.pkl", "wb") as file:
    pickle.dump(xgb_severity, file)

with open("xgb_relapse.pkl", "wb") as file:
    pickle.dump(xgb_relapse, file)

print("✅ Models trained and saved with correct feature set!")
