import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report
import joblib

# Load dataset
df = pd.read_csv("final_balanced_dataset.csv")

# Define features and target
X = df.drop("target_hit", axis=1)
y = df["target_hit"]

# Preprocessing
categorical_features = ["shooter_experience", "wind_direction", "time_of_day", "weapon_type"]
boolean_features = ["is_target_moving"]
numeric_features = ["distance_to_target", "wind_speed", "target_speed"]

preprocessor = ColumnTransformer([
    ("cat", OneHotEncoder(), categorical_features),
    ("passthrough", "passthrough", boolean_features + numeric_features)
])

# Random Forest pipeline
pipeline = Pipeline([
    ("preprocessor", preprocessor),
    ("classifier", RandomForestClassifier(n_estimators=100, random_state=42))
])

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
pipeline.fit(X_train, y_train)

# Save the model
joblib.dump(pipeline, "RandomForest_model.pkl")

# Evaluate
y_pred = pipeline.predict(X_test)
print(classification_report(y_test, y_pred))
