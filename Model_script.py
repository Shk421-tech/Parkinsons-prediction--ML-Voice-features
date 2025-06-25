import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
from sklearn.model_selection import RandomizedSearchCV, train_test_split
from sklearn.preprocessing import StandardScaler

# Load dataset
df = pd.read_csv(r"C:\Users\baboo\Downloads\MINOR-500\parkinsons.csv")
df.drop(columns=["name"], inplace=True)

# Selected top features
selected_features = [
    'PPE',
    'MDVP:Fo(Hz)',
    'spread1',
    'MDVP:Flo(Hz)',
    'Jitter:DDP',
    'MDVP:Fhi(Hz)',
    'spread2',
    'NHR'
]
X = df[selected_features]
y = df["status"]

# Scaling
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Train/test split (with stratification to balance classes)
X_train, X_test, y_train, y_test = train_test_split(
    X_scaled, y, test_size=0.2, random_state=42, stratify=y)

# Model and tuning
param_grid = {
    'n_estimators': [100, 200, 300],
    'max_depth': [10, 20, None],
    'min_samples_split': [2, 5, 10],
    'min_samples_leaf': [1, 2],
    'max_features': ['sqrt']
}
rf = RandomForestClassifier(random_state=42)
random_search = RandomizedSearchCV(
    rf, param_grid, n_iter=10, cv=5, scoring='accuracy', n_jobs=-1, random_state=42
)
random_search.fit(X_train, y_train)
best_rf = random_search.best_estimator_
best_rf.fit(X_train, y_train)

# Evaluation
y_pred = best_rf.predict(X_test)
print("Accuracy:", accuracy_score(y_test, y_pred))
print("Best Params:", random_search.best_params_)

# Save model and scaler
joblib.dump(best_rf, "parkinsons_model.pkl")
joblib.dump(scaler, "parkinsons_scaler.pkl")
print("Model and scaler saved!")
