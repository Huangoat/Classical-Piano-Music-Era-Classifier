import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import joblib

from sklearn.model_selection import train_test_split, StratifiedKFold, cross_val_score
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score

from imblearn.pipeline import Pipeline
from imblearn.over_sampling import SMOTE


PROJECT_ROOT = Path(__file__).resolve().parent.parent
FEATURES_DIR = (PROJECT_ROOT / "data" / "processed" / "features.csv").as_posix()
MODEL_DIR = (PROJECT_ROOT / "models" / "model.pkl")

# Loading model
df = pd.read_csv(FEATURES_DIR)
X = df.drop(columns=["label"])
y = df["label"]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

pipeline = Pipeline([
    ("smote", SMOTE(random_state=42, k_neighbors=3)),
    ("rf", RandomForestClassifier(
        n_estimators=500,
        max_depth=15,
        random_state=42,
        max_features="sqrt",
        min_samples_leaf=1,
        min_samples_split=5
    ))
])

#K-Fold cross validation

cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

cv_scores = cross_val_score(
    pipeline,
    X_train,
    y_train,
    cv=cv,
    scoring="accuracy",
    n_jobs=-1
)

print("--- Cross Validation Results ---")
print(f"Fold Accuracies: {cv_scores}")
print(f"Mean Accuracy: {cv_scores.mean():.4f}")
print(f"Std Dev: {cv_scores.std():.4f}")

pipeline.fit(X_train, y_train)

y_pred = pipeline.predict(X_test)

# Evaluation metrics
print("--- Model Performance ---")
print(f"Accuracy: {accuracy_score(y_test, y_pred):.4f}")
print("Classification Report:")
print(classification_report(y_test, y_pred))

# Confusion matrix
plt.figure(figsize=(10, 8))
cm = confusion_matrix(y_test, y_pred)
sns.heatmap(
    cm,
    annot=True,
    fmt="d",
    cmap="Blues",
    xticklabels=pipeline.named_steps["rf"].classes_,
    yticklabels=pipeline.named_steps["rf"].classes_
)

plt.title("Confusion Matrix: Random Forest + SMOTE")
plt.ylabel("Actual Era")
plt.xlabel("Predicted Era")
plt.show()


# Feature importance analysis
rf_model = pipeline.named_steps["rf"]

feature_importances = pd.DataFrame({
    "feature": X.columns,
    "importance": rf_model.feature_importances_
}).sort_values(by="importance", ascending=False)

print("--- Top 10 Most Important Features ---")
print(feature_importances.head(10))

plt.figure(figsize=(12, 6))
sns.barplot(x="importance", y="feature", data=feature_importances.head(15))
plt.title("Top 15 Features for Predicting Musical Era")
plt.tight_layout()
plt.show()

joblib.dump(pipeline, MODEL_DIR)
