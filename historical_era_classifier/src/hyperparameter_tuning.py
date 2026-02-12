import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix

from imblearn.pipeline import Pipeline
from imblearn.over_sampling import SMOTE

from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parent.parent
FEATURES_DIR = (PROJECT_ROOT / "data" / "processed" / "features.csv" ).as_posix()

df = pd.read_csv(FEATURES_DIR)
X = df.drop(columns=["label"])
y = df["label"]

X_train, X_test, y_train, y_test = train_test_split(
    X, y,
    test_size=0.2,
    random_state=42,
    stratify=y
)

#Pipelining to include smote
pipeline = Pipeline(steps=[
    ("smote", SMOTE(random_state=42)),
    ("rf", RandomForestClassifier(random_state=42))
])

#Hyperparam grid
param_grid = {
    "smote__k_neighbors": [3, 5],

    "rf__n_estimators": [300, 500],
    "rf__max_depth": [None, 15, 25],
    "rf__min_samples_split": [2, 5],
    "rf__min_samples_leaf": [1, 2],
    "rf__max_features": ["sqrt", "log2"]
}

grid = GridSearchCV(
    estimator=pipeline,
    param_grid=param_grid,
    scoring="f1_macro", 
    cv=5,
    n_jobs=-1,
    verbose=2
)

grid.fit(X_train, y_train)

print("--- Best Parameters ---")
print(grid.best_params_)

best_model = grid.best_estimator_
y_pred = best_model.predict(X_test)

print("--- Test Set Performance ---")
print(classification_report(y_test, y_pred))
