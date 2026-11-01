# %% [markdown]
# # Student Performance Prediction System
# Workflow: Data -> Cleaning -> Analysis -> Feature Selection -> Training -> Testing -> Evaluation -> Prediction
#
# Dataset: "Student Performance Factors" (Kaggle). Save the CSV as data/StudentPerformanceFactors.csv

# %% 1. Imports and settings
import json
import warnings
from pathlib import Path

import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import GradientBoostingRegressor, RandomForestRegressor
from sklearn.impute import SimpleImputer
from sklearn.inspection import permutation_importance
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import GridSearchCV, cross_val_score, train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

warnings.filterwarnings("ignore")
sns.set_theme(style="whitegrid")
RANDOM_STATE = 42
DATA_PATH = Path("data/StudentPerformanceFactors.csv")
VISUALS = Path("visuals"); VISUALS.mkdir(exist_ok=True)
MODELS = Path("models"); MODELS.mkdir(exist_ok=True)
TARGET = "Exam_Score"
AT_RISK_THRESHOLD = 65  # predicted score below this = "needs support" (our own choice, change if needed)

# %% 2. Load and understand the data
df = pd.read_csv(DATA_PATH)
print("Shape:", df.shape)
print(df.head())
print("\nData types / non-null counts:")
df.info()
print("\nNumeric summary:")
print(df.describe())
print("\nMissing values per column:")
print(df.isnull().sum()[df.isnull().sum() > 0])

# %% 3. Data cleaning
# 3a. Duplicates
before = len(df)
df = df.drop_duplicates().reset_index(drop=True)
print(f"Removed {before - len(df)} duplicate rows")

# 3b. Strip stray whitespace in text columns
cat_cols_all = df.select_dtypes(include="object").columns
for c in cat_cols_all:
    df[c] = df[c].str.strip()

# 3c. Impossible values: exam score must be 0-100 (this dataset has a few above 100)
bad = ((df[TARGET] > 100) | (df[TARGET] < 0)).sum()
df = df[(df[TARGET] >= 0) & (df[TARGET] <= 100)].reset_index(drop=True)
print(f"Removed {bad} rows with impossible {TARGET}")

# 3d. Attendance must be 0-100 too
if "Attendance" in df.columns:
    df = df[df["Attendance"].between(0, 100)].reset_index(drop=True)

# 3e. Missing values: handled inside the model Pipeline later (median for numbers,
# most-frequent for categories) so the SAME rule applies to new data at prediction time
# and nothing leaks from the test set. We just report them here.
print("Missing values after cleaning:", int(df.isnull().sum().sum()))

# %% 4. Exploratory Data Analysis (EDA)
num_cols = df.select_dtypes(include=np.number).columns.drop(TARGET).tolist()
cat_cols = df.select_dtypes(include="object").columns.tolist()

# 4a. Target distribution
plt.figure(figsize=(7, 4))
sns.histplot(df[TARGET], bins=25, kde=True)
plt.title("Distribution of Exam Score"); plt.tight_layout()
plt.savefig(VISUALS / "01_score_distribution.png", dpi=150); plt.close()

# 4b. Correlation heatmap (numeric features)
plt.figure(figsize=(9, 7))
sns.heatmap(df[num_cols + [TARGET]].corr(), annot=True, fmt=".2f", cmap="coolwarm", center=0)
plt.title("Correlation Heatmap"); plt.tight_layout()
plt.savefig(VISUALS / "02_correlation_heatmap.png", dpi=150); plt.close()

# 4c. Key relationships with the score
pairs = [c for c in ["Hours_Studied", "Attendance", "Sleep_Hours", "Previous_Scores"] if c in df.columns]
fig, axes = plt.subplots(1, len(pairs), figsize=(5 * len(pairs), 4))
axes = np.atleast_1d(axes)
for ax, c in zip(axes, pairs):
    sns.regplot(data=df, x=c, y=TARGET, ax=ax, scatter_kws={"alpha": 0.25, "s": 12}, line_kws={"color": "red"})
    ax.set_title(f"{c} vs {TARGET}")
plt.tight_layout(); plt.savefig(VISUALS / "03_numeric_vs_score.png", dpi=150); plt.close()

# 4d. Categorical features vs average score
show_cats = [c for c in ["Parental_Involvement", "Motivation_Level", "Access_to_Resources", "Teacher_Quality"] if c in df.columns]
if show_cats:
    fig, axes = plt.subplots(1, len(show_cats), figsize=(5 * len(show_cats), 4))
    axes = np.atleast_1d(axes)
    for ax, c in zip(axes, show_cats):
        sns.boxplot(data=df, x=c, y=TARGET, ax=ax)
        ax.set_title(f"{c} vs {TARGET}"); ax.tick_params(axis="x", rotation=20)
    plt.tight_layout(); plt.savefig(VISUALS / "04_categorical_vs_score.png", dpi=150); plt.close()

# 4e. Top correlations, printed so you can write insights in your report
print("\nCorrelation with target:")
print(df[num_cols + [TARGET]].corr()[TARGET].drop(TARGET).sort_values(ascending=False))

# %% 5. Feature selection
# Step 1: keep all columns as candidates; Step 2: rank them with permutation importance
# after training (section 7). Low-importance features are then dropped and the model retrained.
X = df.drop(columns=[TARGET])
y = df[TARGET]

# %% 6. Train / test split + preprocessing pipeline
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=RANDOM_STATE)
print("Train:", X_train.shape, "Test:", X_test.shape)


def build_preprocessor(X_ref):
    nums = X_ref.select_dtypes(include=np.number).columns.tolist()
    cats = X_ref.select_dtypes(include="object").columns.tolist()
    return ColumnTransformer([
        ("num", Pipeline([("imp", SimpleImputer(strategy="median")), ("sc", StandardScaler())]), nums),
        ("cat", Pipeline([("imp", SimpleImputer(strategy="most_frequent")),
                          ("oh", OneHotEncoder(handle_unknown="ignore"))]), cats),
    ])


def evaluate(name, pipe, Xtr, ytr, Xte, yte):
    pipe.fit(Xtr, ytr)
    pred = pipe.predict(Xte)
    cv = cross_val_score(pipe, Xtr, ytr, cv=5, scoring="r2")
    res = {
        "Model": name,
        "CV R2 (mean)": round(cv.mean(), 4),
        "Test R2": round(r2_score(yte, pred), 4),
        "Test MAE": round(mean_absolute_error(yte, pred), 3),
        "Test RMSE": round(np.sqrt(mean_squared_error(yte, pred)), 3),
    }
    return res, pred

# %% 7. Train and compare models
models = {
    "Linear Regression": LinearRegression(),
    "Random Forest": RandomForestRegressor(n_estimators=200, random_state=RANDOM_STATE, n_jobs=-1),
    "Gradient Boosting": GradientBoostingRegressor(random_state=RANDOM_STATE),
}
results, preds = [], {}
for name, model in models.items():
    pipe = Pipeline([("prep", build_preprocessor(X_train)), ("model", model)])
    r, p = evaluate(name, pipe, X_train, y_train, X_test, y_test)
    results.append(r); preds[name] = p
results_df = pd.DataFrame(results).sort_values("Test R2", ascending=False)
print(results_df.to_string(index=False))
results_df.to_csv("models/model_results.csv", index=False)

# %% 8. Improve the best model (light hyperparameter tuning)
# Tune the best of the tree models (or Linear Regression if it wins, in which case skip tuning).
best_name = results_df.iloc[0]["Model"]
print("Best baseline:", best_name)
if best_name == "Linear Regression":
    final_pipe = Pipeline([("prep", build_preprocessor(X_train)), ("model", LinearRegression())])
    final_pipe.fit(X_train, y_train)
else:
    base = models[best_name]
    grid = ({"model__n_estimators": [200, 400], "model__max_depth": [None, 10, 20], "model__min_samples_leaf": [1, 3]}
            if best_name == "Random Forest" else
            {"model__n_estimators": [100, 200], "model__learning_rate": [0.05, 0.1], "model__max_depth": [2, 3]})
    search = GridSearchCV(Pipeline([("prep", build_preprocessor(X_train)), ("model", base)]),
                          grid, cv=3, scoring="r2", n_jobs=-1)
    search.fit(X_train, y_train)
    print("Best params:", search.best_params_)
    final_pipe = search.best_estimator_

# %% 9. Final evaluation on the untouched test set
final_pred = final_pipe.predict(X_test)
print(f"Final Test R2  : {r2_score(y_test, final_pred):.4f}")
print(f"Final Test MAE : {mean_absolute_error(y_test, final_pred):.3f}")
print(f"Final Test RMSE: {np.sqrt(mean_squared_error(y_test, final_pred)):.3f}")

fig, ax = plt.subplots(1, 2, figsize=(12, 4.5))
ax[0].scatter(y_test, final_pred, alpha=0.3, s=12)
lims = [y_test.min(), y_test.max()]
ax[0].plot(lims, lims, "r--"); ax[0].set_xlabel("Actual"); ax[0].set_ylabel("Predicted")
ax[0].set_title("Actual vs Predicted")
sns.histplot(y_test - final_pred, bins=30, kde=True, ax=ax[1]); ax[1].set_title("Residuals")
plt.tight_layout(); plt.savefig(VISUALS / "05_actual_vs_predicted.png", dpi=150); plt.close()

# At-risk view: how well do we catch students who actually scored below the threshold?
actual_risk = y_test < AT_RISK_THRESHOLD
pred_risk = pd.Series(final_pred, index=y_test.index) < AT_RISK_THRESHOLD
tp = int((actual_risk & pred_risk).sum()); fn = int((actual_risk & ~pred_risk).sum()); fp = int((~actual_risk & pred_risk).sum())
print(f"\nAt-risk (<{AT_RISK_THRESHOLD}): actual={int(actual_risk.sum())}, caught={tp}, missed={fn}, false alarms={fp}")

# %% 10. Feature importance (which factors matter most)
perm = permutation_importance(final_pipe, X_test, y_test, n_repeats=10, random_state=RANDOM_STATE, n_jobs=-1)
imp = pd.Series(perm.importances_mean, index=X_test.columns).sort_values()
plt.figure(figsize=(8, 6)); imp.plot(kind="barh")
plt.title("Permutation Importance (drop in R2 when shuffled)"); plt.tight_layout()
plt.savefig(VISUALS / "06_feature_importance.png", dpi=150); plt.close()
print("\nTop factors:\n", imp.sort_values(ascending=False).head(8))

# %% 11. Reality check: model WITHOUT Previous_Scores
# A model that leans on past scores is less useful for early intervention. Compare both.
if "Previous_Scores" in X.columns:
    Xtr2, Xte2 = X_train.drop(columns="Previous_Scores"), X_test.drop(columns="Previous_Scores")
    pipe2 = Pipeline([("prep", build_preprocessor(Xtr2)), ("model", RandomForestRegressor(n_estimators=200, random_state=RANDOM_STATE, n_jobs=-1))])
    r2res, _ = evaluate("RF without Previous_Scores", pipe2, Xtr2, y_train, Xte2, y_test)
    print(r2res)

# %% 12. Save the model + metadata for the prediction app
joblib.dump(final_pipe, MODELS / "student_model.joblib")
meta = {
    "numeric": {c: {"min": float(X[c].min()), "max": float(X[c].max()), "median": float(X[c].median())}
                for c in X.select_dtypes(include=np.number).columns},
    "categorical": {c: sorted(X[c].dropna().unique().tolist()) for c in X.select_dtypes(include="object").columns},
    "columns": X.columns.tolist(),
    "at_risk_threshold": AT_RISK_THRESHOLD,
}
json.dump(meta, open(MODELS / "metadata.json", "w"), indent=2)
print("Saved model and metadata.")

# %% 13. Quick prediction demo
sample = X_test.iloc[[0]]
print("Sample student:\n", sample.T)
print("Predicted score:", round(float(final_pipe.predict(sample)[0]), 1), "| Actual:", y_test.iloc[0])
