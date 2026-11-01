{
 "cells": [
  {
   "cell_type": "code",
   "execution_count": 1,
   "id": "749e96c8-b011-4fb9-b2b3-71b997bda3e2",
   "metadata": {},
   "outputs": [],
   "source": [
    "import json\n",
    "import warnings\n",
    "from pathlib import Path\n",
    "\n",
    "import joblib\n",
    "import matplotlib.pyplot as plt\n",
    "import numpy as np\n",
    "import pandas as pd\n",
    "import seaborn as sns\n",
    "from sklearn.compose import ColumnTransformer\n",
    "from sklearn.ensemble import GradientBoostingRegressor, RandomForestRegressor\n",
    "from sklearn.impute import SimpleImputer\n",
    "from sklearn.inspection import permutation_importance\n",
    "from sklearn.linear_model import LinearRegression\n",
    "from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score\n",
    "from sklearn.model_selection import GridSearchCV, cross_val_score, train_test_split\n",
    "from sklearn.pipeline import Pipeline\n",
    "from sklearn.preprocessing import OneHotEncoder, StandardScaler"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 4,
   "id": "50a0f784-a49b-4308-8400-f47e966a9565",
   "metadata": {},
   "outputs": [],
   "source": [
    "warnings.filterwarnings(\"ignore\")\n",
    "sns.set_theme(style=\"whitegrid\")\n",
    "RANDOM_STATE = 42\n",
    "DATA_PATH = Path(\"data/StudentPerformanceFactors.csv\")\n",
    "VISUALS = Path(\"visuals\"); VISUALS.mkdir(exist_ok=True)\n",
    "MODELS = Path(\"models\"); MODELS.mkdir(exist_ok=True)\n",
    "TARGET = \"Exam_Score\"\n",
    "AT_RISK_THRESHOLD = 65"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 7,
   "id": "8bc4a21d-2e01-4840-8f5a-d909e7c6a987",
   "metadata": {},
   "outputs": [
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "Shape: (6607, 20)\n",
      "   Hours_Studied  Attendance Parental_Involvement Access_to_Resources  \\\n",
      "0             23          84                  Low                High   \n",
      "1             19          64                  Low              Medium   \n",
      "2             24          98               Medium              Medium   \n",
      "3             29          89                  Low              Medium   \n",
      "4             19          92               Medium              Medium   \n",
      "\n",
      "  Extracurricular_Activities  Sleep_Hours  Previous_Scores Motivation_Level  \\\n",
      "0                         No            7               73              Low   \n",
      "1                         No            8               59              Low   \n",
      "2                        Yes            7               91           Medium   \n",
      "3                        Yes            8               98           Medium   \n",
      "4                        Yes            6               65           Medium   \n",
      "\n",
      "  Internet_Access  Tutoring_Sessions Family_Income Teacher_Quality  \\\n",
      "0             Yes                  0           Low          Medium   \n",
      "1             Yes                  2        Medium          Medium   \n",
      "2             Yes                  2        Medium          Medium   \n",
      "3             Yes                  1        Medium          Medium   \n",
      "4             Yes                  3        Medium            High   \n",
      "\n",
      "  School_Type Peer_Influence  Physical_Activity Learning_Disabilities  \\\n",
      "0      Public       Positive                  3                    No   \n",
      "1      Public       Negative                  4                    No   \n",
      "2      Public        Neutral                  4                    No   \n",
      "3      Public       Negative                  4                    No   \n",
      "4      Public        Neutral                  4                    No   \n",
      "\n",
      "  Parental_Education_Level Distance_from_Home  Gender  Exam_Score  \n",
      "0              High School               Near    Male          67  \n",
      "1                  College           Moderate  Female          61  \n",
      "2             Postgraduate               Near    Male          74  \n",
      "3              High School           Moderate    Male          71  \n",
      "4                  College               Near  Female          70  \n",
      "\n",
      "Data types / non-null counts:\n",
      "<class 'pandas.DataFrame'>\n",
      "RangeIndex: 6607 entries, 0 to 6606\n",
      "Data columns (total 20 columns):\n",
      " #   Column                      Non-Null Count  Dtype\n",
      "---  ------                      --------------  -----\n",
      " 0   Hours_Studied               6607 non-null   int64\n",
      " 1   Attendance                  6607 non-null   int64\n",
      " 2   Parental_Involvement        6607 non-null   str  \n",
      " 3   Access_to_Resources         6607 non-null   str  \n",
      " 4   Extracurricular_Activities  6607 non-null   str  \n",
      " 5   Sleep_Hours                 6607 non-null   int64\n",
      " 6   Previous_Scores             6607 non-null   int64\n",
      " 7   Motivation_Level            6607 non-null   str  \n",
      " 8   Internet_Access             6607 non-null   str  \n",
      " 9   Tutoring_Sessions           6607 non-null   int64\n",
      " 10  Family_Income               6607 non-null   str  \n",
      " 11  Teacher_Quality             6529 non-null   str  \n",
      " 12  School_Type                 6607 non-null   str  \n",
      " 13  Peer_Influence              6607 non-null   str  \n",
      " 14  Physical_Activity           6607 non-null   int64\n",
      " 15  Learning_Disabilities       6607 non-null   str  \n",
      " 16  Parental_Education_Level    6517 non-null   str  \n",
      " 17  Distance_from_Home          6540 non-null   str  \n",
      " 18  Gender                      6607 non-null   str  \n",
      " 19  Exam_Score                  6607 non-null   int64\n",
      "dtypes: int64(7), str(13)\n",
      "memory usage: 1.4 MB\n",
      "\n",
      "Numeric summary:\n",
      "       Hours_Studied   Attendance  Sleep_Hours  Previous_Scores  \\\n",
      "count    6607.000000  6607.000000   6607.00000      6607.000000   \n",
      "mean       19.975329    79.977448      7.02906        75.070531   \n",
      "std         5.990594    11.547475      1.46812        14.399784   \n",
      "min         1.000000    60.000000      4.00000        50.000000   \n",
      "25%        16.000000    70.000000      6.00000        63.000000   \n",
      "50%        20.000000    80.000000      7.00000        75.000000   \n",
      "75%        24.000000    90.000000      8.00000        88.000000   \n",
      "max        44.000000   100.000000     10.00000       100.000000   \n",
      "\n",
      "       Tutoring_Sessions  Physical_Activity   Exam_Score  \n",
      "count        6607.000000        6607.000000  6607.000000  \n",
      "mean            1.493719           2.967610    67.235659  \n",
      "std             1.230570           1.031231     3.890456  \n",
      "min             0.000000           0.000000    55.000000  \n",
      "25%             1.000000           2.000000    65.000000  \n",
      "50%             1.000000           3.000000    67.000000  \n",
      "75%             2.000000           4.000000    69.000000  \n",
      "max             8.000000           6.000000   101.000000  \n",
      "\n",
      "Missing values per column:\n",
      "Teacher_Quality             78\n",
      "Parental_Education_Level    90\n",
      "Distance_from_Home          67\n",
      "dtype: int64\n"
     ]
    }
   ],
   "source": [
    "df = pd.read_csv(DATA_PATH)\n",
    "print(\"Shape:\", df.shape)\n",
    "print(df.head())\n",
    "print(\"\\nData types / non-null counts:\")\n",
    "df.info()\n",
    "print(\"\\nNumeric summary:\")\n",
    "print(df.describe())\n",
    "print(\"\\nMissing values per column:\")\n",
    "print(df.isnull().sum()[df.isnull().sum() > 0])"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 9,
   "id": "18d83671-a2ea-4a38-944a-537526193c2e",
   "metadata": {},
   "outputs": [
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "Removed 0 duplicate rows\n",
      "Removed 0 rows with impossible Exam_Score\n",
      "Missing values after cleaning: 235\n"
     ]
    }
   ],
   "source": [
    "before = len(df)\n",
    "df = df.drop_duplicates().reset_index(drop=True)\n",
    "print(f\"Removed {before - len(df)} duplicate rows\")\n",
    "\n",
    "cat_cols_all = df.select_dtypes(include=\"object\").columns\n",
    "for c in cat_cols_all:\n",
    "    df[c] = df[c].str.strip()\n",
    "\n",
    "bad = ((df[TARGET] > 100) | (df[TARGET] < 0)).sum()\n",
    "df = df[(df[TARGET] >= 0) & (df[TARGET] <= 100)].reset_index(drop=True)\n",
    "print(f\"Removed {bad} rows with impossible {TARGET}\")\n",
    "\n",
    "if \"Attendance\" in df.columns:\n",
    "    df = df[df[\"Attendance\"].between(0, 100)].reset_index(drop=True)\n",
    "\n",
    "print(\"Missing values after cleaning:\", int(df.isnull().sum().sum()))"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 10,
   "id": "0ff92249-3190-4725-b2e1-00955279495e",
   "metadata": {},
   "outputs": [
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "\n",
      "Correlation with target:\n",
      "Attendance           0.582458\n",
      "Hours_Studied        0.446514\n",
      "Previous_Scores      0.174461\n",
      "Tutoring_Sessions    0.153754\n",
      "Physical_Activity    0.027943\n",
      "Sleep_Hours         -0.016194\n",
      "Name: Exam_Score, dtype: float64\n"
     ]
    }
   ],
   "source": [
    "# %% 4. Exploratory Data Analysis (EDA)\n",
    "\n",
    "\n",
    "num_cols = df.select_dtypes(include=np.number).columns.drop(TARGET).tolist()\n",
    "cat_cols = df.select_dtypes(include=\"object\").columns.tolist()\n",
    "\n",
    "# 4a. Target distribution\n",
    "plt.figure(figsize=(7, 4))\n",
    "sns.histplot(df[TARGET], bins=25, kde=True)\n",
    "plt.title(\"Distribution of Exam Score\"); plt.tight_layout()\n",
    "plt.savefig(VISUALS / \"01_score_distribution.png\", dpi=150); plt.close()\n",
    "\n",
    "# 4b. Correlation heatmap (numeric features)\n",
    "plt.figure(figsize=(9, 7))\n",
    "sns.heatmap(df[num_cols + [TARGET]].corr(), annot=True, fmt=\".2f\", cmap=\"coolwarm\", center=0)\n",
    "plt.title(\"Correlation Heatmap\"); plt.tight_layout()\n",
    "plt.savefig(VISUALS / \"02_correlation_heatmap.png\", dpi=150); plt.close()\n",
    "\n",
    "# 4c. Key relationships with the score\n",
    "pairs = [c for c in [\"Hours_Studied\", \"Attendance\", \"Sleep_Hours\", \"Previous_Scores\"] if c in df.columns]\n",
    "fig, axes = plt.subplots(1, len(pairs), figsize=(5 * len(pairs), 4))\n",
    "axes = np.atleast_1d(axes)\n",
    "for ax, c in zip(axes, pairs):\n",
    "    sns.regplot(data=df, x=c, y=TARGET, ax=ax, scatter_kws={\"alpha\": 0.25, \"s\": 12}, line_kws={\"color\": \"red\"})\n",
    "    ax.set_title(f\"{c} vs {TARGET}\")\n",
    "plt.tight_layout(); plt.savefig(VISUALS / \"03_numeric_vs_score.png\", dpi=150); plt.close()\n",
    "\n",
    "# 4d. Categorical features vs average score\n",
    "show_cats = [c for c in [\"Parental_Involvement\", \"Motivation_Level\", \"Access_to_Resources\", \"Teacher_Quality\"] if c in df.columns]\n",
    "if show_cats:\n",
    "    fig, axes = plt.subplots(1, len(show_cats), figsize=(5 * len(show_cats), 4))\n",
    "    axes = np.atleast_1d(axes)\n",
    "    for ax, c in zip(axes, show_cats):\n",
    "        sns.boxplot(data=df, x=c, y=TARGET, ax=ax)\n",
    "        ax.set_title(f\"{c} vs {TARGET}\"); ax.tick_params(axis=\"x\", rotation=20)\n",
    "    plt.tight_layout(); plt.savefig(VISUALS / \"04_categorical_vs_score.png\", dpi=150); plt.close()\n",
    "\n",
    "# 4e. Top correlations, printed so you can write insights in your report\n",
    "print(\"\\nCorrelation with target:\")\n",
    "print(df[num_cols + [TARGET]].corr()[TARGET].drop(TARGET).sort_values(ascending=False))"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 18,
   "id": "1efe61e0-2ae7-442f-b192-1e2f176cc8de",
   "metadata": {},
   "outputs": [],
   "source": [
    "# Feature selection\n",
    "X = df.drop(columns=[TARGET])\n",
    "y = df[TARGET]"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 19,
   "id": "a655951d-9670-4882-9baa-5e3ff9f38adc",
   "metadata": {},
   "outputs": [
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "Train: (5284, 19) Test: (1322, 19)\n"
     ]
    }
   ],
   "source": [
    "# Train / test split + preprocessing pipeline\n",
    "X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=RANDOM_STATE)\n",
    "print(\"Train:\", X_train.shape, \"Test:\", X_test.shape)\n",
    "\n",
    "\n",
    "def build_preprocessor(X_ref):\n",
    "    nums = X_ref.select_dtypes(include=np.number).columns.tolist()\n",
    "    cats = X_ref.select_dtypes(include=\"object\").columns.tolist()\n",
    "    return ColumnTransformer([\n",
    "        (\"num\", Pipeline([(\"imp\", SimpleImputer(strategy=\"median\")), (\"sc\", StandardScaler())]), nums),\n",
    "        (\"cat\", Pipeline([(\"imp\", SimpleImputer(strategy=\"most_frequent\")),\n",
    "                          (\"oh\", OneHotEncoder(handle_unknown=\"ignore\"))]), cats),\n",
    "    ])\n",
    "\n",
    "\n",
    "def evaluate(name, pipe, Xtr, ytr, Xte, yte):\n",
    "    pipe.fit(Xtr, ytr)\n",
    "    pred = pipe.predict(Xte)\n",
    "    cv = cross_val_score(pipe, Xtr, ytr, cv=5, scoring=\"r2\")\n",
    "    res = {\n",
    "        \"Model\": name,\n",
    "        \"CV R2 (mean)\": round(cv.mean(), 4),\n",
    "        \"Test R2\": round(r2_score(yte, pred), 4),\n",
    "        \"Test MAE\": round(mean_absolute_error(yte, pred), 3),\n",
    "        \"Test RMSE\": round(np.sqrt(mean_squared_error(yte, pred)), 3),\n",
    "    }\n",
    "    return res, pred"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 20,
   "id": "7b256f17-1e11-4ba5-bfcf-4e2fac74c797",
   "metadata": {},
   "outputs": [
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "            Model  CV R2 (mean)  Test R2  Test MAE  Test RMSE\n",
      "Linear Regression        0.7166   0.8250     0.416      1.521\n",
      "Gradient Boosting        0.6774   0.7813     0.745      1.701\n",
      "    Random Forest        0.6211   0.7087     1.061      1.963\n"
     ]
    }
   ],
   "source": [
    "# Train and compare models\n",
    "models = {\n",
    "    \"Linear Regression\": LinearRegression(),\n",
    "    \"Random Forest\": RandomForestRegressor(n_estimators=200, random_state=RANDOM_STATE, n_jobs=-1),\n",
    "    \"Gradient Boosting\": GradientBoostingRegressor(random_state=RANDOM_STATE),\n",
    "}\n",
    "results, preds = [], {}\n",
    "for name, model in models.items():\n",
    "    pipe = Pipeline([(\"prep\", build_preprocessor(X_train)), (\"model\", model)])\n",
    "    r, p = evaluate(name, pipe, X_train, y_train, X_test, y_test)\n",
    "    results.append(r); preds[name] = p\n",
    "results_df = pd.DataFrame(results).sort_values(\"Test R2\", ascending=False)\n",
    "print(results_df.to_string(index=False))\n",
    "results_df.to_csv(\"models/model_results.csv\", index=False)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 21,
   "id": "c3a5a1b8-5bc0-455c-8750-6e8ca8623662",
   "metadata": {},
   "outputs": [
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "Best baseline: Linear Regression\n"
     ]
    }
   ],
   "source": [
    "# Improve the best model (light hyperparameter tuning)\n",
    "\n",
    "best_name = results_df.iloc[0][\"Model\"]\n",
    "print(\"Best baseline:\", best_name)\n",
    "if best_name == \"Linear Regression\":\n",
    "    final_pipe = Pipeline([(\"prep\", build_preprocessor(X_train)), (\"model\", LinearRegression())])\n",
    "    final_pipe.fit(X_train, y_train)\n",
    "else:\n",
    "    base = models[best_name]\n",
    "    grid = ({\"model__n_estimators\": [200, 400], \"model__max_depth\": [None, 10, 20], \"model__min_samples_leaf\": [1, 3]}\n",
    "            if best_name == \"Random Forest\" else\n",
    "            {\"model__n_estimators\": [100, 200], \"model__learning_rate\": [0.05, 0.1], \"model__max_depth\": [2, 3]})\n",
    "    search = GridSearchCV(Pipeline([(\"prep\", build_preprocessor(X_train)), (\"model\", base)]),\n",
    "                          grid, cv=3, scoring=\"r2\", n_jobs=-1)\n",
    "    search.fit(X_train, y_train)\n",
    "    print(\"Best params:\", search.best_params_)\n",
    "    final_pipe = search.best_estimator_"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 17,
   "id": "af49b24c-e7db-49e0-895b-6fa69fd59891",
   "metadata": {},
   "outputs": [
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "Final Test R2  : 0.8250\n",
      "Final Test MAE : 0.416\n",
      "Final Test RMSE: 1.521\n",
      "\n",
      "At-risk (<65): actual=297, caught=297, missed=0, false alarms=52\n"
     ]
    }
   ],
   "source": [
    "# 9. Final evaluation on the untouched test set\n",
    "final_pred = final_pipe.predict(X_test)\n",
    "print(f\"Final Test R2  : {r2_score(y_test, final_pred):.4f}\")\n",
    "print(f\"Final Test MAE : {mean_absolute_error(y_test, final_pred):.3f}\")\n",
    "print(f\"Final Test RMSE: {np.sqrt(mean_squared_error(y_test, final_pred)):.3f}\")\n",
    "\n",
    "fig, ax = plt.subplots(1, 2, figsize=(12, 4.5))\n",
    "ax[0].scatter(y_test, final_pred, alpha=0.3, s=12)\n",
    "lims = [y_test.min(), y_test.max()]\n",
    "ax[0].plot(lims, lims, \"r--\"); ax[0].set_xlabel(\"Actual\"); ax[0].set_ylabel(\"Predicted\")\n",
    "ax[0].set_title(\"Actual vs Predicted\")\n",
    "sns.histplot(y_test - final_pred, bins=30, kde=True, ax=ax[1]); ax[1].set_title(\"Residuals\")\n",
    "plt.tight_layout(); plt.savefig(VISUALS / \"05_actual_vs_predicted.png\", dpi=150); plt.close()\n",
    "\n",
    "actual_risk = y_test < AT_RISK_THRESHOLD\n",
    "pred_risk = pd.Series(final_pred, index=y_test.index) < AT_RISK_THRESHOLD\n",
    "tp = int((actual_risk & pred_risk).sum()); fn = int((actual_risk & ~pred_risk).sum()); fp = int((~actual_risk & pred_risk).sum())\n",
    "print(f\"\\nAt-risk (<{AT_RISK_THRESHOLD}): actual={int(actual_risk.sum())}, caught={tp}, missed={fn}, false alarms={fp}\")"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 23,
   "id": "15d3dc43-1695-43de-bd83-c317ad30679a",
   "metadata": {},
   "outputs": [
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "\n",
      "Top factors:\n",
      " Attendance              0.800754\n",
      "Hours_Studied           0.487586\n",
      "Previous_Scores         0.075351\n",
      "Access_to_Resources     0.075334\n",
      "Parental_Involvement    0.072707\n",
      "Tutoring_Sessions       0.054672\n",
      "Family_Income           0.025774\n",
      "Peer_Influence          0.024027\n",
      "dtype: float64\n"
     ]
    }
   ],
   "source": [
    "perm = permutation_importance(final_pipe, X_test, y_test, n_repeats=10, random_state=RANDOM_STATE, n_jobs=-1)\n",
    "\n",
    "imp = pd.Series(perm.importances_mean, index=X_test.columns).sort_values()\n",
    "\n",
    "plt.figure(figsize=(8, 6)); imp.plot(kind=\"barh\")\n",
    "plt.title(\"Permutation Importance (drop in R2 when shuffled)\"); plt.tight_layout()\n",
    "plt.savefig(VISUALS / \"06_feature_importance.png\", dpi=150); plt.close()\n",
    "print(\"\\nTop factors:\\n\", imp.sort_values(ascending=False).head(8))"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 24,
   "id": "5edc46c6-62d5-410e-9a60-c3a73aab9d4e",
   "metadata": {},
   "outputs": [
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "{'Model': 'RF without Previous_Scores', 'CV R2 (mean)': np.float64(0.5971), 'Test R2': 0.6892, 'Test MAE': 1.176, 'Test RMSE': np.float64(2.027)}\n"
     ]
    }
   ],
   "source": [
    "# Reality check: model WITHOUT Previous_Scores\n",
    "\n",
    "if \"Previous_Scores\" in X.columns:\n",
    "    Xtr2, Xte2 = X_train.drop(columns=\"Previous_Scores\"), X_test.drop(columns=\"Previous_Scores\")\n",
    "    pipe2 = Pipeline([(\"prep\", build_preprocessor(Xtr2)), (\"model\", RandomForestRegressor(n_estimators=200, random_state=RANDOM_STATE, n_jobs=-1))])\n",
    "    r2res, _ = evaluate(\"RF without Previous_Scores\", pipe2, Xtr2, y_train, Xte2, y_test)\n",
    "    print(r2res)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 25,
   "id": "440a839e-674b-499a-b3a2-7ece0b467ad3",
   "metadata": {},
   "outputs": [
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "Saved model and metadata.\n",
      "Sample student:\n",
      "                                    6414\n",
      "Hours_Studied                        16\n",
      "Attendance                           82\n",
      "Parental_Involvement             Medium\n",
      "Access_to_Resources              Medium\n",
      "Extracurricular_Activities           No\n",
      "Sleep_Hours                           6\n",
      "Previous_Scores                      91\n",
      "Motivation_Level                 Medium\n",
      "Internet_Access                     Yes\n",
      "Tutoring_Sessions                     0\n",
      "Family_Income                       Low\n",
      "Teacher_Quality                  Medium\n",
      "School_Type                      Public\n",
      "Peer_Influence                 Positive\n",
      "Physical_Activity                     3\n",
      "Learning_Disabilities                No\n",
      "Parental_Education_Level    High School\n",
      "Distance_from_Home                 Near\n",
      "Gender                           Female\n",
      "Predicted score: 66.0 | Actual: 66\n"
     ]
    }
   ],
   "source": [
    "#  Save the model + metadata for the prediction app\n",
    "joblib.dump(final_pipe, MODELS / \"student_model.joblib\")\n",
    "meta = {\n",
    "    \"numeric\": {c: {\"min\": float(X[c].min()), \"max\": float(X[c].max()), \"median\": float(X[c].median())}\n",
    "                for c in X.select_dtypes(include=np.number).columns},\n",
    "    \"categorical\": {c: sorted(X[c].dropna().unique().tolist()) for c in X.select_dtypes(include=\"object\").columns},\n",
    "    \"columns\": X.columns.tolist(),\n",
    "    \"at_risk_threshold\": AT_RISK_THRESHOLD,\n",
    "}\n",
    "json.dump(meta, open(MODELS / \"metadata.json\", \"w\"), indent=2)\n",
    "print(\"Saved model and metadata.\")\n",
    "\n",
    "sample = X_test.iloc[[0]]\n",
    "print(\"Sample student:\\n\", sample.T)\n",
    "print(\"Predicted score:\", round(float(final_pipe.predict(sample)[0]), 1), \"| Actual:\", y_test.iloc[0])"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "id": "639e0c3e-dc8b-4368-ab33-301a7d905cfd",
   "metadata": {},
   "outputs": [],
   "source": []
  }
 ],
 "metadata": {
  "kernelspec": {
   "display_name": "Python 3 (ipykernel)",
   "language": "python",
   "name": "python3"
  },
  "language_info": {
   "codemirror_mode": {
    "name": "ipython",
    "version": 3
   },
   "file_extension": ".py",
   "mimetype": "text/x-python",
   "name": "python",
   "nbconvert_exporter": "python",
   "pygments_lexer": "ipython3",
   "version": "3.11.17"
  }
 },
 "nbformat": 4,
 "nbformat_minor": 5
}
