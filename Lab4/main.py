import warnings

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.inspection import PartialDependenceDisplay, permutation_importance
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, roc_auc_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

warnings.filterwarnings("ignore")


# ==========================================
# 1. GENEROWANIE DANYCH MEDYCZNYCH
# ==========================================
rng = np.random.default_rng(7)
N = 600
age = rng.integers(18, 90, size=N)
sex = rng.choice(["F", "M"], size=N)
height = rng.normal(170, 10, size=N).clip(140, 200)
weight = rng.normal(75, 15, size=N).clip(40, 160)
bmi = weight / (height / 100) ** 2
sbp = (100 + 0.5 * age + 1.2 * bmi + (sex == "M") * 5 + rng.normal(0, 10, N)).round()
dbp = (60 + 0.2 * age + 0.6 * bmi + (sex == "M") * 3 + rng.normal(0, 6, N)).round()
glucose = rng.normal(105, 20, size=N).round().clip(60, 300)
hypertension = ((sbp >= 140) | (dbp >= 90)).astype(int)

df = pd.DataFrame(
    {
        "age": age,
        "sex": sex,
        "bmi": bmi.round(1),
        "systolic_bp": sbp.astype(int),
        "diastolic_bp": dbp.astype(int),
        "glucose": glucose.astype(int),
        "hypertension": hypertension,
    }
)

# ==========================================
# 2. PRZYGOTOWANIE MODELU
# ==========================================
features = ["age", "sex", "bmi", "glucose", "systolic_bp", "diastolic_bp"]
X = df[features].copy()
y = df["hypertension"].astype(int)

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.25, random_state=42, stratify=y
)

num = X_train.select_dtypes(include=[np.number]).columns.tolist()
cat = X_train.select_dtypes(exclude=[np.number]).columns.tolist()

pre = ColumnTransformer(
    [
        (
            "num",
            Pipeline(
                [("imp", SimpleImputer(strategy="median")), ("sc", StandardScaler())]
            ),
            num,
        ),
        (
            "cat",
            Pipeline(
                [
                    ("imp", SimpleImputer(strategy="most_frequent")),
                    ("oh", OneHotEncoder(handle_unknown="ignore")),
                ]
            ),
            cat,
        ),
    ]
)

clf = Pipeline([("pre", pre), ("model", LogisticRegression(max_iter=1000))]).fit(
    X_train, y_train
)

proba = clf.predict_proba(X_test)[:, 1]
pred = (proba >= 0.5).astype(int)
print(
    f"ACC: {accuracy_score(y_test, pred):.4f}, AUC: {roc_auc_score(y_test, proba):.4f}\n"
)

# ==========================================
# 3. PERMUTATION IMPORTANCE (PI)
# ==========================================
print("--- PERMUTATION IMPORTANCE ---")
pi = permutation_importance(clf, X_test, y_test, n_repeats=10, random_state=42)
imp = pd.DataFrame({"feature": X_test.columns, "PI": pi.importances_mean})
imp_sorted = imp.sort_values("PI", ascending=False)
print(imp_sorted)

plt.figure(figsize=(6, 4))
plt.barh(imp_sorted["feature"], imp_sorted["PI"])
plt.title("Permutation Importance (test)")
plt.gca().invert_yaxis()
plt.show()

# ==========================================
# 4. PDP i ICE
# ==========================================
print("\n--- GENEROWANIE PDP i ICE ---")
fig, ax = plt.subplots(figsize=(8, 4))
PartialDependenceDisplay.from_estimator(
    clf, X_test, features=["age", "bmi", "systolic_bp"], kind="average", ax=ax
)
plt.suptitle("Partial Dependence Plots (Globalne)")
plt.tight_layout()
plt.show()

fig, ax = plt.subplots(figsize=(6, 4))
PartialDependenceDisplay.from_estimator(
    clf, X_test, features=["age"], kind="individual", subsample=100, ax=ax
)
plt.suptitle("ICE Plot dla cechy: age (Lokalne)")
plt.tight_layout()
plt.show()

# ==========================================
# 5. SHAP
# ==========================================
print("\n--- SHAP ---")
try:
    import shap

    shap.initjs()
    explainer = shap.LinearExplainer(
        clf.named_steps["model"],
        shap.sample(clf.named_steps["pre"].transform(X_train), 200),
    )
    Xte = clf.named_steps["pre"].transform(X_test)

    # Przechwytujemy poprawne nazwy cech po OneHotEncoding
    feature_names = num + list(
        clf.named_steps["pre"]
        .transformers_[1][1]
        .named_steps["oh"]
        .get_feature_names_out(cat)
    )

    sv = explainer.shap_values(Xte)

    print("SHAP Summary Plot:")
    shap.summary_plot(sv, features=Xte, feature_names=feature_names, show=False)
    plt.show()

    print("SHAP Force Plot (Dla próbki 0):")
    shap.force_plot(
        explainer.expected_value,
        sv[0, :],
        Xte[0, :],
        feature_names=feature_names,
        matplotlib=True,
    )
    plt.show()
except Exception as e:
    print("SHAP niedostępny lub błąd:", e)

# ==========================================
# 6. LIME
# ==========================================
print("\n--- LIME ---")
try:
    from lime.lime_tabular import LimeTabularExplainer

    num_cols = X_train.select_dtypes(include=["number"]).columns.tolist()
    X_train_num = X_train[num_cols].to_numpy()
    X_test_num = X_test[num_cols].to_numpy()

    explainer_lime = LimeTabularExplainer(
        training_data=X_train_num,
        feature_names=num_cols,
        class_names=["noHT", "HT"],
        discretize_continuous=True,
        mode="classification",
    )

    def make_predict_fn_for_patient(sex_value):
        def predict_fn(x_num):
            X_df = pd.DataFrame(x_num, columns=num_cols)
            X_df["sex"] = sex_value
            return clf.predict_proba(X_df)

        return predict_fn

    i = 0
    x0_num = X_test_num[i]
    sex0 = X_test.iloc[i]["sex"]
    predict_fn = make_predict_fn_for_patient(sex0)

    exp = explainer_lime.explain_instance(
        data_row=x0_num, predict_fn=predict_fn, num_features=5, top_labels=1
    )

    print(f"Wyjaśnienie LIME dla pacjenta {i} | sex = {sex0}")
    for feat, val in exp.as_list(label=1):
        print(f"{feat} => {val:.5f}")
except Exception as e:
    print("LIME niedostępny lub błąd:", e)
