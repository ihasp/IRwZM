import os
import json
from datetime import datetime

import joblib
import pandas as pd
import streamlit as st

from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    roc_auc_score,
    accuracy_score,
    classification_report,
    confusion_matrix,
)

import matplotlib.pyplot as plt

DATA_PATH = "health_measurements.csv"
MODEL_PATH = "risk_model.joblib"
CONFIG_PATH = "config.json"

st.set_page_config(page_title="Monitor zdrowia + ML", layout="centered")
st.title("📱 Monitor zdrowia + analiza ML (demo)")


# -----------------------------
# Pomocnicze: inicjalizacja
# -----------------------------
def load_config():
    if os.path.exists(CONFIG_PATH):
        with open(CONFIG_PATH, "r") as f:
            return json.load(f)
    return {"sbp_thr": 140, "dbp_thr": 90}


def save_config(config):
    with open(CONFIG_PATH, "w") as f:
        json.dump(config, f)


def ensure_data_file():
    if not os.path.exists(DATA_PATH):
        df = pd.DataFrame(
            columns=[
                "timestamp",
                "age",
                "bmi",
                "glucose",
                "systolic_bp",
                "diastolic_bp",
            ]
        )
        df.to_csv(DATA_PATH, index=False)


def load_data():
    ensure_data_file()
    return pd.read_csv(DATA_PATH)


def append_measurement(row: dict):
    df = load_data()
    df = pd.concat([df, pd.DataFrame([row])], ignore_index=True)
    df.to_csv(DATA_PATH, index=False)


def make_demo_label(df: pd.DataFrame, sbp_thr: int, dbp_thr: int) -> pd.Series:
    """
    Etykieta bazuje teraz na progach zdefiniowanych przez użytkownika.
    """
    return ((df["systolic_bp"] >= sbp_thr) | (df["diastolic_bp"] >= dbp_thr)).astype(
        int
    )


def train_model(df: pd.DataFrame, config: dict):
    if len(df) < 20:
        raise ValueError(
            "Za mało danych do trenowania (min. 20 pomiarów). Dodaj więcej wpisów."
        )

    y = make_demo_label(df, config["sbp_thr"], config["dbp_thr"])
    X = df[["age", "bmi", "glucose", "systolic_bp", "diastolic_bp"]].copy()

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.25, random_state=42, stratify=y
    )

    num_cols = list(X.columns)
    pre = ColumnTransformer(
        transformers=[
            (
                "num",
                Pipeline(
                    steps=[
                        ("imputer", SimpleImputer(strategy="median")),
                        ("scaler", StandardScaler()),
                    ]
                ),
                num_cols,
            )
        ],
        remainder="drop",
    )

    clf = Pipeline(steps=[("pre", pre), ("model", LogisticRegression(max_iter=2000))])

    clf.fit(X_train, y_train)

    # metryki
    proba = clf.predict_proba(X_test)[:, 1]
    pred = (proba >= 0.5).astype(int)

    # Predykcja sztywnymi regułami
    rule_pred = make_demo_label(X_test, config["sbp_thr"], config["dbp_thr"])

    metrics = {
        "accuracy": float(accuracy_score(y_test, pred)),
        "rule_accuracy": float(
            accuracy_score(y_test, rule_pred)
        ),  # Bazowa precyzja reguł
        "roc_auc": float(roc_auc_score(y_test, proba))
        if len(set(y_test)) > 1
        else None,
        "confusion_matrix": confusion_matrix(y_test, pred).tolist(),
        "report": classification_report(y_test, pred, digits=3, zero_division=0),
    }

    joblib.dump({"model": clf, "metrics": metrics}, MODEL_PATH)
    return clf, metrics


def load_model():
    if os.path.exists(MODEL_PATH):
        obj = joblib.load(MODEL_PATH)
        return obj["model"], obj["metrics"]
    return None, None


config = load_config()

# =========================
# ETAP 1: Zbieranie danych
# =========================
st.header("Etap 1 — Zbieranie danych i personalizacja progów")

st.subheader("⚙️ Konfiguracja progów")
c1, c2 = st.columns(2)
with c1:
    new_sbp = st.number_input("Próg SBP (np. 140)", value=config["sbp_thr"], step=1)
with c2:
    new_dbp = st.number_input("Próg DBP (np. 90)", value=config["dbp_thr"], step=1)

if st.button("Zapisz progi"):
    config["sbp_thr"] = new_sbp
    config["dbp_thr"] = new_dbp
    save_config(config)
    st.success("Zapisano nowe progi do config.json!")

st.subheader("📝 Nowy pomiar")
with st.form("health_form", clear_on_submit=False):
    col1, col2 = st.columns(2)
    with col1:
        age = st.number_input(
            "Wiek [lata]", min_value=18, max_value=110, value=40, step=1
        )
        bmi = st.number_input(
            "BMI", min_value=10.0, max_value=60.0, value=24.0, step=0.1
        )
        glucose = st.number_input(
            "Glukoza [mg/dl]", min_value=40, max_value=300, value=95, step=1
        )
    with col2:
        systolic_bp = st.number_input(
            "Ciśnienie skurczowe SBP [mmHg]",
            min_value=70,
            max_value=260,
            value=120,
            step=1,
        )
        diastolic_bp = st.number_input(
            "Ciśnienie rozkurczowe DBP [mmHg]",
            min_value=40,
            max_value=150,
            value=80,
            step=1,
        )

    submitted = st.form_submit_button("💾 Zapisz pomiar")

if submitted:
    row = {
        "timestamp": datetime.now().isoformat(timespec="seconds"),
        "age": int(age),
        "bmi": float(bmi),
        "glucose": int(glucose),
        "systolic_bp": int(systolic_bp),
        "diastolic_bp": int(diastolic_bp),
    }
    append_measurement(row)
    st.success("Zapisano pomiar do pliku health_measurements.csv")

df = load_data()
st.caption(f"Liczba zapisanych pomiarów: {len(df)}")
st.dataframe(df.tail(10), use_container_width=True)


# =====================================
# ETAP 2: Analiza i wizualizacja danych
# =====================================
st.header("Etap 2 — Analiza i wizualizacja")

if len(df) == 0:
    st.info("Dodaj co najmniej jeden pomiar, aby zobaczyć analizę.")
else:
    st.subheader("Wykres trendu z progami")
    plot_cols = st.multiselect(
        "Wybierz parametry do wykresu:",
        options=["bmi", "glucose", "systolic_bp", "diastolic_bp"],
        default=["systolic_bp", "diastolic_bp"],
    )

    if plot_cols:
        df_plot = df.copy()
        df_plot["timestamp"] = pd.to_datetime(df_plot["timestamp"], errors="coerce")
        df_plot = df_plot.dropna(subset=["timestamp"]).sort_values("timestamp").tail(50)

        fig = plt.figure(figsize=(7, 4))
        for c in plot_cols:
            plt.plot(df_plot["timestamp"], df_plot[c], label=c)
            # Rysowanie progów z configu
            if c == "systolic_bp":
                plt.axhline(
                    config["sbp_thr"],
                    color="red",
                    linestyle="--",
                    label="Próg SBP",
                    alpha=0.6,
                )
            if c == "diastolic_bp":
                plt.axhline(
                    config["dbp_thr"],
                    color="orange",
                    linestyle="--",
                    label="Próg DBP",
                    alpha=0.6,
                )

        plt.xlabel("czas")
        plt.ylabel("wartość")
        plt.xticks(rotation=30, ha="right")
        plt.legend()
        plt.tight_layout()
        st.pyplot(fig)

    st.subheader("Szybka flaga progowa (własne progi)")
    df_flag = df.tail(10).copy()
    df_flag["flag_high_bp"] = make_demo_label(
        df_flag, config["sbp_thr"], config["dbp_thr"]
    )
    st.dataframe(
        df_flag[["timestamp", "systolic_bp", "diastolic_bp", "flag_high_bp"]],
        use_container_width=True,
    )


# ==============================
# ETAP 3: Model uczenia maszynowego
# ==============================
st.header("Etap 3 — Budowa modelu ML vs Progi")

model, metrics = load_model()

colA, colB = st.columns([1, 2])
with colA:
    if st.button("🧠 Wytrenuj / odśwież model"):
        try:
            model, metrics = train_model(df, config)
            st.success("Model został wytrenowany (risk_model.joblib).")
        except Exception as e:
            st.error(str(e))

with colB:
    if metrics:
        st.subheader("Porównanie (część testowa)")
        st.write(
            f"Skuteczność (Accuracy) reguł progowych: **{metrics.get('rule_accuracy', 1.0):.3f}** (Punkt odniesienia)"
        )
        st.write(f"Skuteczność (Accuracy) modelu ML: **{metrics['accuracy']:.3f}**")

        if metrics["roc_auc"] is not None:
            st.write(f"ROC AUC (ML): **{metrics['roc_auc']:.3f}**")
    else:
        st.info("Model nie jest jeszcze wytrenowany. Kliknij przycisk obok.")


# ===================================
# ETAP 4: Integracja modelu z aplikacją
# ===================================
st.header("Etap 4 — Progowo vs ML dla bieżącego pomiaru")

if model is None:
    st.warning("Najpierw wytrenuj model w Etapie 3.")
else:
    X_one = pd.DataFrame(
        [
            {
                "age": int(age),
                "bmi": float(bmi),
                "glucose": int(glucose),
                "systolic_bp": int(systolic_bp),
                "diastolic_bp": int(diastolic_bp),
            }
        ]
    )

    # 1. Wynik progowy
    rule_pred = (
        1
        if (systolic_bp >= config["sbp_thr"] or diastolic_bp >= config["dbp_thr"])
        else 0
    )

    # 2. Wynik ML
    proba = float(model.predict_proba(X_one)[0, 1])
    ml_pred = int(proba >= 0.5)

    c_rule, c_ml = st.columns(2)
    with c_rule:
        st.subheader("Logika Progowa")
        if rule_pred == 1:
            st.error(
                f"Wynik: **PODWYŻSZONE RYZYKO** \n\n(Bieżący pomiar {int(systolic_bp)}/{int(diastolic_bp)} przekracza próg {config['sbp_thr']}/{config['dbp_thr']})"
            )
        else:
            st.success("Wynik: **NISKIE RYZYKO** (Bieżący pomiar poniżej progów)")

    with c_ml:
        st.subheader("Predykcja ML")
        st.write(f"Prawdopodobieństwo: **{proba:.3f}**")
        if ml_pred == 1:
            st.error("Wynik: **PODWYŻSZONE RYZYKO**")
        else:
            st.success("Wynik: **NISKIE RYZYKO**")

st.divider()
st.caption("Pliki lokalne: health_measurements.csv, risk_model.joblib, config.json")
