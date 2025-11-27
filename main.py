import os
import pandas as pd
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    confusion_matrix,
    classification_report,
)
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.dummy import DummyClassifier

PLOTS_DIR = "plots"
os.makedirs(PLOTS_DIR, exist_ok=True)


def load_data(path: str) -> pd.DataFrame:
    """Load CSV data."""
    df = pd.read_csv(path)
    return df


def descriptive_analysis(df: pd.DataFrame) -> None:
    print("1. BETIMLEYICI ANALIZ\n")

    print("---Veri Boyutu---")
    print(df.shape)
    print()

    print("---Veri Ornegi---")
    print(df.head())
    print()

    print("---Veri Tipleri---")
    print(df.dtypes)
    print()

    print("---Churn Dagilimi---")
    print(df["Exited"].value_counts())
    print(df["Exited"].value_counts(normalize=True) * 100)
    print()

    print("---Ulkelere gore churn dagilimi---")
    geo_churn = pd.crosstab(df["Geography"], df["Exited"], normalize="index")
    geo_churn.columns = ["Stayed", "Churned"]
    print(geo_churn)
    print()

    print("---Cinsiyete gore churn dagilimi---")
    gender_churn = pd.crosstab(df["Gender"], df["Exited"], normalize="index")
    gender_churn.columns = ["Stayed", "Churned"]
    print(gender_churn)
    print()

    if "IsActiveMember" in df.columns:
        print("---Aktiflige gore churn dagilimi---")
        active_churn = pd.crosstab(df["IsActiveMember"], df["Exited"], normalize="index")
        active_churn.columns = ["Stayed", "Churned"]
        print(active_churn)
        print()

    if "Satisfaction Score" in df.columns:
        print("---Memnuniyet skoruna gore ortalama churn---")
        sat_churn = df.groupby("Satisfaction Score")["Exited"].mean()
        print(sat_churn)
        print()

    if "Complain" in df.columns:
        print("---Sikayet eden ve churn olan musteriler---")
        complain_rate = pd.crosstab(df["Complain"], df["Exited"], normalize="index")
        complain_rate.columns = ["Stayed", "Churned"]
        print(complain_rate)
        print()

    # Churn distribution plot
    df["Exited"].value_counts().plot(kind="bar")
    plt.title("Churn Distribution (0=Stayed, 1=Churned)")
    plt.xlabel("Exited")
    plt.ylabel("Count")
    plt.tight_layout()
    plt.savefig(os.path.join(PLOTS_DIR, "plot_churn_distribution.png"))
    plt.close()

    # Age distribution plot
    df["Age"].hist(bins=20)
    plt.title("Age Distribution")
    plt.xlabel("Age")
    plt.ylabel("Frequency")
    plt.tight_layout()
    plt.savefig(os.path.join(PLOTS_DIR, "plot_age_distribution.png"))
    plt.close()


def preprocess_data(df: pd.DataFrame):
    print("2. VERI TEMIZLIGI\n")

    data = df.copy()

    id_cols = ["RowNumber", "CustomerId", "Surname"]
    data = data.drop(columns=id_cols, errors="ignore")
    print("Silinen kolonlar:", id_cols)

    if "Complain" in data.columns:
        data = data.drop(columns=["Complain"])
        print("Complain sutunu kaldirildi.")

    target_col = "Exited"
    y = data[target_col]
    X = data.drop(columns=[target_col])

    categorical_cols = X.select_dtypes(include=["object"]).columns.tolist()
    print("Kategorik degiskenler sayisal verilere donusturuldu:", categorical_cols)

    X_encoded = pd.get_dummies(X, columns=categorical_cols, drop_first=True)

    feature_names = X_encoded.columns.tolist()

    X_train, X_test, y_train, y_test = train_test_split(
        X_encoded,
        y,
        test_size=0.3,
        random_state=42,
        stratify=y,
    )

    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    print("Train/Test split ve olcekleme tamamlandi.\n")

    return X_train, X_test, X_train_scaled, X_test_scaled, y_train, y_test, feature_names


def evaluate_model(name: str, y_test, y_pred, y_proba=None) -> dict:
    acc = accuracy_score(y_test, y_pred)
    prec = precision_score(y_test, y_pred, zero_division=0)
    rec = recall_score(y_test, y_pred, zero_division=0)
    f1 = f1_score(y_test, y_pred, zero_division=0)
    roc = None
    if y_proba is not None:
        roc = roc_auc_score(y_test, y_proba)

    print(f"\n=== {name} ===")
    print(f"Dogruluk (Accuracy): {acc:.4f}")
    print(f"Duyarlilik (Recall): {rec:.4f}")
    print(f"Kesinlik (Precision): {prec:.4f}")
    print(f"F1 Skoru: {f1:.4f}")
    if roc is not None:
        print(f"ROC-AUC: {roc:.4f}")

    print("\nSiniflandirma Raporu:")
    print(classification_report(y_test, y_pred, zero_division=0))

    cm = confusion_matrix(y_test, y_pred)
    print("Confusion Matrix:")
    print(cm)

    return {
        "Model": name,
        "Accuracy": acc,
        "Precision": prec,
        "Recall": rec,
        "F1": f1,
        "ROC_AUC": roc,
    }


def train_baseline_model(X_train, X_test, y_train, y_test) -> dict:
    print("Baseline Model")
    dummy = DummyClassifier(strategy="most_frequent")
    dummy.fit(X_train, y_train)
    y_pred_dummy = dummy.predict(X_test)

    results = evaluate_model("Dummy (Most Frequent)", y_test, y_pred_dummy, y_proba=None)
    return results


def train_and_evaluate_models(
    X_train,
    X_test,
    X_train_scaled,
    X_test_scaled,
    y_train,
    y_test,
    feature_names,
) -> pd.DataFrame:
    print("3. MODELLEME\n")
    results = []

    # Logistic Regression
    log_reg = LogisticRegression(max_iter=1000, class_weight="balanced")
    log_reg.fit(X_train_scaled, y_train)
    y_pred_lr = log_reg.predict(X_test_scaled)
    y_proba_lr = log_reg.predict_proba(X_test_scaled)[:, 1]
    results.append(evaluate_model("Lojistik Regresyon", y_test, y_pred_lr, y_proba_lr))

    # Random Forest
    rf = RandomForestClassifier(
        n_estimators=200,
        random_state=42,
        n_jobs=-1,
        class_weight="balanced",
    )
    rf.fit(X_train, y_train)
    y_pred_rf = rf.predict(X_test)
    y_proba_rf = rf.predict_proba(X_test)[:, 1]
    results.append(evaluate_model("Random Forest", y_test, y_pred_rf, y_proba_rf))

    # KNN
    knn = KNeighborsClassifier(n_neighbors=5)
    knn.fit(X_train_scaled, y_train)
    y_pred_knn = knn.predict(X_test_scaled)
    y_proba_knn = knn.predict_proba(X_test_scaled)[:, 1]
    results.append(evaluate_model("KNN", y_test, y_pred_knn, y_proba_knn))

    # Gradient Boosting
    gb = GradientBoostingClassifier(random_state=42)
    gb.fit(X_train, y_train)
    y_pred_gb = gb.predict(X_test)
    y_proba_gb = gb.predict_proba(X_test)[:, 1]
    results.append(
        evaluate_model("Gradient Boosting", y_test, y_pred_gb, y_proba_gb)
    )

    results_df = pd.DataFrame(results)
    print("\n---Global Model Karsilastirmasi---")
    print(results_df)

    return results_df


def plot_global_model_metrics(
    global_results: pd.DataFrame,
    filename: str = "plot_global_model_metrics.png",
) -> None:
    """Plot global models' metrics with legend on the right."""
    metrics = ["Accuracy", "Recall", "Precision", "F1", "ROC_AUC"]
    df_plot = global_results.set_index("Model")[metrics].copy()
    df_plot = df_plot.fillna(0.0)

    ax = df_plot.plot(kind="bar")
    plt.title("Global Modellerin Performansi")
    plt.ylabel("Skor")
    plt.xticks(rotation=45, ha="right")
    plt.legend(loc="center left", bbox_to_anchor=(1.02, 0.5))
    plt.tight_layout()
    plt.savefig(os.path.join(PLOTS_DIR, filename))
    plt.close()


def plot_baseline_vs_models(
    global_results: pd.DataFrame,
    baseline_results: dict,
    filename: str = "plot_baseline_vs_models.png",
) -> None:
    """Plot baseline vs other models' metrics."""
    metrics = ["Accuracy", "Recall", "Precision", "F1"]
    df_plot = global_results.set_index("Model")[metrics].copy()
    baseline_row = [baseline_results[m] for m in metrics]
    df_plot.loc[baseline_results["Model"]] = baseline_row
    df_plot = df_plot.fillna(0.0)

    ax = df_plot.plot(kind="bar")
    plt.title("Baseline vs Modellerin Performansi")
    plt.ylabel("Skor")
    plt.xticks(rotation=45, ha="right")
    plt.legend(loc="center left", bbox_to_anchor=(1.02, 0.5))
    plt.tight_layout()
    plt.savefig(os.path.join(PLOTS_DIR, filename))
    plt.close()


def plot_single_model_metrics(
    results: dict,
    filename: str = "plot_baseline_metrics.png",
) -> None:
    """Plot single model (e.g., baseline) metrics as a bar chart."""
    metrics = ["Accuracy", "Recall", "Precision", "F1"]
    values = [results[m] for m in metrics]

    plt.bar(metrics, values)
    plt.ylim(0, 1)
    plt.title(f"{results['Model']} Metrikleri")
    plt.ylabel("Skor")
    plt.tight_layout()
    plt.savefig(os.path.join(PLOTS_DIR, filename))
    plt.close()


def plot_segment_model_metrics(
    results_df: pd.DataFrame,
    segment_name: str,
) -> None:
    """Plot model metrics for a specific country segment."""
    metrics = ["Accuracy", "Recall", "Precision", "F1", "ROC_AUC"]
    df_plot = results_df.set_index("Model")[metrics].copy()
    df_plot = df_plot.fillna(0.0)

    safe_name = (
        segment_name.lower()
        .replace(" ", "_")
        .replace("(", "")
        .replace(")", "")
        .replace("+", "plus")
    )

    ax = df_plot.plot(kind="bar")
    plt.title(f"{segment_name} - Model Performanslari")
    plt.ylabel("Skor")
    plt.xticks(rotation=45, ha="right")
    plt.legend(loc="center left", bbox_to_anchor=(1.02, 0.5))
    plt.tight_layout()
    plt.savefig(os.path.join(PLOTS_DIR, f"plot_segment_{safe_name}_metrics.png"))
    plt.close()


def run_country_segment_models(df: pd.DataFrame) -> None:
    segments = {
        "Germany": df[df["Geography"] == "Germany"],
        "Other Countries (France + Spain)": df[df["Geography"] != "Germany"],
    }

    for segment_name, segment_df in segments.items():
        print("\n\n############################")
        print(f"COUNTRY SEGMENT: {segment_name}")
        print("############################\n")

        if segment_df.empty:
            print(f"{segment_name} segmenti icin veri bulunamadi, atlandi.")
            continue

        (
            X_train,
            X_test,
            X_train_scaled,
            X_test_scaled,
            y_train,
            y_test,
            feature_names,
        ) = preprocess_data(segment_df)

        results_df = train_and_evaluate_models(
            X_train,
            X_test,
            X_train_scaled,
            X_test_scaled,
            y_train,
            y_test,
            feature_names,
        )

        print("\nSummary metrics for segment:")
        print(results_df)

        # Segment grafiği
        plot_segment_model_metrics(results_df, segment_name)
        print(
            f"{segment_name} segmenti için metrik grafiği plots klasörüne kaydedildi."
        )


def main():
    data_path = "data.csv"
    df = load_data(data_path)

    descriptive_analysis(df)

    (
        X_train,
        X_test,
        X_train_scaled,
        X_test_scaled,
        y_train,
        y_test,
        feature_names,
    ) = preprocess_data(df)

    baseline_results = train_baseline_model(X_train, X_test, y_train, y_test)
    global_results = train_and_evaluate_models(
        X_train,
        X_test,
        X_train_scaled,
        X_test_scaled,
        y_train,
        y_test,
        feature_names,
    )

    print("\n---Baseline Model Metrikleri---")
    print(baseline_results)
    print("\n---Genel Model Metrikleri---")
    print(global_results)

    # Global görseller
    plot_global_model_metrics(global_results)
    plot_baseline_vs_models(global_results, baseline_results)
    plot_single_model_metrics(baseline_results)

    # Country segment modelleri + grafikleri
    run_country_segment_models(df)


if __name__ == "__main__":
    main()
