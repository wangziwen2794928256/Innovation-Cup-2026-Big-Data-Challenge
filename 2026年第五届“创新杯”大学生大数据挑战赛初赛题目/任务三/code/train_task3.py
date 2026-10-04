from __future__ import annotations

import json
import os
import time
from itertools import product
from pathlib import Path

os.environ.setdefault("MPLCONFIGDIR", "/tmp/mpl-task3")

import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.calibration import calibration_curve
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import ExtraTreesClassifier, HistGradientBoostingClassifier, RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.inspection import permutation_importance
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_recall_curve,
    roc_auc_score,
    roc_curve,
)
from sklearn.model_selection import StratifiedKFold, train_test_split
from sklearn.neural_network import MLPClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, OrdinalEncoder, StandardScaler, label_binarize

ROOT = Path(__file__).resolve().parents[1]
DATA = Path(os.environ.get("TASK3_DATA", ROOT / "data" / "A题数据集.csv"))
OUT = Path(os.environ.get("TASK3_OUTPUT", ROOT / "reproduce_results"))
FIG = OUT / "figures"
TAB = OUT / "tables"
MOD = OUT / "models"
PRED = OUT / "predictions"
for p in (FIG, TAB, MOD, PRED):
    p.mkdir(parents=True, exist_ok=True)

SEED = 2026
CLASS_ORDER = ["Average", "Good", "Excellent"]
CLASS_COLORS = {"Average": "#D8A03B", "Good": "#4E91A8", "Excellent": "#398564"}
MODEL_COLORS = {
    "Logistic": "#7A6FA8",
    "RandomForest": "#6A8CAF",
    "HistGB": "#D18B47",
    "ExtraTrees": "#3B8F83",
    "MLP": "#B65C5A",
    "Fusion": "#225B6A",
}
DIRECT = ["Health_Score", "Fitness_Level"]
COMPOSITE = [
    "Healthy_Aging_Score",
    "Obesity_Risk",
    "Hypertension_Risk",
    "Diabetes_Risk",
    "Cardiovascular_Risk",
    "Sleep_Disorder_Risk",
]


def setup_style() -> None:
    plt.rcParams.update(
        {
            "font.family": "sans-serif",
            "font.sans-serif": ["DejaVu Sans"],
            "axes.unicode_minus": False,
            "figure.facecolor": "white",
            "axes.facecolor": "white",
            "axes.edgecolor": "#4D4D4D",
            "axes.grid": True,
            "grid.color": "#D9D9D9",
            "grid.alpha": 0.35,
            "grid.linewidth": 0.55,
            "axes.titlesize": 12,
            "axes.labelsize": 10,
            "xtick.labelsize": 9,
            "ytick.labelsize": 9,
            "legend.fontsize": 8.5,
            "savefig.dpi": 500,
            "savefig.bbox": "tight",
        }
    )


def savefig(name: str) -> None:
    plt.savefig(FIG / f"{name}.png", dpi=500, bbox_inches="tight", pad_inches=0.08)
    plt.savefig(FIG / f"{name}.pdf", bbox_inches="tight", pad_inches=0.08)
    plt.close()


def time_to_minutes(s: pd.Series) -> pd.Series:
    part = s.astype(str).str.split(":", n=1, expand=True)
    return pd.to_numeric(part[0], errors="coerce") * 60 + pd.to_numeric(part[1], errors="coerce")


def engineer(df: pd.DataFrame, scope: str) -> pd.DataFrame:
    x = df.drop(columns=["Wellness_Category", "Person_ID"], errors="ignore").copy()
    if scope in {"NO_DIRECT", "NO_COMPOSITE"}:
        x = x.drop(columns=DIRECT, errors="ignore")
    if scope == "NO_COMPOSITE":
        x = x.drop(columns=COMPOSITE, errors="ignore")

    for c in ["Exercise_Type", "Workout_Intensity"]:
        if c in x:
            x[c] = x[c].fillna("None")
    if "Alcohol_Consumption" in x:
        x["Alcohol_Consumption_is_missing"] = x["Alcohol_Consumption"].isna().astype(int)
        x["Alcohol_Consumption"] = x["Alcohol_Consumption"].fillna("Unknown")

    wake = time_to_minutes(x["Wake_Up_Time"])
    sleep = time_to_minutes(x["Sleep_Time"])
    x["wake_sin"] = np.sin(2 * np.pi * wake / 1440)
    x["wake_cos"] = np.cos(2 * np.pi * wake / 1440)
    x["sleep_sin"] = np.sin(2 * np.pi * sleep / 1440)
    x["sleep_cos"] = np.cos(2 * np.pi * sleep / 1440)
    x["cross_midnight"] = (sleep > wake).astype(int)
    calc_sleep = ((wake - sleep) % 1440) / 60
    x["calculated_sleep_duration"] = calc_sleep
    x["sleep_duration_difference"] = (calc_sleep - x["Sleep_Duration_Hours"]).abs()
    x = x.drop(columns=["Wake_Up_Time", "Sleep_Time"])

    eps = 1e-6
    x["sleep_interruption_density"] = x["Number_of_Night_Awakenings"] / (x["Sleep_Duration_Hours"] + eps)
    x["screen_sleep_ratio"] = x["Screen_Time_Before_Bed_Hours"] / (x["Sleep_Duration_Hours"] + eps)
    x["sleep_duration_deviation"] = (x["Sleep_Duration_Hours"] - 8).abs()
    x["weekly_exercise_minutes"] = x["Exercise_Frequency_Per_Week"] * x["Exercise_Duration_Minutes"]
    x["steps_per_sitting_hour"] = x["Daily_Steps"] / (x["Sitting_Hours_Per_Day"] + eps)
    x["fruit_vegetable_total"] = x["Fruit_Intake_Per_Day"] + x["Vegetable_Intake_Per_Day"]
    x["sugary_fastfood_index"] = x["Sugary_Drinks_Per_Week"] + x["Fast_Food_Meals_Per_Week"]
    x["protein_per_kg"] = x["Protein_Intake_Grams"] / (x["Weight_kg"] + eps)
    x["water_per_kg"] = x["Water_Intake_Liters"] / (x["Weight_kg"] + eps)
    x["anxiety_depression_index"] = x["Anxiety_Score"] + x["Depression_Risk_Score"]
    x["mood_satisfaction_index"] = x["Mood_Score"] + x["Life_Satisfaction_Score"]
    x["stress_work_interaction"] = x["Stress_Level"] * x["Working_Hours_Per_Day"]
    x["energy_fatigue_difference"] = x["Energy_Level_Score"] - x["Fatigue_Level_Score"]
    x["pulse_pressure"] = x["Systolic_BP"] - x["Diastolic_BP"]
    x["mean_arterial_pressure"] = (x["Systolic_BP"] + 2 * x["Diastolic_BP"]) / 3
    x["bmi_recalculated"] = x["Weight_kg"] / ((x["Height_cm"] / 100) ** 2 + eps)
    return x.replace([np.inf, -np.inf], np.nan)


def preprocessor(x: pd.DataFrame, neural: bool) -> ColumnTransformer:
    nums = x.select_dtypes(include=np.number).columns.tolist()
    cats = [c for c in x.columns if c not in nums]
    if neural:
        num_pipe = Pipeline([("imputer", SimpleImputer(strategy="median")), ("scale", StandardScaler())])
        cat_pipe = Pipeline(
            [
                ("imputer", SimpleImputer(strategy="most_frequent")),
                ("onehot", OneHotEncoder(handle_unknown="ignore", sparse_output=False)),
            ]
        )
    else:
        num_pipe = Pipeline([("imputer", SimpleImputer(strategy="median"))])
        cat_pipe = Pipeline(
            [
                ("imputer", SimpleImputer(strategy="most_frequent")),
                ("ordinal", OrdinalEncoder(handle_unknown="use_encoded_value", unknown_value=-1)),
            ]
        )
    return ColumnTransformer([("num", num_pipe, nums), ("cat", cat_pipe, cats)], verbose_feature_names_out=False)


def make_model(name: str, x: pd.DataFrame) -> Pipeline:
    if name == "Logistic":
        clf = LogisticRegression(C=1.0, max_iter=800, class_weight="balanced", random_state=SEED)
        return Pipeline([("pre", preprocessor(x, True)), ("clf", clf)])
    if name == "RandomForest":
        clf = RandomForestClassifier(
            n_estimators=450, max_features="sqrt", min_samples_leaf=1,
            class_weight="balanced_subsample", n_jobs=-1, random_state=SEED,
        )
        return Pipeline([("pre", preprocessor(x, False)), ("clf", clf)])
    if name == "HistGB":
        clf = HistGradientBoostingClassifier(
            learning_rate=0.08, max_iter=260, max_leaf_nodes=31,
            l2_regularization=0.5, random_state=SEED,
        )
        return Pipeline([("pre", preprocessor(x, False)), ("clf", clf)])
    if name == "ExtraTrees":
        clf = ExtraTreesClassifier(
            n_estimators=600, max_features=0.8, min_samples_leaf=1,
            class_weight="balanced", n_jobs=-1, random_state=SEED,
        )
        return Pipeline([("pre", preprocessor(x, False)), ("clf", clf)])
    if name == "MLP":
        clf = MLPClassifier(
            hidden_layer_sizes=(128, 64, 32), activation="relu", solver="adam",
            alpha=3e-4, batch_size=128, learning_rate_init=8e-4,
            max_iter=140, early_stopping=False, tol=1e-4,
            n_iter_no_change=14, random_state=SEED,
        )
        return Pipeline([("pre", preprocessor(x, True)), ("clf", clf)])
    raise KeyError(name)


def align_prob(model: Pipeline, prob: np.ndarray) -> np.ndarray:
    classes = list(model.named_steps["clf"].classes_)
    return prob[:, [classes.index(c) for c in CLASS_ORDER]]


def metrics(y: np.ndarray, prob: np.ndarray) -> dict:
    pred = np.asarray(CLASS_ORDER)[prob.argmax(axis=1)]
    yb = label_binarize(y, classes=CLASS_ORDER)
    yid = pd.Series(y).map({c: i for i, c in enumerate(CLASS_ORDER)}).to_numpy()
    ll = -np.log(np.clip(prob[np.arange(len(yid)), yid], 1e-15, 1)).mean()
    return {
        "ACC": accuracy_score(y, pred),
        "Macro_F1": f1_score(y, pred, average="macro"),
        "Weighted_F1": f1_score(y, pred, average="weighted"),
        "Macro_AUC": roc_auc_score(yb, prob, multi_class="ovr", average="macro"),
        "Log_Loss": ll,
    }


def oof_model(name: str, x: pd.DataFrame, y: np.ndarray, folds: list[tuple[np.ndarray, np.ndarray]]) -> tuple[np.ndarray, list[dict]]:
    prob = np.zeros((len(x), len(CLASS_ORDER)))
    fold_rows = []
    for fold, (tr, va) in enumerate(folds):
        t0 = time.time()
        model = make_model(name, x.iloc[tr])
        model.fit(x.iloc[tr], y[tr])
        p = align_prob(model, model.predict_proba(x.iloc[va]))
        prob[va] = p
        fold_rows.append({"model": name, "fold": fold, **metrics(y[va], p), "seconds": time.time() - t0})
        print(name, "fold", fold, fold_rows[-1], flush=True)
    return prob, fold_rows


def search_fusion(oof: dict[str, np.ndarray], y: np.ndarray) -> tuple[dict[str, float], np.ndarray, pd.DataFrame]:
    names = ["Logistic", "HistGB", "ExtraTrees", "MLP"]
    rows = []
    best = None
    grid = np.arange(0, 1.001, 0.05)
    for a in grid:
        for b in grid[grid <= 1 - a + 1e-9]:
            for c in grid[grid <= 1 - a - b + 1e-9]:
                d = 1 - a - b - c
                if d < -1e-9:
                    continue
                w = [float(a), float(b), float(c), float(max(0, d))]
                p = sum(wi * oof[n] for wi, n in zip(w, names))
                m = metrics(y, p)
                row = {f"w_{n}": wi for n, wi in zip(names, w)}
                row.update(m)
                rows.append(row)
                key = (m["ACC"], m["Macro_F1"], -m["Log_Loss"])
                if best is None or key > best[0]:
                    best = (key, w, p)
    table = pd.DataFrame(rows).sort_values(["ACC", "Macro_F1", "Log_Loss"], ascending=[False, False, True])
    return dict(zip(names, best[1])), best[2], table


def plot_class_distribution(original: pd.Series, y3: pd.Series) -> None:
    setup_style()
    fig, ax = plt.subplots(1, 2, figsize=(11.2, 4.2))
    order4 = ["Poor", "Average", "Good", "Excellent"]
    colors4 = ["#A65654", "#D8A03B", "#4E91A8", "#398564"]
    c4 = original.value_counts().reindex(order4)
    c3 = y3.value_counts().reindex(CLASS_ORDER)
    for a, counts, title, colors in [
        (ax[0], c4, "(a) Original four-class labels", colors4),
        (ax[1], c3, "(b) Official three-class mapping", [CLASS_COLORS[c] for c in CLASS_ORDER]),
    ]:
        bars = a.bar(counts.index, counts.values, color=colors, width=0.62)
        a.set_title(title, pad=8)
        a.set_ylabel("Count")
        a.set_ylim(0, max(counts) * 1.18)
        for bar, v in zip(bars, counts):
            a.text(bar.get_x() + bar.get_width() / 2, v + max(counts) * 0.025, f"{v:,}", ha="center", va="bottom", fontsize=9)
        a.spines[["top", "right"]].set_visible(False)
    fig.suptitle("Task 3 label-scope audit", y=1.01, fontsize=13)
    fig.tight_layout()
    savefig("FigT3_01_label_scope_audit")


def plot_model_comparison(summary: pd.DataFrame) -> None:
    setup_style()
    fig, ax = plt.subplots(figsize=(9.2, 5.0))
    order = ["Logistic", "RandomForest", "HistGB", "ExtraTrees", "MLP", "Fusion"]
    d = summary.set_index("model").reindex(order).reset_index()
    x = np.arange(len(d)); width = 0.26
    for off, metric, hatch in [(-width, "ACC", ""), (0, "Macro_F1", "//"), (width, "Macro_AUC", "..")]:
        bars = ax.bar(x + off, d[metric], width, label=metric, color=[MODEL_COLORS[m] for m in d.model], alpha=0.92, hatch=hatch, edgecolor="white", linewidth=0.5)
        if metric == "ACC":
            for b, v in zip(bars, d[metric]):
                ax.text(b.get_x() + b.get_width()/2, v + .007, f"{v:.3f}", ha="center", va="bottom", fontsize=7.5, rotation=90)
    ax.set_xticks(x, d.model)
    ax.set_ylim(max(0.65, d[["ACC", "Macro_F1", "Macro_AUC"]].min().min() - 0.05), 1.015)
    ax.set_ylabel("Score")
    ax.set_title("NO_DIRECT five-fold OOF model comparison")
    ax.legend(ncol=3, frameon=False, loc="upper center", bbox_to_anchor=(0.5, -0.14))
    ax.spines[["top", "right"]].set_visible(False)
    fig.tight_layout(rect=(0, 0.06, 1, 1))
    savefig("FigT3_02_model_comparison")


def plot_confusion(y: np.ndarray, prob: np.ndarray) -> None:
    setup_style()
    pred = np.asarray(CLASS_ORDER)[prob.argmax(axis=1)]
    cm = confusion_matrix(y, pred, labels=CLASS_ORDER)
    norm = cm / cm.sum(axis=1, keepdims=True)
    labels = np.array([[f"{cm[i,j]:,}\n{norm[i,j]:.1%}" for j in range(3)] for i in range(3)])
    fig, ax = plt.subplots(figsize=(6.2, 5.1))
    sns.heatmap(norm, annot=labels, fmt="", cmap=sns.light_palette("#225B6A", as_cmap=True), vmin=0, vmax=1, cbar_kws={"label": "Row-normalized rate"}, linewidths=.7, linecolor="white", ax=ax)
    ax.set_xticklabels(CLASS_ORDER, rotation=0)
    ax.set_yticklabels(CLASS_ORDER, rotation=0)
    ax.set_xlabel("Predicted class"); ax.set_ylabel("True class")
    ax.set_title("NO_DIRECT fusion confusion matrix (OOF)")
    fig.tight_layout()
    savefig("FigT3_03_confusion_matrix")


def plot_roc_pr(y: np.ndarray, prob: np.ndarray) -> None:
    setup_style()
    yb = label_binarize(y, classes=CLASS_ORDER)
    fig, ax = plt.subplots(1, 2, figsize=(11.2, 4.6))
    for i, cls in enumerate(CLASS_ORDER):
        fpr, tpr, _ = roc_curve(yb[:, i], prob[:, i])
        precision, recall, _ = precision_recall_curve(yb[:, i], prob[:, i])
        auc = np.trapezoid(tpr, fpr)
        ap = average_precision_score(yb[:, i], prob[:, i])
        ax[0].plot(fpr, tpr, color=CLASS_COLORS[cls], label=f"{cls} (AUC={auc:.3f})")
        ax[1].plot(recall, precision, color=CLASS_COLORS[cls], label=f"{cls} (AP={ap:.3f})")
    ax[0].plot([0,1],[0,1],"--",color="#777777",lw=1)
    ax[0].set(xlabel="False positive rate", ylabel="True positive rate", title="(a) One-vs-Rest ROC")
    ax[1].set(xlabel="Recall", ylabel="Precision", title="(b) One-vs-Rest PR")
    for a in ax:
        a.legend(frameon=False, loc="lower right")
        a.spines[["top", "right"]].set_visible(False)
    fig.suptitle("NO_DIRECT fusion discrimination (five-fold OOF)", y=1.01, fontsize=13)
    fig.tight_layout()
    savefig("FigT3_04_roc_pr")


def plot_calibration(y: np.ndarray, prob: np.ndarray) -> None:
    setup_style()
    fig, ax = plt.subplots(figsize=(6.6, 5.0))
    for i, cls in enumerate(CLASS_ORDER):
        yy = (y == cls).astype(int)
        frac, mean = calibration_curve(yy, prob[:, i], n_bins=10, strategy="quantile")
        ax.plot(mean, frac, "o-", ms=4, color=CLASS_COLORS[cls], label=cls)
    ax.plot([0,1],[0,1],"--",color="#666666",lw=1,label="Ideal calibration")
    ax.set(xlabel="Mean predicted probability", ylabel="Observed frequency", title="NO_DIRECT fusion reliability (OOF)")
    ax.legend(frameon=False)
    ax.spines[["top", "right"]].set_visible(False)
    fig.tight_layout()
    savefig("FigT3_05_reliability")


def plot_feature_importance(model: Pipeline, x: pd.DataFrame, y: np.ndarray) -> pd.DataFrame:
    setup_style()
    p = permutation_importance(model, x, y, scoring="accuracy", n_repeats=5, random_state=SEED, n_jobs=-1)
    imp = pd.DataFrame({"feature": x.columns, "importance": p.importances_mean, "std": p.importances_std}).sort_values("importance", ascending=False)
    top = imp.head(15).sort_values("importance")
    fig, ax = plt.subplots(figsize=(8.2, 5.8))
    ax.barh(top.feature, top.importance, xerr=top["std"], color="#3B8F83", alpha=.9, ecolor="#666666", capsize=2)
    ax.set_xlabel("Decrease in holdout ACC after permutation")
    ax.set_title("NO_DIRECT ExtraTrees permutation importance")
    ax.spines[["top", "right"]].set_visible(False)
    fig.tight_layout()
    savefig("FigT3_06_permutation_importance")
    return imp


def plot_mlp_loss(model: Pipeline) -> None:
    setup_style()
    clf = model.named_steps["clf"]
    fig, ax = plt.subplots(figsize=(7.5, 4.4))
    ax.plot(np.arange(1, len(clf.loss_curve_) + 1), clf.loss_curve_, color=MODEL_COLORS["MLP"], label="Training loss")
    if hasattr(clf, "validation_scores_") and clf.validation_scores_:
        ax2 = ax.twinx()
        ax2.plot(np.arange(1, len(clf.validation_scores_) + 1), clf.validation_scores_, color="#225B6A", label="Internal validation ACC")
        ax2.set_ylabel("Internal validation ACC")
        lines = ax.get_lines()+ax2.get_lines()
        ax.legend(lines,[l.get_label() for l in lines],frameon=False,loc="center right")
    ax.set(xlabel="Iteration", ylabel="Cross-entropy loss", title="MLP training convergence")
    ax.spines[["top", "right"]].set_visible(False)
    fig.tight_layout()
    savefig("FigT3_07_mlp_training_curve")


def plot_ablation(scope_rows: pd.DataFrame) -> None:
    setup_style()
    fig, ax = plt.subplots(figsize=(7.4, 4.6))
    d = scope_rows.set_index("scope").reindex(["FULL", "NO_DIRECT", "NO_COMPOSITE"]).reset_index()
    bars = ax.bar(d.scope, d.ACC, color=["#777777", "#225B6A", "#BBBBBB"], width=.58)
    ax.set_ylim(max(.55, d.ACC.min()-.08), 1.02)
    ax.set_ylabel("Locked-holdout ACC")
    ax.set_title("Feature-scope ablation (official three classes)")
    for b,v in zip(bars,d.ACC):
        ax.text(b.get_x()+b.get_width()/2,v+.008,f"{v:.4f}",ha="center",va="bottom",fontsize=9)
    ax.spines[["top", "right"]].set_visible(False)
    fig.tight_layout()
    savefig("FigT3_08_scope_ablation")


def main() -> None:
    setup_style()
    raw = pd.read_csv(DATA)
    y3 = raw["Wellness_Category"].replace({"Poor": "Average"})
    audit = {
        "rows": len(raw),
        "columns": raw.shape[1],
        "missing": raw.isna().sum()[raw.isna().sum() > 0].to_dict(),
        "original_distribution": raw.Wellness_Category.value_counts().to_dict(),
        "official_3c_distribution": y3.value_counts().to_dict(),
        "fitness_exact_match_4c": float((raw.Fitness_Level == raw.Wellness_Category).mean()),
    }
    score4 = pd.cut(raw.Health_Score, [-np.inf,45,65,80,np.inf], labels=["Poor","Average","Good","Excellent"], right=False).astype(str)
    audit["health_score_rule_exact_match_4c"] = float((score4 == raw.Wellness_Category).mean())
    (TAB / "data_audit.json").write_text(json.dumps(audit, ensure_ascii=False, indent=2), encoding="utf-8")
    pd.DataFrame({"label": y3}).value_counts().rename("count").reset_index().to_csv(TAB / "official_3c_distribution.csv", index=False, encoding="utf-8-sig")
    plot_class_distribution(raw.Wellness_Category, y3)

    ids = np.arange(len(raw))
    dev_idx, hold_idx = train_test_split(ids, test_size=.2, random_state=SEED, stratify=y3)
    split = pd.DataFrame({"Person_ID": raw.Person_ID, "partition": np.where(np.isin(ids, dev_idx), "development", "locked_holdout"), "label_3c": y3})
    split.to_csv(TAB / "split_seed2026.csv", index=False, encoding="utf-8-sig")

    x = engineer(raw, "NO_DIRECT")
    xdev, xhold = x.iloc[dev_idx].reset_index(drop=True), x.iloc[hold_idx].reset_index(drop=True)
    ydev, yhold = y3.iloc[dev_idx].to_numpy(), y3.iloc[hold_idx].to_numpy()
    cv = StratifiedKFold(5, shuffle=True, random_state=SEED)
    folds = list(cv.split(xdev, ydev))

    names = ["Logistic", "RandomForest", "HistGB", "ExtraTrees", "MLP"]
    oof, fold_rows = {}, []
    for name in names:
        saved = PRED / f"oof_{name}.csv"
        if saved.exists():
            cached = pd.read_csv(saved)
            oof[name] = cached[[f"prob_{c}" for c in CLASS_ORDER]].to_numpy()
            print("RESUME", name, flush=True)
        else:
            oof[name], rows = oof_model(name, xdev, ydev, folds)
            fold_rows.extend(rows)
            pd.DataFrame({"Person_ID": raw.Person_ID.iloc[dev_idx].values, "true_label": ydev, **{f"prob_{c}":oof[name][:,i] for i,c in enumerate(CLASS_ORDER)}}).to_csv(saved, index=False, encoding="utf-8-sig")

    weights, fusion_oof, fusion_search = search_fusion(oof, ydev)
    oof["Fusion"] = fusion_oof
    fusion_search.to_csv(TAB / "fusion_weight_search.csv", index=False, encoding="utf-8-sig")
    (TAB / "fusion_weights.json").write_text(json.dumps(weights, indent=2), encoding="utf-8")

    summary = pd.DataFrame([{"model": n, **metrics(ydev, oof[n])} for n in names + ["Fusion"]])
    summary.to_csv(TAB / "oof_model_comparison.csv", index=False, encoding="utf-8-sig")
    pd.DataFrame(fold_rows).to_csv(TAB / "fold_metrics.csv", index=False, encoding="utf-8-sig")
    plot_model_comparison(summary)
    plot_confusion(ydev, fusion_oof)
    plot_roc_pr(ydev, fusion_oof)
    plot_calibration(ydev, fusion_oof)

    hold_probs, fitted = {}, {}
    for name in names:
        model = make_model(name, xdev)
        model.fit(xdev, ydev)
        fitted[name] = model
        hold_probs[name] = align_prob(model, model.predict_proba(xhold))
        joblib.dump(model, MOD / f"{name}_NO_DIRECT_3C.joblib")
    hold_fusion = sum(weights[n] * hold_probs[n] for n in weights)
    hold_summary = pd.DataFrame([{"model":n, **metrics(yhold, hold_probs[n])} for n in names] + [{"model":"Fusion", **metrics(yhold, hold_fusion)}])
    hold_summary.to_csv(TAB / "locked_holdout_model_comparison.csv", index=False, encoding="utf-8-sig")
    hold_pred = np.asarray(CLASS_ORDER)[hold_fusion.argmax(axis=1)]
    hold_out = pd.DataFrame({"Person_ID":raw.Person_ID.iloc[hold_idx].values, "true_label":yhold, "predicted_label":hold_pred, **{f"prob_{c}":hold_fusion[:,i] for i,c in enumerate(CLASS_ORDER)}})
    hold_out.to_csv(PRED / "task3_locked_holdout_predictions_3C.csv", index=False, encoding="utf-8-sig")
    report = pd.DataFrame(classification_report(yhold, hold_pred, labels=CLASS_ORDER, output_dict=True, zero_division=0)).T
    report.to_csv(TAB / "locked_holdout_classification_report.csv", encoding="utf-8-sig")
    cm = pd.DataFrame(confusion_matrix(yhold, hold_pred, labels=CLASS_ORDER), index=CLASS_ORDER, columns=CLASS_ORDER)
    cm.to_csv(TAB / "locked_holdout_confusion_matrix.csv", encoding="utf-8-sig")
    imp = plot_feature_importance(fitted["ExtraTrees"], xhold, yhold)
    imp.to_csv(TAB / "permutation_importance.csv", index=False, encoding="utf-8-sig")
    plot_mlp_loss(fitted["MLP"])

    scope_rows = []
    for scope in ["FULL", "NO_DIRECT", "NO_COMPOSITE"]:
        xs = engineer(raw, scope)
        xd, xh = xs.iloc[dev_idx].reset_index(drop=True), xs.iloc[hold_idx].reset_index(drop=True)
        model = make_model("ExtraTrees", xd)
        model.fit(xd, ydev)
        p = align_prob(model, model.predict_proba(xh))
        scope_rows.append({"scope":scope, **metrics(yhold,p)})
        joblib.dump(model, MOD / f"ExtraTrees_{scope}_3C.joblib")
    scope_table = pd.DataFrame(scope_rows)
    scope_table.to_csv(TAB / "feature_scope_ablation.csv", index=False, encoding="utf-8-sig")
    plot_ablation(scope_table)

    # Final models are refit on all 10,000 labelled rows after evaluation.
    final_manifest = {"label_mode":"official_3C_Poor_to_Average", "class_order":CLASS_ORDER, "seed":SEED, "fusion_weights":weights, "models":[]}
    for name in weights:
        model = make_model(name, x)
        model.fit(x, y3.to_numpy())
        path = MOD / f"FINAL_{name}_NO_DIRECT_3C_all10000.joblib"
        joblib.dump(model, path)
        final_manifest["models"].append(path.name)
    full_x = engineer(raw, "FULL")
    full_model = make_model("ExtraTrees", full_x)
    full_model.fit(full_x, y3.to_numpy())
    full_path = MOD / "FINAL_ExtraTrees_FULL_3C_all10000.joblib"
    joblib.dump(full_model, full_path)
    final_manifest["competition_model"] = full_path.name
    (MOD / "final_model_manifest.json").write_text(json.dumps(final_manifest, ensure_ascii=False, indent=2), encoding="utf-8")
    print("DONE")
    print(summary.to_string(index=False))
    print(hold_summary.to_string(index=False))
    print(scope_table.to_string(index=False))
    print(weights)


if __name__ == "__main__":
    main()
