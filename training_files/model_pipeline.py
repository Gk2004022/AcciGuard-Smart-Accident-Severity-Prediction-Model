import os
import time
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.utils.class_weight import compute_class_weight, compute_sample_weight
from sklearn.metrics import (
    classification_report, confusion_matrix, accuracy_score,
    f1_score, precision_score, recall_score, roc_auc_score, precision_recall_curve, auc
)
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from lightgbm import LGBMClassifier
from xgboost import XGBClassifier

sns.set_theme(style="whitegrid", context="talk")
plt.rcParams.update({'savefig.dpi': 300, 'savefig.bbox': 'tight'})

SEVERITY_LABELS = {0: 'Fatal', 1: 'Serious', 2: 'Slight'}
SEVERITY_PALETTE = {0: '#d9534f', 1: '#f0ad4e', 2: '#5cb85c'}

def check_dir(path):
    """Ensures directory exists."""
    os.makedirs(path, exist_ok=True)

def optimize_memory(df):
    for col in df.columns:
        if df[col].dtype == 'float64':
            df[col] = df[col].astype('float32')
        elif df[col].dtype == 'int64':
            df[col] = df[col].astype('int32')
    return df

def preprocess_and_encode_data(df):
    df_clean = df.copy()
    
    if 'accident_severity' in df_clean.columns:
        df_clean['target'] = df_clean['accident_severity'] - 1
    elif 'accident_seriousness' in df_clean.columns:
        df_clean['target'] = df_clean['accident_seriousness'].map({'Fatal': 0, 'Serious': 1, 'Not Serious': 2})
    else:
        raise ValueError("Could not find accident_severity or accident_seriousness target in DataFrame.")
        
    y = df_clean['target'].astype(int)
    
    cols_to_drop = [
        'accident_severity', 'accident_seriousness', 'target', 
        'accident_datetime', 'season', 'accident_year'
    ]
    X = df_clean.drop(columns=[col for col in cols_to_drop if col in df_clean.columns])

    categorical_cols = [
        'number_of_vehicles', 'number_of_casualties', 'day_of_week', 
        'first_road_class', 'road_type', 'junction_detail', 'junction_control', 
        'second_road_class', 'pedestrian_crossing_human_control', 
        'pedestrian_crossing_physical_facilities', 'light_conditions', 
        'weather_conditions', 'road_surface_conditions', 'special_conditions_at_site', 
        'carriageway_hazards', 'urban_or_rural_area', 'did_police_officer_attend_scene_of_accident', 
        'trunk_road_flag', 'is_weekend', 'time_of_day'
    ]

    categorical_cols = [col for col in categorical_cols if col in X.columns]

    numeric_cols = ['longitude', 'latitude', 'speed_limit', 'accident_hour']
    numeric_cols = [col for col in numeric_cols if col in X.columns]

    for col in categorical_cols:
        X[col] = X[col].astype(str)

    for col in numeric_cols:
        if X[col].isnull().sum() > 0:
            X[col] = X[col].fillna(X[col].median())

    scaler = StandardScaler()
    X[numeric_cols] = scaler.fit_transform(X[numeric_cols])

    X_encoded = pd.get_dummies(X, columns=categorical_cols, drop_first=True)

    X_encoded = optimize_memory(X_encoded)
    
    return X_encoded, y

def get_train_test_splits(X, y, test_size=0.2, sample_fraction=1.0, random_state=42):
    if sample_fraction < 1.0:
        X_sample, _, y_sample, _ = train_test_split(
            X, y, train_size=sample_fraction, stratify=y, random_state=random_state
        )
    else:
        X_sample, y_sample = X, y
        
    X_train, X_test, y_train, y_test = train_test_split(
        X_sample, y_sample, test_size=test_size, stratify=y_sample, random_state=random_state
    )
    
    return X_train, X_test, y_train, y_test

def train_and_evaluate_model(model_name, model, X_train, y_train, X_test, y_test, use_sample_weight=False):
    print(f"\nTraining {model_name}...")
    start_time = time.time()
    
    if use_sample_weight:
        sample_weights = compute_sample_weight('balanced', y_train)
        model.fit(X_train, y_train, sample_weight=sample_weights)
    else:
        model.fit(X_train, y_train)
        
    train_time = time.time() - start_time
    print(f"{model_name} Training Completed in {train_time:.2f} seconds.")

    y_pred = model.predict(X_test)

    if hasattr(model, "predict_proba"):
        y_proba = model.predict_proba(X_test)
    else:
        dfunc = model.decision_function(X_test)
        y_proba = np.exp(dfunc) / np.sum(np.exp(dfunc), axis=1, keepdims=True)

    accuracy = accuracy_score(y_test, y_pred)
    precision_macro = precision_score(y_test, y_pred, average='macro', zero_division=0)
    recall_macro = recall_score(y_test, y_pred, average='macro', zero_division=0)
    f1_macro = f1_score(y_test, y_pred, average='macro', zero_division=0)

    try:
        roc_auc_macro = roc_auc_score(y_test, y_proba, average='macro', multi_class='ovr')
    except Exception:
        roc_auc_macro = np.nan

    class_report = classification_report(y_test, y_pred, output_dict=True, zero_division=0)
    
    metrics = {
        "model_name": model_name,
        "accuracy": accuracy,
        "precision_macro": precision_macro,
        "recall_macro": recall_macro,
        "f1_macro": f1_macro,
        "roc_auc_macro": roc_auc_macro,
        "train_time_sec": train_time,
        "class_report": class_report,
        "y_pred": y_pred,
        "y_proba": y_proba
    }
    
    return model, metrics

def plot_confusion_matrix(y_true, y_pred, model_name, output_dir='reports/figures'):
    check_dir(output_dir)
    cm = confusion_matrix(y_true, y_pred)

    cm_percent = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis] * 100
    
    plt.figure(figsize=(9, 8))

    labels = np.asarray([
        f"{count:,}\n({pct:.1f}%)" 
        for count, pct in zip(cm.flatten(), cm_percent.flatten())
    ]).reshape(3, 3)

    sns.heatmap(
        cm, 
        annot=labels, 
        fmt="", 
        cmap="Blues", 
        cbar=True,
        xticklabels=[SEVERITY_LABELS[0], SEVERITY_LABELS[1], SEVERITY_LABELS[2]],
        yticklabels=[SEVERITY_LABELS[0], SEVERITY_LABELS[1], SEVERITY_LABELS[2]],
        annot_kws={"size": 12, "weight": "bold"},
        linewidths=0.5,
        edgecolor='black'
    )
    
    plt.title(f"Confusion Matrix: {model_name}", fontsize=18, fontweight='bold', pad=15)
    plt.ylabel("Actual Severity Class", fontsize=14, labelpad=10)
    plt.xlabel("Predicted Severity Class", fontsize=14, labelpad=10)
    plt.tight_layout()
    
    safe_name = model_name.lower().replace(" ", "_").replace("(", "").replace(")", "")
    save_path = os.path.join(output_dir, f'confusion_matrix_{safe_name}.png')
    plt.savefig(save_path)
    plt.close()
    return save_path

def plot_multiclass_roc_pr_curves(y_test, y_proba, model_name, output_dir='reports/figures'):
    check_dir(output_dir)
    
    from sklearn.preprocessing import label_binarize
    y_test_bin = label_binarize(y_test, classes=[0, 1, 2])
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(18, 8))
    
    from sklearn.metrics import roc_curve
    for i in range(3):
        fpr, tpr, _ = roc_curve(y_test_bin[:, i], y_proba[:, i])
        roc_auc = auc(fpr, tpr)
        ax1.plot(
            fpr, tpr, 
            color=SEVERITY_PALETTE[i], 
            lw=2.5, 
            label=f"{SEVERITY_LABELS[i]} vs All (AUC = {roc_auc:.3f})"
        )
        
    ax1.plot([0, 1], [0, 1], 'k--', lw=1.5)
    ax1.set_xlim([0.0, 1.0])
    ax1.set_ylim([0.0, 1.05])
    ax1.set_xlabel('False Positive Rate (1 - Specificity)', fontsize=13, labelpad=10)
    ax1.set_ylabel('True Positive Rate (Recall / Sensitivity)', fontsize=13, labelpad=10)
    ax1.set_title('Receiver Operating Characteristic (ROC) Curves', fontsize=15, fontweight='bold', pad=15)
    ax1.legend(loc="lower right", fontsize=11)
    
    for i in range(3):
        precision, recall, _ = precision_recall_curve(y_test_bin[:, i], y_proba[:, i])
        pr_auc = auc(recall, precision)
        ax2.plot(
            recall, precision, 
            color=SEVERITY_PALETTE[i], 
            lw=2.5, 
            label=f"{SEVERITY_LABELS[i]} vs All (AUC = {pr_auc:.3f})"
        )

    for i in range(3):
        baseline = y_test_bin[:, i].sum() / len(y_test_bin)
        ax2.axhline(
            y=baseline, 
            color=SEVERITY_PALETTE[i], 
            linestyle=':', 
            alpha=0.6, 
            label=f"Baseline {SEVERITY_LABELS[i]} ({baseline*100:.1f}%)"
        )
        
    ax2.set_xlim([0.0, 1.0])
    ax2.set_ylim([0.0, 1.05])
    ax2.set_xlabel('Recall (Sensitivity)', fontsize=13, labelpad=10)
    ax2.set_ylabel('Precision (Positive Predictive Value)', fontsize=13, labelpad=10)
    ax2.set_title('Precision-Recall (PR) Curves', fontsize=15, fontweight='bold', pad=15)
    ax2.legend(loc="upper right", fontsize=10)
    
    plt.suptitle(f"Model Performance Curves: {model_name}", fontsize=20, fontweight='bold', y=0.98)
    plt.tight_layout()
    
    safe_name = model_name.lower().replace(" ", "_").replace("(", "").replace(")", "")
    save_path = os.path.join(output_dir, f'performance_curves_{safe_name}.png')
    plt.savefig(save_path)
    plt.close()
    return save_path

def plot_feature_importance(model, feature_names, model_name, output_dir='reports/figures', top_n=20):
    check_dir(output_dir)
    
    importance = None
    
    if hasattr(model, "feature_importances_"):
        importance = model.feature_importances_
    elif hasattr(model, "booster_"):
        importance = model.booster_.feature_importance(importance_type='gain')
    elif hasattr(model, "coef_"):
        importance = np.mean(np.abs(model.coef_), axis=0)
        
    if importance is None:
        print(f"Could not extract feature importances for {model_name}.")
        return None
        
    feat_imp = pd.DataFrame({
        'Feature': feature_names,
        'Importance': importance
    }).sort_values(by='Importance', ascending=False)

    feat_imp_top = feat_imp.head(top_n)
    
    plt.figure(figsize=(12, 8))
    sns.barplot(
        x='Importance', 
        y='Feature', 
        data=feat_imp_top, 
        palette="plasma_r",
        edgecolor='black',
        linewidth=0.8
    )
    
    plt.title(f"Top {top_n} Most Predictive Features - {model_name}", fontsize=16, fontweight='bold', pad=15)
    plt.xlabel("Importance Score (Gain/Magnitude)", fontsize=13)
    plt.ylabel("Dataset Feature", fontsize=13)
    plt.tight_layout()
    
    safe_name = model_name.lower().replace(" ", "_").replace("(", "").replace(")", "")
    save_path = os.path.join(output_dir, f'feature_importance_{safe_name}.png')
    plt.savefig(save_path)
    plt.close()
    return save_path

def build_comparison_leaderboard(all_metrics):
    comparison_data = []
    
    for metric in all_metrics:
        name = metric["model_name"]
        cr = metric["class_report"]
        
        f1_fatal = cr.get("0", cr.get(0, {})).get("f1-score", 0.0)
        f1_serious = cr.get("1", cr.get(1, {})).get("f1-score", 0.0)
        f1_slight = cr.get("2", cr.get(2, {})).get("f1-score", 0.0)

        rec_fatal = cr.get("0", cr.get(0, {})).get("recall", 0.0)
        rec_serious = cr.get("1", cr.get(1, {})).get("recall", 0.0)
        rec_slight = cr.get("2", cr.get(2, {})).get("recall", 0.0)
        
        comparison_data.append({
            "Classifier Model": name,
            "Accuracy": f"{metric['accuracy']*100:.2f}%",
            "Macro F1-Score": f"{metric['f1_macro']*100:.2f}%",
            "Macro Recall": f"{metric['recall_macro']*100:.2f}%",
            "Fatal F1 (Class 1)": f"{f1_fatal*100:.2f}%",
            "Serious F1 (Class 2)": f"{f1_serious*100:.2f}%",
            "Slight F1 (Class 3)": f"{f1_slight*100:.2f}%",
            "Fatal Recall (Class 1)": f"{rec_fatal*100:.2f}%",
            "Serious Recall (Class 2)": f"{rec_serious*100:.2f}%",
            "ROC-AUC (Macro)": f"{metric['roc_auc_macro']*100:.2f}%" if not np.isnan(metric['roc_auc_macro']) else "N/A",
            "Training Time": f"{metric['train_time_sec']:.2f}s"
        })
        
    return pd.DataFrame(comparison_data).sort_values(by="Macro F1-Score", ascending=False)
