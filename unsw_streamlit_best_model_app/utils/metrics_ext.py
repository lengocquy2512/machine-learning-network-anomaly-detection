import numpy as np
from sklearn import metrics


def ids_metrics(y_true, y_pred, score=None):
    y_true = np.asarray(y_true).astype(int)
    y_pred = np.asarray(y_pred).astype(int)

    cm = metrics.confusion_matrix(y_true, y_pred, labels=[0, 1])
    TN, FP, FN, TP = cm.ravel()

    dr = TP / (TP + FN) if (TP + FN) else 0.0                 # Recall_Attack
    r0 = TN / (TN + FP) if (TN + FP) else 0.0                 # Recall_Normal (Specificity)
    fpr = FP / (FP + TN) if (FP + TN) else 0.0
    far = (FP + FN) / (TN + FP + FN + TP) if (TN + FP + FN + TP) else 0.0

    prec_a = TP / (TP + FP) if (TP + FP) else 0.0
    prec_n = TN / (TN + FN) if (TN + FN) else 0.0

    f1_a = metrics.f1_score(y_true, y_pred, pos_label=1, zero_division=0)

    try:
        auc = metrics.roc_auc_score(y_true, score) if score is not None else float("nan")
    except Exception:
        auc = float("nan")

    return {
        "TN": int(TN), "FP": int(FP), "FN": int(FN), "TP": int(TP),
        "DR(Recall_Attack)": float(dr),
        "R0(Recall_Normal)": float(r0),
        "FPR": float(fpr),
        "FAR": float(far),
        "Prec_Attack": float(prec_a),
        "Prec_Normal": float(prec_n),
        "F1": float(f1_a),
        "AUC": float(auc),
        "confusion_matrix": cm,
        "classification_report": metrics.classification_report(
            y_true, y_pred, target_names=["normal(0)", "attack(1)"], digits=4, zero_division=0
        ),
    }
