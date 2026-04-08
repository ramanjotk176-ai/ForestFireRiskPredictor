# member4_evaluation.py
import numpy as np
import joblib
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import (
    accuracy_score, f1_score, classification_report,
    confusion_matrix, roc_curve, auc
)
X_test = np.load("X_test.npy")
y_test = np.load("y_test.npy")
rf     = joblib.load("model_rf.pkl")
xgb    = joblib.load("model_xgb.pkl")
for name, model in [("Random Forest", rf), ("XGBoost", xgb)]:
    y_pred = model.predict(X_test)
    y_prob = model.predict_proba(X_test)[:, 1]
    print(f"\n===== {name} =====")
    print(f"Accuracy : {accuracy_score(y_test, y_pred):.4f}")
    print(f"F1 Score : {f1_score(y_test, y_pred):.4f}")
    print(classification_report(y_test, y_pred,
                                target_names=["No Fire","Fire"]))
    # Confusion matrix
    cm = confusion_matrix(y_test, y_pred)
    plt.figure(figsize=(5, 4))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Reds",
                xticklabels=["No Fire","Fire"],
                yticklabels=["No Fire","Fire"])
    plt.title(f"{name} - Confusion Matrix")
    plt.tight_layout()
    tag = name.replace(" ", "_").lower()
    plt.savefig(f"cm_{tag}.png")
    plt.show()
    # ROC curve
    fpr, tpr, _ = roc_curve(y_test, y_prob)
    plt.figure()
    plt.plot(fpr, tpr, color="tomato",
             label=f"AUC = {auc(fpr,tpr):.3f}")
    plt.plot([0,1],[0,1],"k--")
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title(f"{name} - ROC Curve")
    plt.legend()
    plt.tight_layout()
    plt.savefig(f"roc_{tag}.png")
    plt.show()
