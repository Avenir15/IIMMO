import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier, GradientBoostingClassifier
from sklearn.metrics import accuracy_score, f1_score, roc_curve, auc

# ==================== Загрузка данных ====================

df = pd.read_csv(r"C:\Users\aveni\OneDrive\Desktop\UU\Auto\processed_auto.csv")

# бинарная цель
df["Economy"] = (df["mpg"] > df["mpg"].mean()).astype(int)

X = df.drop(["Economy", "mpg"], axis=1)
y = df["Economy"]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.4, random_state=42
)

# ==================== RANDOM FOREST + OOB ====================

rf = RandomForestClassifier(
    n_estimators=120,
    max_depth=8,
    oob_score=True,
    random_state=42
)

rf.fit(X_train, y_train)

print("\n=== Random Forest ===")
print("OOB score:", rf.oob_score_)

y_pred_rf = rf.predict(X_test)
y_proba_rf = rf.predict_proba(X_test)[:, 1]

print("Accuracy:", accuracy_score(y_test, y_pred_rf))
print("F1:", f1_score(y_test, y_pred_rf))

# ==================== ADABOOST ====================

ada = AdaBoostClassifier(
    n_estimators=200,
    learning_rate=0.6,
    random_state=42
)

ada.fit(X_train, y_train)

y_pred_ada = ada.predict(X_test)
y_proba_ada = ada.predict_proba(X_test)[:, 1]

print("\n=== AdaBoost ===")
print("Accuracy:", accuracy_score(y_test, y_pred_ada))
print("F1:", f1_score(y_test, y_pred_ada))

# ==================== GRADIENT BOOSTING ====================

gb = GradientBoostingClassifier(
    n_estimators=150,
    learning_rate=0.1,
    max_depth=5,
    random_state=42
)

gb.fit(X_train, y_train)

y_pred_gb = gb.predict(X_test)
y_proba_gb = gb.predict_proba(X_test)[:, 1]

print("\n=== Gradient Boosting ===")
print("Accuracy:", accuracy_score(y_test, y_pred_gb))
print("F1:", f1_score(y_test, y_pred_gb))

# ==================== ROC-КРИВЫЕ ====================

plt.figure(figsize=(7, 6))

for name, proba in [
    ("Random Forest", y_proba_rf),
    ("AdaBoost", y_proba_ada),
    ("Gradient Boosting", y_proba_gb)
]:
    fpr, tpr, _ = roc_curve(y_test, proba)
    plt.plot(fpr, tpr, label=f"{name} (AUC={auc(fpr,tpr):.2f})")

plt.plot([0, 1], [0, 1], "--", color="gray")
plt.xlabel("FPR")
plt.ylabel("TPR")
plt.title("ROC-кривые (Economy)")
plt.legend()
plt.grid(True)
plt.show()