import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sbn
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeRegressor, DecisionTreeClassifier, plot_tree
from sklearn.metrics import (
    mean_absolute_error, mean_squared_error, root_mean_squared_error,
    accuracy_score, precision_score, recall_score, f1_score,
    roc_curve, auc, confusion_matrix
)


# 1. Загрузка подготовленного датасета

df = pd.read_csv(r"C:\Users\aveni\OneDrive\Desktop\UU\Auto\processed_auto.csv")




print("\n================ РЕГРЕССИЯ (Мощность horsepower) ================")

y_reg = df["horsepower"]
X_reg = df.drop(["horsepower"], axis=1)

# Разделение
X_train_r, X_test_r, y_train_r, y_test_r = train_test_split(
    X_reg, y_reg, test_size=0.4, random_state=42
)

# Модель
regressor = DecisionTreeRegressor(max_depth=4, max_leaf_nodes=20)
regressor.fit(X_train_r, y_train_r)

# Предсказания
y_pred_r = regressor.predict(X_test_r)

# Метрики
print(f"Среднеквадратичная ошибка: {mean_squared_error(y_test_r, y_pred_r)}")
print(f"Корень среднеквадратичной ошибки: {root_mean_squared_error(y_test_r, y_pred_r)}")
print(f"Средняя абсолютная ошибка: {mean_absolute_error(y_test_r, y_pred_r)}")

# Дерево регрессии
plt.figure(figsize=(20, 10))
plot_tree(regressor, filled=True, feature_names=X_reg.columns)
plt.title("Дерево решений (РЕГРЕССИЯ прогноз horsepower, max_depth=4)")
plt.show()



# ========================== КЛАССИФИКАЦИЯ — Economy ==========================

print("\n================ КЛАССИФИКАЦИЯ (Экономичный автомобиль) ================")

# создаём бинарную колонку
df["Economy"] = (df["mpg"] > df["mpg"].mean()).astype(int)

y_clf = df["Economy"]
X_clf = df.drop(["Economy", "mpg"], axis=1)

# Разделение
X_train_c, X_test_c, y_train_c, y_test_c = train_test_split(
    X_clf, y_clf, test_size=0.4, random_state=42
)

# Модель классификации
clf = DecisionTreeClassifier(max_depth=4, max_leaf_nodes=20, random_state=42)
clf.fit(X_train_c, y_train_c)

# Предсказания
y_pred_c = clf.predict(X_test_c)

# Метрики
print("Accuracy:", accuracy_score(y_test_c, y_pred_c))  #доля верных предсказаний
print("Precision:", precision_score(y_test_c, y_pred_c))    #точность (из предсказанных экономичных — сколько действительно экономичные)
print("Recall:", recall_score(y_test_c, y_pred_c))      #полнота (сколько экономичных нашли)
print("F1 Score:", f1_score(y_test_c, y_pred_c))        #Насколько модель одновременно и точная, и полная? баланс precision + recall

# Матрица ошибок
cm = confusion_matrix(y_test_c, y_pred_c)
plt.figure(figsize=(4, 3))
sbn.heatmap(cm, annot=True, fmt="d", cmap="bwr")
plt.title("Confusion Matrix (Economy)")
plt.ylabel("True label")
plt.xlabel("Predicted label")
plt.show()



# ========================== ROC-кривая + AUC ==========================

y_proba = clf.predict_proba(X_test_c)
fpr, tpr, thresholds = roc_curve(y_test_c, y_proba[:, 1])
roc_auc = auc(fpr, tpr)

plt.figure()
plt.plot(fpr, tpr, marker="o")
plt.plot([0, 1], [0, 1], "--")
plt.xlabel("FPR")
plt.ylabel("TPR")
plt.title("ROC-кривая (Economy)")
plt.grid(True)
plt.show()

print("AUC:", roc_auc)



# ========================== Дерево классификации ==========================

plt.figure(figsize=(20, 10))
plot_tree(
    clf, filled=True,
    feature_names=X_clf.columns,
    class_names=["Не экономичный", "Экономичный"]
)
plt.title("Дерево решений (КЛАССИФИКАЦИЯ Economy, max_depth=4)")
plt.show()
