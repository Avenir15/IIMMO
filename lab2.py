import pandas as pd
import numpy as np

import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.metrics import (
    r2_score,
    confusion_matrix,
    mean_squared_error,
    root_mean_squared_error,
    mean_absolute_error,
    accuracy_score
)

# ========================== ЗАГРУЗКА ДАННЫХ ==========================

df = pd.read_csv(r"C:\Users\aveni\OneDrive\Desktop\UU\Auto\processed_auto.csv")

# ========================== ЛИНЕЙНАЯ РЕГРЕССИЯ ==========================
# Предсказываем МОЩНОСТЬ (horsepower)

print("\n----- Линейная регрессия (Мощность) -----")

y_reg = df['horsepower']
X_reg = df.drop(['horsepower'], axis=1)

# Разделение на выборки
X_train1, X_test1, y_train1, y_test1 = train_test_split(
    X_reg, y_reg, test_size=0.4, random_state=42
)

# Обучение модели
reg_model = LinearRegression()
reg_model.fit(X_train1, y_train1)

# Предсказания
y_pred1 = reg_model.predict(X_test1)

# Метрики
r2 = r2_score(y_test1, y_pred1)

print(f"Среднеквадратичная ошибка: {mean_squared_error(y_test1, y_pred1)}")
print(f"Корень среднеквадратичной ошибки: {root_mean_squared_error(y_test1, y_pred1)}")
print(f"Средняя абсолютная ошибка: {mean_absolute_error(y_test1, y_pred1)}")
print(f"R^2: {r2}")
# ========================== КЛАССИФИКАЦИЯ: ЭКОНОМ / НЕ ЭКОНОМ ==========================

print("\n----- Логистическая регрессия (Экономичный автомобиль) -----")

# Экономичный = расход выше среднего
df['Economy'] = (df['mpg'] > df['mpg'].mean()).astype(int)

X_clf = df.drop(['mpg', 'Economy'], axis=1)
y_clf = df['Economy']

# Разделение
X_train, X_test, y_train, y_test = train_test_split(
    X_clf, y_clf, test_size=0.4, random_state=42
)

# Модель
clf_model = LogisticRegression(max_iter=1000)
clf_model.fit(X_train, y_train)

# Предсказание
y_pred = clf_model.predict(X_test)

# Метрики
accuracy = accuracy_score(y_test, y_pred)
cm = confusion_matrix(y_test, y_pred)

print(f"Точность классификации: {accuracy}")
print(f"Доля ошибок: {1 - accuracy}")
print(cm)

# Визуализация
plt.figure(figsize=(4, 3))
sns.heatmap(cm, annot=True, fmt='d', cmap='bwr')
plt.title('Confusion matrix (Economy)')
plt.ylabel('True label')
plt.xlabel('Predicted label')
plt.show()
