import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.metrics import (
    mean_squared_error,
    root_mean_squared_error,
    mean_absolute_error,
    accuracy_score
)

# Загружаем подготовленный файл
df = pd.read_csv(r"C:\Users\aveni\OneDrive\Desktop\UU\lab1\processed_train.csv")

# ========================== ЛИНЕЙНАЯ РЕГРЕССИЯ ==========================
# Предсказываем возраст

print("\n----- Линейная регрессия -----")

X_reg = df.drop(['Age', 'Survived'], axis=1)   # Убираем таргет и то, что мешает
y_reg = df['Age']

# Разделение на выборки
X_train, X_test, y_train, y_test = train_test_split(
    X_reg, y_reg, test_size=0.4, random_state=42
)

# Обучение модели
reg_model = LinearRegression()
reg_model.fit(X_train, y_train)

# Предсказания
y_pred = reg_model.predict(X_test)

# Метрики
print(f"Среднеквадратичная ошибка: {mean_squared_error(y_test, y_pred)}")
print(f"Корень среднеквадратичной ошибки: {root_mean_squared_error(y_test, y_pred)}")
print(f"Средняя абсолютная ошибка: {mean_absolute_error(y_test, y_pred)}")

# ========================== ЛОГИСТИЧЕСКАЯ РЕГРЕССИЯ ==========================
# Предсказываем, выжил пассажир или нет

print("\n----- Логистическая регрессия -----")

X_clf = df.drop(['Survived', 'Age'], axis=1)  # Убираем признаки, которые нельзя использовать
y_clf = df['Survived']

# Разделение
X_train, X_test, y_train, y_test = train_test_split(
    X_clf, y_clf, test_size=0.4, random_state=42
)

# Модель
clf_model = LogisticRegression(max_iter=500)
clf_model.fit(X_train, y_train)

# Предсказание
y_pred = clf_model.predict(X_test)

# Метрика
accuracy = accuracy_score(y_test, y_pred)
print(f"Точность классификации: {accuracy}")
print(f"Доля ошибок: {1 - accuracy}")
