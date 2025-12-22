# 1. Импорт библиотек и загрузка данных
import numpy as np
import random
import tensorflow as tf
from tensorflow import keras
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score
from sklearn.datasets import make_moons
import matplotlib.pyplot as plt

np.random.seed(42)


# --- Генерация тестовых данных (как указано в задании) ---
X = np.random.randint(0, 2, size=(100, 12))  # 100 примеров, 12 бинарных признаков
Y = np.array([[1, 0] if x.sum() > 6 else [0, 1] for x in X])

# Сохранение данных
np.savetxt('dataIn.txt', X, fmt='%d')
np.savetxt('dataOut.txt', Y, fmt='%d')

# Разделение
X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=0.4, random_state=42)

# Нормализация данных
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# 2. Создание нейросети с logsig (sigmoid)
model = keras.Sequential([
    keras.layers.Dense(12, activation='sigmoid', input_shape=(12,)),  # logsig
    keras.layers.Dense(2, activation='softmax')                       # выход 2 класса
])

# Компиляция модели
model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

# Обучение модели
history = model.fit(X_train, y_train, epochs=45, batch_size=17,
                    validation_data=(X_test, y_test))

# 3. Оценка модели
y_pred = np.argmax(model.predict(X_test), axis=1)
y_true = np.argmax(y_test, axis=1)
print("Accuracy:", accuracy_score(y_true, y_pred))

# График функции ошибки
plt.plot(history.history['loss'], label='Training Loss')
plt.plot(history.history['val_loss'], label='Validation Loss')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.legend()
plt.show()

# 4. Сравнение с классическими алгоритмами
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression

# Для классических моделей нужно преобразовать one-hot → label
y_train_labels = np.argmax(y_train, axis=1)
y_test_labels = np.argmax(y_test, axis=1)

# Логистическая регрессия
lr = LogisticRegression()
lr.fit(X_train, y_train_labels)
y_pred_lr = lr.predict(X_test)
print("Logistic Regression Accuracy:", accuracy_score(y_test_labels, y_pred_lr))

# Случайный лес
rf = RandomForestClassifier(n_estimators=100)
rf.fit(X_train, y_train_labels)
y_pred_rf = rf.predict(X_test)
print("Random Forest Accuracy:", accuracy_score(y_test_labels, y_pred_rf))
