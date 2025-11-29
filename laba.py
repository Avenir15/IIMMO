import pandas as pd
from sklearn.preprocessing import MinMaxScaler

# === 1. Загрузка данных ===
train_df = pd.read_csv(r"C:\Users\aveni\OneDrive\Desktop\UU\train.csv")

print("Исходные размеры:")
print(f"Train: {train_df.shape}")


# === 2. Функция обработки данных ===
def process_dataset(df, name="dataset"):
    print(f"\n=== Обработка {name} ===")
    print(f"Исходный размер: {df.shape}")

    df_processed = df.copy()

    # --- 2.1 Заполнение пропусков ---
    for col in df_processed.columns:
        if df_processed[col].isnull().any():
            if pd.api.types.is_numeric_dtype(df_processed[col]):
                df_processed[col] = df_processed[col].fillna(df_processed[col].median())
                print(f"Заполнены пропуски в числовой колонке: {col}")
            else:
                df_processed[col] = df_processed[col].fillna(df_processed[col].mode()[0])
                print(f"Заполнены пропуски в категориальной колонке: {col}")

    # --- 2.2 Нормализация числовых данных ---
    numeric_cols = df_processed.select_dtypes(include='number').columns.tolist()

    exclude_cols = ['PassengerId', 'Transported']
    numeric_cols_to_scale = [col for col in numeric_cols if col not in exclude_cols]

    if numeric_cols_to_scale:
        scaler = MinMaxScaler()
        df_processed[numeric_cols_to_scale] = scaler.fit_transform(df_processed[numeric_cols_to_scale])
        print(f"Нормализованы числовые колонки: {numeric_cols_to_scale}")

    # --- 2.3 Определение категориальных колонок ---
    categorical_cols = df_processed.select_dtypes(exclude=['number', 'bool']).columns.tolist()
    categorical_cols = [col for col in categorical_cols if col != 'PassengerId']

    # --- 2.4 Удаление колонок с >50 уникальными значениями ---
    cols_to_drop = []
    for col in categorical_cols:
        if df_processed[col].nunique() > 50:
            cols_to_drop.append(col)
            print(f"Удалена колонка {col} (уникальных значений: {df_processed[col].nunique()})")

    df_processed.drop(cols_to_drop, axis=1, inplace=True)

    # --- 2.5 OHE после удаления коллонок ---
    categorical_cols = df_processed.select_dtypes(exclude=['number', 'bool']).columns.tolist()
    categorical_cols = [col for col in categorical_cols if col != 'PassengerId']

    if categorical_cols:
        df_processed = pd.get_dummies(df_processed, columns=categorical_cols, drop_first=True)
        print(f"One-Hot Encoding применён к колонкам: {categorical_cols}")

    print(f"Финальный размер {name}: {df_processed.shape}")

    return df_processed


# === 3. Обработка train ===
train_processed = process_dataset(train_df, "train_df")


# === 4. Сохранение результата ===
save_path = r"C:\Users\aveni\OneDrive\Desktop\UU\lab1\processed_train.csv"
train_processed.to_csv(save_path, index=False)

print(f"\nОбработанный файл сохранён как: {save_path}")
