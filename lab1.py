import pandas as pd
from sklearn.preprocessing import MinMaxScaler

# === 1. Загрузка данных ===
auto_df = pd.read_csv(r"C:\Users\aveni\OneDrive\Desktop\UU\Auto\auto-mpg.csv")

print("Исходные размеры:")
print(f"Auto MPG: {auto_df.shape}")
print("=== ПЕРВЫЕ СТРОКИ ===")
print(auto_df.head())
print(auto_df.info())

# === 2. Функция обработки данных ===
def process_dataset(df, name="dataset"):
    print(f"\n=== Обработка {name} ===")
    df_processed = df.copy()

    # --- 2.1 Удаление колонки car name ---
    if 'car name' in df_processed.columns:
        df_processed.drop('car name', axis=1, inplace=True)
        print("Удалена колонка: car name")

    # --- 2.2 Обработка horsepower ---
    df_processed['horsepower'] = pd.to_numeric(df_processed['horsepower'], errors='coerce')

    # --- 2.3 Заполнение пропусков ---
    for col in df_processed.columns:
        if df_processed[col].isnull().any():
            if pd.api.types.is_numeric_dtype(df_processed[col]):
                df_processed[col] = df_processed[col].fillna(df_processed[col].median())
                print(f"Заполнены пропуски в числовой колонке: {col}")
            else:
                df_processed[col] = df_processed[col].fillna(df_processed[col].mode()[0])
                print(f"Заполнены пропуски в категориальной колонке: {col}")

    print("\n=== ПРОПУСКИ ПОСЛЕ ОБРАБОТКИ ===")
    print(df_processed.isnull().sum())

    # --- 2.4 Нормализация числовых данных ---
    numeric_cols_to_scale = ['mpg', 'displacement', 'horsepower', 'weight', 'model year']
    scaler = MinMaxScaler()
    df_processed[numeric_cols_to_scale] = scaler.fit_transform(df_processed[numeric_cols_to_scale])
    print(f"Нормализованы числовые колонки: {numeric_cols_to_scale}")

    # --- 2.5 Дискретизация acceleration ---
    df_processed['acceleration_cat'] = pd.qcut(
        df_processed['acceleration'],
        q=3,
        labels=['slow', 'medium', 'fast']
    )

    # --- 2.6 Преобразование origin в текстовые категории ---
    origin_map = {1: 'USA', 2: 'Europe', 3: 'Japan'}
    df_processed['origin'] = df_processed['origin'].map(origin_map)

    # --- 2.7 One-Hot Encoding для категориальных признаков ---
    categorical_cols = ['acceleration_cat', 'origin']
    df_processed = pd.get_dummies(df_processed, columns=categorical_cols, drop_first=False)
    print(f"One-Hot Encoding применён к колонкам: {categorical_cols}")

    print(f"Финальный размер {name}: {df_processed.shape}")
    print(df_processed.info())
    print(df_processed.head())

    return df_processed

# === 3. Обработка данных ===
auto_processed = process_dataset(auto_df, "auto_df")

# === 4. Сохранение результата ===
save_path = r"C:\Users\aveni\OneDrive\Desktop\UU\Auto\processed_auto.csv"
auto_processed.to_csv(save_path, index=False)

print(f"\nОбработанный файл сохранён как: {save_path}")
