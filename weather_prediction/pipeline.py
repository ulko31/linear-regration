import numpy as np
import pandas as pd

from sklearn.model_selection import train_test_split

from model import GDRegressor

FEATURE_COLUMNS = ["temperature_2m", "relative_humidity_2m", "wind_speed_10m"]
TARGET_COLUMN = "apparent_temperature"


def mae_score(y_true, y_pred) -> float:
    return float(np.mean(np.abs(np.asarray(y_true) - np.asarray(y_pred))))


def rmse_score(y_true, y_pred) -> float:
    return float(np.sqrt(np.mean((np.asarray(y_true) - np.asarray(y_pred)) ** 2)))


def r2_score_custom(y_true, y_pred) -> float:
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)
    ss_res = np.sum((y_true - y_pred) ** 2)
    ss_tot = np.sum((y_true - y_true.mean()) ** 2)
    return float(1 - ss_res / ss_tot)



def train_model(data: pd.DataFrame):
    """В этой функции описывается пайплайн обучения вашей модели линейной регрессии.
    - reg – обученная модель
    - anomalies – самые неточные предсказания в тестовой выборке. Самые неточные – это те, в которых
    отклонение от реальных данных было не несколько MAE или RMSE. Обычно берется в 3-4 раза. Но может быть и выше.
    - scaler – функция стандартизации данных."""
    if data.empty:
        raise ValueError("Данных нет")

    prepared = data.copy()
    if "date" in prepared.columns:
        prepared["date"] = pd.to_datetime(prepared["date"], errors="coerce")

    required_columns = FEATURE_COLUMNS + [TARGET_COLUMN]
    prepared = prepared.dropna(subset=required_columns).reset_index(drop=True)
    if prepared.empty:
        raise ValueError("После очистки не осталось строк для обучения")

    X = prepared[FEATURE_COLUMNS]
    y = prepared[TARGET_COLUMN]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    train_mean = X_train.mean()
    train_std = X_train.std(ddof=0).replace(0, 1)

    def scaler(values):
        values = pd.DataFrame(values, columns=FEATURE_COLUMNS)
        return (values - train_mean) / train_std

    reg = GDRegressor(alpha=0.03, n_iter=5000, progress=False)
    reg.fit(X_train, y_train)

    predictions = reg.predict(X_test).ravel()
    y_test_array = y_test.to_numpy()

    mae = mae_score(y_test_array, predictions)
    rmse = rmse_score(y_test_array, predictions)
    r2 = r2_score_custom(y_test_array, predictions)

    anomalies = X_test.copy()
    anomalies[TARGET_COLUMN] = y_test_array
    anomalies["prediction"] = predictions
    anomalies["absolute_error"] = np.abs(y_test_array - predictions)
    anomalies = anomalies.sort_values("absolute_error", ascending=False).head(10)


    return reg, {"MAE": mae, "RMSE": rmse, "R2": r2}, anomalies, scaler
