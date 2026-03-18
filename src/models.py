import pandas as pd
import numpy as np
from sklearn.metrics import mean_absolute_error
from sklearn.metrics import mean_squared_error
from statsforecast import StatsForecast
from statsforecast.models import Naive, SeasonalNaive, AutoETS, AutoTheta
from mlforecast import MLForecast
from catboost import CatBoostRegressor
from neuralforecast import NeuralForecast
from neuralforecast.models import PatchTST
from neuralforecast.losses.pytorch import SMAPE, MSE
from config import SEED


def smape(y_true, y_pred):
    """Расчет метрики SMAPE"""
    return np.mean(np.abs(y_true - y_pred) / (np.abs(y_true) + np.abs(y_pred)) / 2) * 100


def get_metrics(y_true, y_pred):
    """Расчет метрик"""
    return {
        "smape": smape(y_true, y_pred),
        "mae": mean_absolute_error(y_true, y_pred),
        "rmse": np.sqrt(mean_squared_error(y_true, y_pred))
    }


def evaluate_models(forecasts, test_df, models=("Naive", "SeasonalNaive", "AutoETS", "AutoTheta")):
    """Вычисление метрик для моделей"""
    results = {}
    # для каждой модели из полученного списка замеряю качество через get_metrics
    for model in models:
        merged = forecasts[["unique_id", "ds", model]].merge(test_df, on=["unique_id", "ds"])
        results[model] = get_metrics(merged["y"].values, merged[model].values)

    # транспонирую, чтобы модели были по строкам
    return pd.DataFrame(results).T


def train_predict_evaluate(train_df, test_df, h, suffix='', scaler=None):
    """Обучение моделей, прогноз и оценка качества"""
    train_df_scaled = train_df.copy()
    test_df_scaled = test_df.copy()

    # применяю scaker, если был передан
    if scaler is not None:
        train_df_scaled['y'] = scaler.fit_transform(train_df_scaled[['y']])
        test_df_scaled['y'] = scaler.transform(test_df_scaled[['y']])

    # naive, seasonalnaive, autoets,  autotheta
    sf = StatsForecast(
        models=[
                AutoETS(season_length=12),
                AutoTheta(season_length=12),
                Naive(),
                SeasonalNaive(season_length=12)
            ],
        freq="MS",
        n_jobs=-1
    )
    forecasts = sf.forecast(df=train_df_scaled, h=h)

    # catboost
    cat_model = CatBoostRegressor(
        iterations=300,
        depth=6,
        learning_rate=0.05,
        loss_function="RMSE",
        task_type="GPU",
        random_seed=SEED,
        verbose=False
    )

    mlf = MLForecast(models=[cat_model], freq="MS", lags=[1, 2, 3, 6, 12])
    mlf.fit(train_df_scaled)
    cat_forecasts = mlf.predict(h=h)

    # PatchTST
    patchtst_model = PatchTST(
        input_size=36,
        h=h,
        patch_len=6,
        stride=1,
        encoder_layers=2,
        hidden_size=64,
        n_heads=4,
        batch_size=64,
        max_steps=500,
        learning_rate=1e-3,
        loss=MSE(),
        random_seed=SEED,
        enable_progress_bar=True,
        accelerator='gpu',
        devices=1
    )

    nf = NeuralForecast(models=[patchtst_model], freq='MS')

    nf.fit(train_df_scaled)
    patch_forecasts = nf.predict(h=h)
    
    # объединяю прогнозы
    all_forecasts = forecasts.merge(cat_forecasts, on=["unique_id", "ds"])
    all_forecasts = all_forecasts.merge(patch_forecasts, on=["unique_id", "ds"])

    # обратное преобразование прогнозов, если передан scaler 
    if scaler is not None:
        for col in all_forecasts.columns:
            if col not in ['unique_id', 'ds']:
                # создаю временный датафрейм из колонки с предсказаниями
                temp_df = all_forecasts[[col]].rename(columns={col: 'y'})
                inverse_transformed = scaler.inverse_transform(temp_df)
                # беру первый столбец
                all_forecasts[col] = inverse_transformed[:, 0]

    # добавляю суффиксы для удобства
    if suffix:
        all_forecasts.columns = [f"{col}{suffix}" if col not in ['unique_id', 'ds'] else col 
                            for col in all_forecasts.columns]

    return all_forecasts, evaluate_models(forecasts=all_forecasts, test_df=test_df, 
                                          models=all_forecasts.drop(columns=["unique_id", "ds"]).columns.tolist())


def train_test_split(df, horizon):

    train_list = []
    test_list = []
    for series_id, ser in df.groupby("unique_id"):
        ser = ser.sort_values("ds")

        train_list.append(ser.iloc[:-horizon])
        test_list.append(ser.iloc[-horizon:])

    return pd.concat(train_list), pd.concat(test_list)


def sample_series(df, n_series=200, seed=42):
    rng = np.random.default_rng(seed)
    sampled_ids = rng.choice(df["unique_id"].unique(), size=n_series, replace=False)
    return df[df["unique_id"].isin(sampled_ids)].copy()
