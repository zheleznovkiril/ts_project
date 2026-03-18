import random
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler, RobustScaler, QuantileTransformer
from config import SEED, DATASET_PATH, N_SERIES
from src.dataloader import convert_tsf_to_long_dataframe
from src.models import train_predict_evaluate, train_test_split, sample_series


def run_experiments():
    random.seed(SEED)
    np.random.seed(SEED)

    print(f'Загрузка данных из {DATASET_PATH}')
    df, frequency, forecast_horizon, contain_missing_values, contain_equal_length = convert_tsf_to_long_dataframe(DATASET_PATH)
    print(f"Загружен {DATASET_PATH}:")
    print("frequency:", frequency, "\nforecast_horizon:", forecast_horizon, "\ncontain_missing_values:", contain_missing_values, "\ncontain_equal_length:", contain_equal_length) 
    print(df.head())
    df = sample_series(df, N_SERIES, SEED)
    if N_SERIES != df['unique_id'].nunique():
        raise ValueError(f"Количество рядов не соответствует ожидаемому: {N_SERIES} != {df['unique_id'].nunique()}")
    print(f"Отобрано {df['unique_id'].nunique()} рядов")
    train_df, test_df = train_test_split(df, forecast_horizon)

    print("Обучение без нормализации...")
    all_forecasts_vanilla, scores_vanilla = train_predict_evaluate(train_df, test_df, forecast_horizon)
    print("Обучение с StandardScaler...")
    all_forecasts_standard, scores_standard = train_predict_evaluate(train_df, test_df, forecast_horizon, '_Standard', StandardScaler())
    print("Обучение с RobustScaler...")
    all_forecasts_robust, scores_robust = train_predict_evaluate(train_df, test_df, forecast_horizon, '_Robust', RobustScaler())
    print("Обучение с QuantileTransformer...")
    all_forecasts_quantile, scores_quantile = train_predict_evaluate(train_df, test_df, forecast_horizon, '_QuantileTransformer', QuantileTransformer(output_distribution='normal', random_state=SEED))

    print('Сохранение результатов')
    final_forecasts = all_forecasts_vanilla.merge(all_forecasts_standard, on=["unique_id", "ds"])
    final_forecasts = final_forecasts.merge(all_forecasts_robust, on=["unique_id", "ds"])
    final_forecasts = final_forecasts.merge(all_forecasts_quantile, on=["unique_id", "ds"])

    df.to_csv('results/dataset.csv', index=False)
    final_forecasts.to_csv('results/forecasts.csv', index=False)
    pd.concat([scores_vanilla, scores_standard, scores_robust, scores_quantile]).reset_index().rename(columns={'index': 'model'}).to_csv('results/metrics.csv', index=False)
    print("Эксперименты успешно отработали, результаты сохранены в results/")


if __name__ == '__main__':
    run_experiments()
