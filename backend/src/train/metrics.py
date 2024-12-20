import pandas as pd
import numpy as np
from sklearn.metrics import (
    roc_auc_score,
    precision_score,
    recall_score,
    f1_score,
    log_loss,
)
import pandas as pd
import json


def create_dict_metrics(
    y_test: pd.Series, y_predict: np.ndarray, y_probability: np.ndarray
) -> dict:
    """
    Получение словаря с метриками для задачи классификации и запись в словарь
    :param y_test: реальные данные
    :param y_predict: предсказанные значения
    :param y_probability: предсказанные вероятности
    :return: словарь с метриками
    """
    dict_metrics = {
        "roc_auc": round(roc_auc_score(y_test, y_probability[:, 1]), 3),
        "precision": round(precision_score(y_test, y_predict), 3),
        "recall": round(recall_score(y_test, y_predict), 3),
        "f1": round(f1_score(y_test, y_predict), 3),
        "logloss": round(log_loss(y_test, y_probability), 3),
    }
    return dict_metrics


def save_metrics(
    X_test: pd.DataFrame, y_test: pd.Series, model: object, metric_path: str
) -> None:
    """
    Получение и сохранение метрик
    :param X_test: объект-признаки
    :param y_test: целевая переменная
    :param model: модель
    :param metric_path: путь для сохранения метрик
    """
    result_metrics = create_dict_metrics(
        y_test=y_test,
        y_predict=model.predict(X_test),
        y_probability=model.predict_proba(X_test),
    )
    with open(metric_path, "w") as file:
        json.dump(result_metrics, file)
