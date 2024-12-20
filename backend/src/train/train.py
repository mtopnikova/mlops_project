import pandas as pd
import numpy as np
from sklearn.svm import SVC
from sklearn.inspection import permutation_importance

from ..train.metrics import save_metrics


def train_model(
    X_train: pd.DataFrame,
    X_test: pd.DataFrame,
    y_train: pd.Series,
    y_test: pd.Series,
    best_params: dict,
    metric_path: str,
) -> SVC:
    """
    Обучение модели на лучших параметрах
    :param X_train: объект-признаки трейн
    :param X_test: объект-признаки холдаут
    :param y_train: ответы трейн
    :param y_test: ответы холдаут
    :param best_params: лучшие параметры, найденные GridSearchCV
    :param metric_path: путь до папки с метриками
    :return: SVC
    """
    # тренировка на лучших параметрах
    svc = SVC(**best_params)
    svc.fit(X_train, y_train)

    # сохранение метрик
    save_metrics(X_test=X_test, y_test=y_test, model=svc, metric_path=metric_path)
    return svc


def save_feature_importances(
    model: object, X_data: pd.DataFrame, y_data: pd.Series, **kwargs
) -> None:
    """
    Сохранение датафрейма с permutation importances
    :param model: обученная модель
    :param X_data: матрица объект-признаки
    :param y_data: целевая переменная
    """
    # вычисление и сортировка permutation importances
    perm = permutation_importance(
        model,
        X_data,
        y_data,
        random_state=kwargs["random_state"],
        n_repeats=kwargs["n_repeats"],
    )
    perm_df = pd.DataFrame(
        {"feature": X_data.columns, "value": perm["importances_mean"]}
    ).sort_values(by="value", ascending=False)

    # сохранение датафрейма с permutation importances
    perm_df.to_csv(kwargs["permutation_importances_path"], index=False)
