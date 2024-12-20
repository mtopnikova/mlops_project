import pandas as pd
import os
import joblib
import yaml
import json

from ..train.train import train_model, save_feature_importances
from ..data.get_data import get_dataset
from ..transform.transform import train_preprocess


def pipeline_training(config_path: str) -> None:
    """
    Полный цикл получения данных, предобработки и тренировки модели
    :param config_path: путь до файла с конфигурациями
    :return: None
    """
    # get params
    with open(config_path) as file:
        config = yaml.load(file, Loader=yaml.FullLoader)
    preprocessing_config = config["preprocessing"]
    train_config = config["train"]
    feature_imp_config = config["permutation_importances"]

    # get data
    train_data = get_dataset(data_path=preprocessing_config["raw_train_path"])

    # splitting data and preprocessing
    X_train, X_test, y_train, y_test = train_preprocess(data=train_data, **config)

    # подобранные в исследовательской части гиперпараметры
    best_params_path = train_config["params_path"]
    with open(best_params_path) as json_file:
        best_params = json.load(json_file)

    # тренировка с лучшими гиперпараметрами
    svc = train_model(
        X_train=X_train,
        X_test=X_test,
        y_train=y_train,
        y_test=y_test,
        best_params=best_params,
        metric_path=train_config["metrics_path"],
    )

    # сохранение feature importances
    save_feature_importances(svc, X_test, y_test, **feature_imp_config)

    # сохранение обученной модели
    joblib.dump(svc, os.path.join(train_config["model_path"]))
