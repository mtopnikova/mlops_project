import pandas as pd
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.compose import ColumnTransformer
import json
import joblib
from typing import Tuple
import warnings

warnings.filterwarnings("ignore")
from ..data.split_data import split_train_test


def save_unique_train_data(
    data: pd.DataFrame,
    drop_columns: list,
    map_change_columns: dict,
    target_column: str,
    unique_values_path: str,
) -> None:
    """
    Сохранение словаря с признаками и уникальными значениями
    :param drop_columns: список с признаками для удаления
    :param map_change_columns: список с признаками для замены значений
    :param data: датасет
    :param target_column: целевая переменная
    :param unique_values_path: путь до файла со словарем
    :return: None
    """
    df = data.drop(columns=drop_columns + [target_column], axis=1)
    df.replace(map_change_columns, inplace=True)
    # создаем словарь с уникальными значениями

    dict_unique = {key: df[key].unique().tolist() for key in df.columns}

    with open(unique_values_path, "w") as file:
        json.dump(dict_unique, file)


def check_columns_evaluate(data: pd.DataFrame, unique_values_path: str) -> pd.DataFrame:
    """
    Проверка на наличие признаков из train и упорядочивание признаков согласно train
    :param data: датасет test
    :param unique_values_path: путь до списока с признаками train для сравнения
    :return: датасет test
    """
    with open(unique_values_path) as json_file:
        unique_values = json.load(json_file)

    column_sequence = unique_values.keys()

    assert set(column_sequence) == set(data.columns), "Разные признаки"
    return data[column_sequence]


def transform_columns(
    data: pd.DataFrame, flg_fit: bool = False, **kwargs
) -> pd.DataFrame:
    """
    Преобразование колонок: масштабирование для числовых признаков,
    one-hot-encoding для категориальных, либо отсутствие преобразований
    для уже бинаризованных признаков
    :param data: датасет
    :param flg_fit: флаг для тренировочных данных
    :return: датасет
    """
    preproc = kwargs["preprocessing"]
    train = kwargs["train"]

    if flg_fit:
        transformers_list = [
            (
                "encode",
                OneHotEncoder(dtype="int", drop="first"),
                preproc["one_hot_columns"],
            ),
            ("scale", StandardScaler(), preproc["scale_columns"]),
            ("skip", "passthrough", preproc["passthrough_columns"]),
        ]
        column_transformer = ColumnTransformer(
            transformers_list, verbose_feature_names_out=False
        )

        transformed_raw = column_transformer.fit_transform(data)

        # сохраняем обученный column transformer
        joblib.dump(column_transformer, train["col_transform_path"])

    else:
        column_transformer = joblib.load(train["col_transform_path"])
        transformed_raw = column_transformer.transform(data)

    data_transformed = pd.DataFrame(
        transformed_raw, columns=column_transformer.get_feature_names_out()
    )

    return data_transformed


def train_preprocess(
    data: pd.DataFrame, **kwargs
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series]:
    """
    Пайплайн по предобработке тренировочных данных c разбиением
    на трейн и холдаут,
    так как есть преобразования с fit transform
    :param data: датасет
    :return: датасет
    """
    preproc = kwargs["preprocessing"]
    # сохранение уникальных данных с признаками из train
    save_unique_train_data(
        data=data,
        drop_columns=preproc["drop_columns"],
        map_change_columns=preproc["map_change_columns"],
        target_column=preproc["target_column"],
        unique_values_path=preproc["unique_values_path"],
    )
    # удаление ненужных признаков
    data = data.drop(preproc["drop_columns"], axis=1, errors="ignore")

    # замена значений
    data.replace(preproc["map_change_columns"], inplace=True)

    # разделение на тренировочный и тестовый датасеты
    X_train, X_test, y_train, y_test = split_train_test(data, **preproc)

    # трансформация колонок(масштабирование, one-hot-encoding)
    X_train_transformed = transform_columns(X_train, flg_fit=True, **kwargs)
    X_test_transformed = transform_columns(X_test, **kwargs)

    return X_train_transformed, X_test_transformed, y_train, y_test


def test_preprocess(test_data: pd.DataFrame, **kwargs) -> pd.DataFrame:
    """
    Пайплайн по предобработке тестовых данных
    :param data: исходный датасет
    :return: предобработанный датасет
    """
    preproc = kwargs["preprocessing"]

    # удаление ненужных признаков
    test_data = test_data.drop(preproc["drop_columns"], axis=1, errors="ignore")

    # замена значений для one-hot-encoding:
    # (не нужно при ручном вводе данных)
    for col in preproc["one_hot_columns"]:
        if test_data[col].dtype != "object":
            test_data.replace(preproc["map_change_columns"], inplace=True)

    # проверка dataset на совпадение с признаками из train
    # и упорядочивание признаков согласно train
    test_data = check_columns_evaluate(
        data=test_data, unique_values_path=preproc["unique_values_path"]
    )

    # трансформация колонок(масштабирование и one-hot encoding)
    test_data_transformed = transform_columns(test_data, **kwargs)

    return test_data_transformed
