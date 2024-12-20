import pandas as pd


def get_data_selected_features(
    raw_data: pd.DataFrame, **kwargs
) -> None:  # path_proc: str
    """
    Получение датасета с отобранными признаками
    для вывода в Streamlit
    :param dataset_path: путь до данных
    :return: датасет
    """
    preproc = kwargs["preprocessing"]
    # удаление ненужных признаков
    data = raw_data.drop(preproc["drop_columns"], axis=1, errors="ignore")
    # замена числовых значений на понятные категории
    data.replace(preproc["map_change_columns"], inplace=True)

    return data
