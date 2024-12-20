import pandas as pd

def get_dataset(data_path: str) -> pd.DataFrame:
    """
    Получение данных по заданному пути
    :param dataset_path: путь до данных 
    :return: датасет
    """
    return pd.read_csv(data_path)