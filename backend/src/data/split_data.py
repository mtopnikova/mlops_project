import pandas as pd
from sklearn.model_selection import train_test_split
from typing import Tuple

def split_train_test(data: pd.DataFrame, **kwargs
    ) -> Tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series]:
    """
    Разделение данных на train/test
    :param dataset: датасет
    :return: train/test датасеты
    """
    X = data.drop(kwargs['target_column'], axis=1)
    y = data[kwargs['target_column']]

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=kwargs['test_size'],
        stratify=y,
        random_state=kwargs['random_state'])
    
    return X_train, X_test, y_train, y_test
