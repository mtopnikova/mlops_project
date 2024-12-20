import os
import json
import requests
import streamlit as st

from ..plotting.plots import plot_feature_importances


def start_training(config: dict, endpoint: object) -> None:
    """
    Тренировка модели
    :param config: конфигурационный файл
    :param endpoint: endpoint
    """
    # Train
    with st.spinner("Модель подбирает параметры..."):
        requests.post(endpoint, timeout=5000)
    st.success("Success!")


def display_metrics(metrics_path: str) -> None:
    """
    Вывод метрик
    :param metrics_path: путь до json c метриками
    """
    if os.path.exists(metrics_path):
        with open(metrics_path) as json_file:
            metrics = json.load(json_file)
        roc_auc, precision, recall, f1_metric, logloss = st.columns(5)
        roc_auc.metric("ROC-AUC", metrics["roc_auc"])
        precision.metric("Precision", metrics["precision"])
        recall.metric("Recall", metrics["recall"])
        f1_metric.metric("F1 score", metrics["f1"])
        logloss.metric("Logloss", metrics["logloss"])
    else:
        st.write("Train model first")


def show_feature_importances(perm_path: str) -> None:
    """
    Вывод feature importances
    :param perm_path: путь до .csv с feature importances
    """
    if os.path.exists(perm_path):
        st.markdown("## Feature importances")
        st.pyplot(plot_feature_importances(perm_path))
        st.markdown(
            """
                Самыми значимыми для модели признаками оказались функциональная оценка Functional Assessment,
                оценка активности повседневной жизни ADL, наличие жалоб на память,
                оценка по краткой шкале оценки психического состояния MMSE,
                наличие поведенческих расстройств и возраст.
                Также достаточно важными признаками являются качество сна, этническая принадлежность,
                наличие депрессии, систолическое давление и наличие трудностей с выполнением задач.
                Остальные признаки оказались менее значимыми.
                """
        )
    else:
        st.write("Train model first")
