import os
import yaml
import streamlit as st
import pandas as pd

from src.data.get_data import get_data_selected_features
from src.plotting.plots import barplot_norm_target, kde_and_boxplots
from src.train.training import start_training, display_metrics, show_feature_importances

from src.evaluate.evaluate import predict_from_input, predict_from_file

CONFIG_PATH = "../config/params.yaml"


def main_page():
    """
    Страница с описанием проекта
    """
    st.image(
        "https://mygenetics.ru/upload/content/a2e1453d6950489a74aa5159a1f5500d.jpg",
        width=600,
    )

    st.title("Alzheimer's Disease Prediction")
    st.markdown("## Описание проекта")
    st.write(
        """
        Предсказание диагноза болезни Альцгеймера по демографическим данным, факторам образа жизни, \
        истории болезни, клиническим измерениям, когнитивным и функциональным оценкам и симптомам."""
    )
    st.markdown(
        """
        ### Target
            - Diagnosis(1 - есть болезнь Альцгеймера, 0 - нет болезни Альцгеймера)
        """
    )

    # name of the columns
    st.markdown(
        """
        ### Описание полей 
            - Age - Возраст
            - Ethnicity - этническая принадлежность
            - Smoking - 1: пациент курит, 0: пациент не курит
            - SleepQuality - показатель качества сна (от 4 до 10)
            - FamilyHistoryAlzheimers - семейный анамнез болезни Альцгеймера(0 - «нет», 1 - «да»)
            - CardiovascularDisease - наличие сердечно-сосудистых заболеваний(0 - «нет», 1 - «да»)
            - Depression - наличие депрессии(0 - «нет», 1 - «да»)
            - HeadInjury - история черепно-мозговой травмы(0 - «нет», 1 - «да») 
            - Hypertension - наличие гипертонии(0 - «нет», 1 - «да»)
            - SystolicBP - систолическое артериальное давление(от 90 до 180 мм рт. ст.)
            - CholesterolLDL - уровень холестерина липопротеинов низкой плотности(от 50 до 200 мг/дл)
            - CholesterolHDL - уровень холестерина липопротеинов высокой плотности(от 20 до 100 мг/дл)
            - MMSE - оценка по краткой шкале оценки психического состояния, от 0 до 30.   
            Более низкие баллы указывают на когнитивные нарушения.
            - FunctionalAssessment - функциональная оценка, от 0 до 10.   
            Более низкие баллы указывают на более серьезные нарушения.
            - MemoryComplaints - наличие жалоб на память(0 - «нет», 1 - «да»)
            - BehavioralProblems - наличие поведенческих нарушений(0 - «нет», 1 - «да»)
            - ADL - оценка активности повседневной жизни, от 0 до 10.   
            Более низкие баллы указывают на более серьезные нарушения.
            - Confusion - наличие спутанности сознания(0 - «нет», 1 - «да»)
            - PersonalityChanges - Наличие изменений личности(0 - «нет», 1 - «да»)
            - DifficultyCompletingTasks - Наличие трудностей с выполнением задач(0 - «нет», 1 - «да»)
    """
    )


def exploratory():
    """
    Exploratory data analysis
    """
    st.markdown("# Exploratory data analysis️")

    with open(CONFIG_PATH) as file:
        config = yaml.load(file, Loader=yaml.FullLoader)

    raw_path = config["preprocessing"]["raw_train_path"]
    raw_data = pd.read_csv(raw_path)
    data_disp = get_data_selected_features(raw_data, **config)

    # вывод датасета с отобранными признаками
    st.write(data_disp.head())
    # вывод исходного датасета (до отбора фичей) при нажатии на кнопку:
    if st.button("Show original raw data"):
        st.write(raw_data.head())

    mmse_target = st.sidebar.checkbox("MMSE and Diagnosis")
    func_asses_target = st.sidebar.checkbox("Functional Assessment and Diagnosis")
    adl_target = st.sidebar.checkbox("ADL and Diagnosis")
    sleep_target = st.sidebar.checkbox("Sleep Quality and Diagnosis")
    memory_target = st.sidebar.checkbox("Memory Complaints and Diagnosis")
    behavior_target = st.sidebar.checkbox("Behavioral Problems and Diagnosis")

    if mmse_target:
        st.pyplot(kde_and_boxplots(data=data_disp, column="MMSE", target="Diagnosis"))
        st.write(
            """У пациентов с болезнью Альцгеймера оценка MMSE ниже, чем у здоровых людей, \
                 что говорит о когнитивных нарушениях."""
        )

    if func_asses_target:
        st.pyplot(
            kde_and_boxplots(
                data=data_disp, column="FunctionalAssessment", target="Diagnosis"
            )
        )
        st.write(
            """У пациентов с болезнью Альцгеймера функциональная оценка ниже, \
                  чем у здоровых людей."""
        )

    if adl_target:
        st.pyplot(kde_and_boxplots(data=data_disp, column="ADL", target="Diagnosis"))
        st.write(
            """У пациентов с болезнью Альцгеймера оценка активности повседневной жизни ниже, \
                  чем у здоровых людей."""
        )

    if sleep_target:
        st.pyplot(
            kde_and_boxplots(data=data_disp, column="SleepQuality", target="Diagnosis")
        )
        st.write(
            """У пациентов с болезнью Альцгеймера качество сна ниже, \
                  чем у здоровых людей."""
        )

    if memory_target:
        st.pyplot(
            barplot_norm_target(
                data=data_disp, column="MemoryComplaints", target="Diagnosis"
            )
        )
        st.write(
            """У пациентов с болезнью Альцгеймера чаще возникают жалобы на память, \
                  чем у здоровых людей."""
        )

    if behavior_target:
        st.pyplot(
            barplot_norm_target(
                data=data_disp, column="BehavioralProblems", target="Diagnosis"
            )
        )
        st.write(
            """У пациентов с болезнью Альцгеймера поведенческие нарушения возникают чаще, \
                  чем у здоровых людей."""
        )


def training():
    """
    Тренировка модели
    """
    st.markdown("# Training model SVC")

    with open(CONFIG_PATH) as file:
        config = yaml.load(file, Loader=yaml.FullLoader)
    # endpoint
    endpoint = config["endpoints"]["train"]
    metrics_path = config["train"]["metrics_path"]
    perm_path = config["permutation_importances"]["permutation_importances_path"]

    tab1, tab2 = st.tabs(["Training", "Training results"])

    with tab1:
        if st.button("Start training"):
            start_training(config=config, endpoint=endpoint)
    with tab2:
        # show metrics
        display_metrics(metrics_path)
        # show feature importances
        show_feature_importances(perm_path)


def prediction_input():
    """
    Получение предсказаний путем ввода данных
    """
    st.markdown("# Prediction")
    with open(CONFIG_PATH) as file:
        config = yaml.load(file, Loader=yaml.FullLoader)
    endpoint = config["endpoints"]["prediction_input"]
    unique_data_path = config["preprocessing"]["unique_values_path"]

    # проверка на наличие сохраненной модели
    if os.path.exists(config["train"]["model_path"]):
        predict_from_input(unique_data_path=unique_data_path, endpoint=endpoint)
    else:
        st.error("Сначала обучите модель")


def prediction_from_file():
    """
    Получение предсказаний из файла с данными
    """
    st.markdown("# Prediction")
    with open(CONFIG_PATH) as file:
        config = yaml.load(file, Loader=yaml.FullLoader)
    endpoint = config["endpoints"]["prediction_from_file"]

    upload_file = st.file_uploader(
        "", type=["csv", "xlsx"], accept_multiple_files=False
    )
    # проверка загружен ли файл
    if upload_file:
        raw_data = pd.read_csv(upload_file)
        data_disp = get_data_selected_features(raw_data, **config)
        st.write(data_disp.head())
        # проверка на наличие сохраненной модели
        if os.path.exists(config["train"]["model_path"]):
            predict_from_file(file=upload_file, data=data_disp, endpoint=endpoint)
        else:
            st.error("Сначала обучите модель")


def main():
    """
    Сборка пайплайна в одном блоке
    """
    page_names_to_funcs = {
        "Описание проекта": main_page,
        "Exploratory data analysis": exploratory,
        "Training model": training,
        "Prediction": prediction_input,
        "Prediction from file": prediction_from_file,
    }
    selected_page = st.sidebar.selectbox("Выберите пункт", page_names_to_funcs.keys())
    page_names_to_funcs[selected_page]()


if __name__ == "__main__":
    main()
