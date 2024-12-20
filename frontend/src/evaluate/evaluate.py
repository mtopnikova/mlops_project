import json
import pandas as pd
import requests
import streamlit as st


def predict_from_input(unique_data_path: str, endpoint: str) -> None:
    """
    Получение входных данных путем ввода в UI -> вывод результата
    :param unique_data_path: путь до json c уникальными значениями
    :param endpoint: endpoint
    """

    with open(unique_data_path) as file:
        unique_data = json.load(file)
    # делаем поля для ввода данных вручную:
    age = st.sidebar.slider(
        "Age", min_value=min(unique_data["Age"]), max_value=max(unique_data["Age"])
    )
    ethnicity = st.sidebar.selectbox("Ethnicity", unique_data["Ethnicity"])
    smoking = st.sidebar.selectbox("Smoking", ["Yes", "No"])
    sleep_quality = st.sidebar.slider(
        "Sleep quality",
        min_value=min(unique_data["SleepQuality"]),
        max_value=max(unique_data["SleepQuality"]),
    )
    fam_hist_alz = st.sidebar.selectbox("Family history Alzheimer's", ["Yes", "No"])
    cardovasc_dis = st.sidebar.selectbox("Cardiovascular disease", ["Yes", "No"])
    depression = st.sidebar.selectbox("Depression", ["Yes", "No"])
    head_injury = st.sidebar.selectbox("Head injury", ["Yes", "No"])
    hypertension = st.sidebar.selectbox("Hypertension", ["Yes", "No"])
    systolic_bp = st.sidebar.slider(
        "Systolic blood pressure",
        min_value=min(unique_data["SystolicBP"]),
        max_value=max(unique_data["SystolicBP"]),
    )
    cholesterol_ldl = st.sidebar.slider(
        "Cholesterol LDL",
        min_value=min(unique_data["CholesterolLDL"]),
        max_value=max(unique_data["CholesterolLDL"]),
    )
    cholesterol_hdl = st.sidebar.slider(
        "Cholesterol HDL",
        min_value=min(unique_data["CholesterolHDL"]),
        max_value=max(unique_data["CholesterolHDL"]),
    )
    mmse = st.sidebar.slider(
        "MMSE", min_value=min(unique_data["MMSE"]), max_value=max(unique_data["MMSE"])
    )
    func_asses = st.sidebar.slider(
        "Functional assessment",
        min_value=min(unique_data["FunctionalAssessment"]),
        max_value=max(unique_data["FunctionalAssessment"]),
    )
    memory_complaints = st.sidebar.selectbox("Memory complaints", ["Yes", "No"])
    behavioral_problems = st.sidebar.selectbox("Behavioral problems", ["Yes", "No"])
    adl = st.sidebar.slider(
        "ADL", min_value=min(unique_data["ADL"]), max_value=max(unique_data["ADL"])
    )
    confusion = st.sidebar.selectbox("Confusion", ["Yes", "No"])
    personality_changes = st.sidebar.selectbox("Personality changes", ["Yes", "No"])
    difficulty_tasks = st.sidebar.selectbox(
        "Difficulty completing tasks", ["Yes", "No"]
    )

    data_dict = {
        "Age": age,
        "Ethnicity": ethnicity,
        "Smoking": smoking,
        "SleepQuality": sleep_quality,
        "FamilyHistoryAlzheimers": fam_hist_alz,
        "CardiovascularDisease": cardovasc_dis,
        "Depression": depression,
        "HeadInjury": head_injury,
        "Hypertension": hypertension,
        "SystolicBP": systolic_bp,
        "CholesterolLDL": cholesterol_ldl,
        "CholesterolHDL": cholesterol_hdl,
        "MMSE": mmse,
        "FunctionalAssessment": func_asses,
        "MemoryComplaints": memory_complaints,
        "BehavioralProblems": behavioral_problems,
        "ADL": adl,
        "Confusion": confusion,
        "PersonalityChanges": personality_changes,
        "DifficultyCompletingTasks": difficulty_tasks,
    }

    # evaluate and return prediction (text)
    if st.button("Predict"):
        result = requests.post(endpoint, timeout=5000, json=data_dict)
        try:
            if int(result.text) == 1:
                st.write("## The patient might have Alzheimer's disease")
            elif int(result.text) == 0:
                st.write("## The patient probably does not have Alzheimer's disease")
            st.success("Success!")
        except:
            st.write("## Error")


def predict_from_file(file: object, data: pd.DataFrame, endpoint: str) -> None:
    """
    Получение входных данных в качестве файла -> вывод результата в виде таблицы
    :param data: датасет
    :param endpoint: endpoint
    :param files:
    """
    if st.button("Predict"):
        # заглушка так как не выводим все предсказания
        data_ = data[:5]
        files = {"file": file.getvalue()}
        output = requests.post(endpoint, files=files, timeout=5000)
        result = json.loads(output.text)
        data_["predict"] = result["prediction"]
        st.write(data_.head())
