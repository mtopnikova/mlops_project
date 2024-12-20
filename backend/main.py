import pandas as pd

import uvicorn
from fastapi import FastAPI
from fastapi import File
from fastapi import UploadFile
from pydantic import BaseModel

from src.pipeline.pipeline import pipeline_training
from src.evaluate.evaluate import pipeline_evaluate

# import warnings
# warnings.filterwarnings("ignore")

app = FastAPI()
CONFIG_PATH = "../config/params.yaml"


class Patient(BaseModel):
    """
    Признаки для получения результатов модели
    """

    Age: int
    Ethnicity: str
    Smoking: str
    SleepQuality: float
    FamilyHistoryAlzheimers: str
    CardiovascularDisease: str
    Depression: str
    HeadInjury: str
    Hypertension: str
    SystolicBP: int
    CholesterolLDL: float
    CholesterolHDL: float
    MMSE: float
    FunctionalAssessment: float
    MemoryComplaints: str
    BehavioralProblems: str
    ADL: float
    Confusion: str
    PersonalityChanges: str
    DifficultyCompletingTasks: str


@app.post("/train")
def training():
    """
    Обучение модели, логирование метрик
    """
    pipeline_training(config_path=CONFIG_PATH)


@app.post("/predict")
def prediction(file: UploadFile = File(...)):
    """
    Предсказание модели по данным из файла
    """
    result = pipeline_evaluate(config_path=CONFIG_PATH, data_path=file.file)
    assert isinstance(result, list), "Результат не соответствует типу list"
    # заглушка так как не выводим все предсказания, иначе зависнет
    return {"prediction": result[:5]}


@app.post("/predict_input")
def prediction_input(patient: Patient):
    """
    Предсказание модели по введенным данным
    """
    features = [
        [
            patient.Age,
            patient.Ethnicity,
            patient.Smoking,
            patient.SleepQuality,
            patient.FamilyHistoryAlzheimers,
            patient.CardiovascularDisease,
            patient.Depression,
            patient.HeadInjury,
            patient.Hypertension,
            patient.SystolicBP,
            patient.CholesterolLDL,
            patient.CholesterolHDL,
            patient.MMSE,
            patient.FunctionalAssessment,
            patient.MemoryComplaints,
            patient.BehavioralProblems,
            patient.ADL,
            patient.Confusion,
            patient.PersonalityChanges,
            patient.DifficultyCompletingTasks,
        ]
    ]

    cols = [
        "Age",
        "Ethnicity",
        "Smoking",
        "SleepQuality",
        "FamilyHistoryAlzheimers",
        "CardiovascularDisease",
        "Depression",
        "HeadInjury",
        "Hypertension",
        "SystolicBP",
        "CholesterolLDL",
        "CholesterolHDL",
        "MMSE",
        "FunctionalAssessment",
        "MemoryComplaints",
        "BehavioralProblems",
        "ADL",
        "Confusion",
        "PersonalityChanges",
        "DifficultyCompletingTasks",
    ]

    data = pd.DataFrame(features, columns=cols)
    data.replace({"Yes": 1, "No": 0}, inplace=True)
    predictions = pipeline_evaluate(config_path=CONFIG_PATH, dataset=data)[0]

    return predictions


if __name__ == "__main__":
    # Запустите сервер, используя заданный хост и порт
    uvicorn.run(app, host="127.0.0.1", port=80)
