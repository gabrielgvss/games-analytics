"""Dados, modelos e avaliação sem dependência da interface."""
from __future__ import annotations

import hashlib
from pathlib import Path

import kagglehub
import numpy as np
import pandas as pd
from sklearn.base import clone
from sklearn.compose import ColumnTransformer, TransformedTargetRegressor
from sklearn.dummy import DummyRegressor
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import KFold, cross_validate, train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

DATASET = "rush4ratio/video-game-sales-with-ratings/versions/2"
FILENAME = "Video_Games_Sales_as_at_22_Dec_2016.csv"
SHA256 = "4ad802a9e51792e7a96b84da4501e5f64c97484d937ede9441d2e1145c1d7b03"
TARGET = "Global_Sales"
SEED = 42
NUM_BASE = ["Year_of_Release", "Critic_Score", "Critic_Count", "User_Score", "User_Count"]
NUM_LOG = ["Year_of_Release", "Critic_Score", "Log_Critic_Count", "User_Score", "Log_User_Count"]
NUM_POLY = NUM_LOG + ["Critic_Score_Quadrado", "User_Score_Quadrado", "Interacao_Critica_Usuario"]
CAT_BASE = ["Platform", "Genre", "Rating"]
CAT_EXT = CAT_BASE + ["Publisher", "Developer"]
NUM_INPUT = ["Year_of_Release", "Critic_Score", "Critic_Count", "User_Score", "User_Count"]


def load_data(path: str | Path | None = None) -> pd.DataFrame:
    """Baixa somente a versão fixada e rejeita alterações silenciosas da base."""
    source = Path(path) if path else Path(kagglehub.dataset_download(DATASET)) / FILENAME
    digest = hashlib.sha256(source.read_bytes()).hexdigest()
    if digest != SHA256:
        raise ValueError(f"Dataset inesperado: SHA256 {digest}; esperado {SHA256}")
    frame = pd.read_csv(source, encoding="cp1252", encoding_errors="replace")
    if frame.shape != (16719, 16):
        raise ValueError(f"Dimensões inesperadas: {frame.shape}")
    return frame


def prepare(frame: pd.DataFrame) -> pd.DataFrame:
    data = frame.copy()
    data["User_Score"] = pd.to_numeric(data["User_Score"].replace("tbd", np.nan), errors="coerce")
    data["Log_Critic_Count"] = np.log1p(data["Critic_Count"])
    data["Log_User_Count"] = np.log1p(data["User_Count"])
    data["Critic_Score_Quadrado"] = data["Critic_Score"] ** 2
    data["User_Score_Quadrado"] = data["User_Score"] ** 2
    data["Interacao_Critica_Usuario"] = data["Critic_Score"] * data["User_Score"]
    return data


def split(data: pd.DataFrame):
    train_idx, test_idx = train_test_split(data.index, test_size=0.2, random_state=SEED)
    return data.loc[train_idx].copy(), data.loc[test_idx].copy()


def preprocessor(numeric: list[str], categorical: list[str]) -> ColumnTransformer:
    numeric_pipe = Pipeline([
        ("imputacao", SimpleImputer(strategy="median", add_indicator=True)),
        ("escala", StandardScaler()),
    ])
    categorical_pipe = Pipeline([
        ("imputacao", SimpleImputer(strategy="most_frequent")),
        ("one_hot", OneHotEncoder(handle_unknown="ignore", min_frequency=20)),
    ])
    return ColumnTransformer([
        ("numericas", numeric_pipe, numeric),
        ("categoricas", categorical_pipe, categorical),
    ])


def pipeline(regressor, numeric, categorical):
    return Pipeline([
        ("preprocessamento", preprocessor(numeric, categorical)),
        ("regressao", regressor),
    ])


def candidates():
    """Ordem estável para relatório e testes."""
    return {
        "Baseline — mediana": (Pipeline([("imputacao", SimpleImputer(strategy="median")), ("regressao", DummyRegressor(strategy="median"))]), ["Critic_Count"]),
        "Linear simples — Critic_Count": (Pipeline([("imputacao", SimpleImputer(strategy="median", add_indicator=True)), ("regressao", LinearRegression())]), ["Critic_Count"]),
        "Linear múltipla — base": (pipeline(LinearRegression(), NUM_BASE, CAT_BASE), NUM_BASE + CAT_BASE),
        "Linear múltipla — Publisher/Developer": (pipeline(LinearRegression(), NUM_BASE, CAT_EXT), NUM_BASE + CAT_EXT),
        "Linear múltipla — contagens em log": (pipeline(LinearRegression(), NUM_LOG, CAT_EXT), NUM_LOG + CAT_EXT),
        "Linear — alvo em log": (TransformedTargetRegressor(regressor=pipeline(LinearRegression(), NUM_LOG, CAT_EXT), func=np.log1p, inverse_func=np.expm1), NUM_LOG + CAT_EXT),
        "Regressão polinomial — grau 2": (pipeline(LinearRegression(), NUM_POLY, CAT_EXT), NUM_POLY + CAT_EXT),
    }


def metrics(actual, predicted) -> dict[str, float]:
    return {
        "RMSE": float(np.sqrt(mean_squared_error(actual, predicted))),
        "MAE": float(mean_absolute_error(actual, predicted)),
        "R2": float(r2_score(actual, predicted)),
    }


def cross_validation(train: pd.DataFrame) -> pd.DataFrame:
    folds = KFold(n_splits=5, shuffle=True, random_state=SEED)
    scoring = {"RMSE": "neg_root_mean_squared_error", "MAE": "neg_mean_absolute_error", "R2": "r2"}
    rows = []
    for name, (model, columns) in candidates().items():
        scores = cross_validate(model, train[columns], train[TARGET], cv=folds, scoring=scoring, return_train_score=True, n_jobs=1)
        rows.append({
            "modelo": name,
            "RMSE_cv": -float(scores["test_RMSE"].mean()),
            "RMSE_cv_dp": float(scores["test_RMSE"].std()),
            "RMSE_treino": -float(scores["train_RMSE"].mean()),
            "MAE_cv": -float(scores["test_MAE"].mean()),
            "R2_cv": float(scores["test_R2"].mean()),
        })
    return pd.DataFrame(rows).sort_values("RMSE_cv").reset_index(drop=True)


def choose_model(cv: pd.DataFrame) -> str:
    best = cv.iloc[0]
    second = cv.iloc[1]
    gain = (second["RMSE_cv"] - best["RMSE_cv"]) / second["RMSE_cv"]
    if "polinomial" in best["modelo"].lower() and gain < 0.005:
        return cv[~cv["modelo"].str.contains("polinomial", case=False)].iloc[0]["modelo"]
    return str(best["modelo"])


def test_all(train: pd.DataFrame, test: pd.DataFrame, selected: str):
    """Executar somente depois de choose_model; o teste não altera a seleção."""
    rows, fitted, predictions = [], {}, {}
    for name, (candidate, columns) in candidates().items():
        model = clone(candidate).fit(train[columns], train[TARGET])
        predicted = model.predict(test[columns])
        rows.append({"modelo": name, **metrics(test[TARGET], predicted)})
        fitted[name], predictions[name] = model, predicted
    return pd.DataFrame(rows).sort_values("RMSE").reset_index(drop=True), fitted[selected], predictions[selected]


def input_to_model(record: dict) -> pd.DataFrame:
    """Converte campos amigáveis do formulário às colunas do modelo final."""
    row = {key: record.get(key) for key in NUM_INPUT + CAT_EXT}
    for key in NUM_INPUT:
        value = row[key]
        row[key] = np.nan if value is None or value == "" else float(value)
    for key in CAT_EXT:
        if row[key] is None or row[key] == "":
            row[key] = np.nan
    row["Log_Critic_Count"] = np.log1p(row["Critic_Count"])
    row["Log_User_Count"] = np.log1p(row["User_Count"])
    return pd.DataFrame([row])[NUM_LOG + CAT_EXT]
