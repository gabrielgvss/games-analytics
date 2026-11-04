"""Treina, avalia e gera resultados e interface offline em uma execução."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
from statsmodels.stats.outliers_influence import variance_inflation_factor

from .core import (
    CAT_EXT, NUM_INPUT, NUM_LOG, SEED, TARGET, choose_model, cross_validation,
    input_to_model, load_data, metrics, prepare, split, test_all,
)

ROOT = Path(__file__).resolve().parents[2]


def model_rows(records: list[dict]) -> pd.DataFrame:
    rows = []
    for record in records:
        row = input_to_model(record).iloc[0].to_dict()
        rows.append(row)
    return pd.DataFrame(rows)[NUM_LOG + CAT_EXT]


def portable_model(model, train: pd.DataFrame) -> dict:
    """Fatora a função linear em contribuições independentes por campo de entrada."""
    ref = {}
    for field in NUM_INPUT:
        ref[field] = float(train[field].median())
    for field in CAT_EXT:
        ref[field] = str(train[field].mode().iloc[0])
    baseline = float(model.predict(input_to_model(ref))[0])
    numeric = {}
    for field in NUM_INPUT:
        plus = ref.copy()
        plus[field] = np.expm1(np.log1p(ref[field]) + 1) if field.endswith("_Count") else ref[field] + 1
        missing = ref.copy()
        missing[field] = None
        slope = float(model.predict(input_to_model(plus))[0] - baseline)
        missing_effect = float(model.predict(input_to_model(missing))[0] - baseline)
        numeric[field] = {"reference": ref[field], "slope": slope, "missing": missing_effect, "log": field.endswith("_Count")}
    categorical = {}
    options = {}
    for field in CAT_EXT:
        frequencies = train[field].value_counts(dropna=True)
        values = [str(value) for value in frequencies.index]
        changed = []
        for value in values + ["__UNKNOWN__", None]:
            row = ref.copy()
            row[field] = value
            changed.append(row)
        predicted = model.predict(model_rows(changed))
        effects = {value: float(predicted[i] - baseline) for i, value in enumerate(values)}
        effects["__UNKNOWN__"] = float(predicted[-2] - baseline)
        effects["__MISSING__"] = float(predicted[-1] - baseline)
        categorical[field] = effects
        options[field] = values[:60]
    ranges = {
        field: {"min": float(train[field].min()), "max": float(train[field].max())}
        for field in NUM_INPUT
    }
    return {
        "baseline": baseline,
        "reference": ref,
        "numeric": numeric,
        "categorical": categorical,
        "options": options,
        "context": {"numeric_ranges": ranges},
    }


def predict_portable(spec: dict, record: dict) -> tuple[float, dict]:
    effects = {}
    for field, item in spec["numeric"].items():
        value = record.get(field)
        if value is None or value == "":
            effects[field] = item["missing"]
        else:
            value = float(value)
            ref = item["reference"]
            effects[field] = item["slope"] * ((np.log1p(value) - np.log1p(ref)) if item["log"] else (value - ref))
    for field, mapping in spec["categorical"].items():
        value = record.get(field)
        key = "__MISSING__" if value is None or value == "" else str(value)
        effects[field] = mapping.get(key, mapping["__UNKNOWN__"])
    return spec["baseline"] + sum(effects.values()), effects


def collinearity(train: pd.DataFrame) -> list[dict]:
    numeric = train[NUM_LOG].fillna(train[NUM_LOG].median())
    matrix = np.column_stack([np.ones(len(numeric)), numeric.to_numpy(dtype=float)])
    return [{"atributo": column, "VIF": float(variance_inflation_factor(matrix, i + 1))} for i, column in enumerate(NUM_LOG)]


def report(train, test, cv, comparison, selected, raw, spec):
    actual = test[TARGET]
    clipped = np.clip(raw, 0, None)
    evaluation = test[["Name", "Platform", "Genre", TARGET]].copy()
    evaluation["previsao"] = clipped
    evaluation["erro"] = (actual - clipped).abs()
    wii = evaluation[evaluation["Name"].eq("Wii Sports") & evaluation["Platform"].eq("Wii")]
    if len(wii) != 1:
        raise ValueError("Wii Sports não foi encontrado exatamente uma vez no teste")
    wii_idx = wii.index[0]
    without_wii = actual.index != wii_idx
    deciles = pd.DataFrame({"real": actual, "previsto": clipped})
    deciles["decil"] = pd.qcut(deciles["real"], 10, duplicates="drop")
    decile_summary = deciles.groupby("decil", observed=True).agg(real=("real", "mean"), previsto=("previsto", "mean"), n=("real", "size"))
    examples_idx = [wii_idx]
    for target in (0.01, 0.29, 10.0):
        index = (actual - target).abs().sort_values().index
        examples_idx.append(next(i for i in index if i not in examples_idx))
    examples = []
    for idx in examples_idx:
        row = test.loc[idx]
        record = {}
        for field in NUM_INPUT + CAT_EXT:
            value = row[field]
            record[field] = None if pd.isna(value) else (float(value) if field in NUM_INPUT else str(value))
        predicted, _ = predict_portable(spec, record)
        examples.append({"name": str(row["Name"]), "record": record, "actual": float(row[TARGET]), "predicted": max(0, predicted)})
    catalog = []
    for _, row in test.iterrows():
        if pd.isna(row["Name"]):
            continue
        record = {}
        for field in NUM_INPUT + CAT_EXT:
            value = row[field]
            record[field] = None if pd.isna(value) else (float(value) if field in NUM_INPUT else str(value))
        catalog.append({
            "name": str(row["Name"]),
            "platform": str(row["Platform"]),
            "year": None if pd.isna(row["Year_of_Release"]) else int(row["Year_of_Release"]),
            "actual": float(row[TARGET]),
            "record": record,
        })
    catalog.sort(key=lambda item: (item["name"].casefold(), item["platform"], item["year"] or 0))
    effects = []
    for _, row in test.iterrows():
        record = {field: (None if pd.isna(row[field]) else row[field]) for field in NUM_INPUT + CAT_EXT}
        _, local = predict_portable(spec, record)
        effects.append(local)
    importance = pd.DataFrame(effects).abs().mean().sort_values(ascending=False)
    corr = train[["Critic_Count", "User_Count", "Critic_Score", "User_Score"]].corr(method="spearman")
    return {
        "dataset": {"rows": int(len(train) + len(test)), "train": int(len(train)), "test": int(len(test)), "version": 2, "seed": SEED},
        "selected": selected,
        "cv": cv.to_dict(orient="records"),
        "test_models": comparison.to_dict(orient="records"),
        "final_raw": metrics(actual, raw),
        "final_clipped": metrics(actual, clipped),
        "negative_count": int((raw < 0).sum()),
        "negative_pct": float((raw < 0).mean() * 100),
        "wii": {"actual": float(wii.iloc[0][TARGET]), "predicted": float(wii.iloc[0]["previsao"]), "residual": float(wii.iloc[0][TARGET] - wii.iloc[0]["previsao"]), "rmse_without": metrics(actual[without_wii], raw[without_wii])["RMSE"]},
        "deciles": decile_summary.reset_index(drop=True).to_dict(orient="records"),
        "vif": collinearity(train),
        "spearman": {"critic_user_counts": float(corr.loc["Critic_Count", "User_Count"]), "critic_user_scores": float(corr.loc["Critic_Score", "User_Score"])},
        "importance": importance.rename_axis("field").reset_index(name="mean_abs_effect").to_dict(orient="records"),
        "examples": examples,
        "catalog": catalog,
    }


def build_html(payload: dict, output: Path):
    template = (ROOT / "web" / "index.template.html").read_text(encoding="utf-8")
    style = (ROOT / "web" / "style.css").read_text(encoding="utf-8")
    script = (ROOT / "web" / "app.js").read_text(encoding="utf-8")
    data = json.dumps(payload, ensure_ascii=False, separators=(",", ":"), allow_nan=False).replace("</", "<\\/")
    result = template.replace("/*__STYLE__*/", style).replace("/*__DATA__*/", data).replace("/*__SCRIPT__*/", script)
    output.write_text(result, encoding="utf-8")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path, help="CSV local da versão 2; por padrão, baixar pelo kagglehub")
    args = parser.parse_args()
    frame = load_data(args.data)
    train, test = split(prepare(frame))
    cv = cross_validation(train)
    selected = choose_model(cv)
    comparison, model, raw = test_all(train, test, selected)
    if selected != "Linear múltipla — contagens em log":
        raise ValueError(f"Exportação offline para {selected} ainda não implementada")
    spec = portable_model(model, train)
    report_data = report(train, test, cv, comparison, selected, raw, spec)
    payload = {"report": report_data, "model": spec}
    artifact_dir = ROOT / "artifacts"
    artifact_dir.mkdir(exist_ok=True)
    (artifact_dir / "results.json").write_text(json.dumps(payload, ensure_ascii=False, indent=2, allow_nan=False), encoding="utf-8")
    build_html(payload, ROOT / "index.html")
    print(json.dumps({"selected": selected, "test_raw": report_data["final_raw"], "negative_pct": report_data["negative_pct"], "html": "index.html"}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
